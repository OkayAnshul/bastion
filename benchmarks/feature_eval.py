"""Time online feature evaluation on fixed Redis snapshots, for before/after comparisons.

Phase 3 profiling found feature evaluation to be the largest CPU cost inside the scoring handler.
This script isolates it from HTTP, Redis and load-generator noise: it fetches snapshots once, then
times ``features_from_snapshot`` on exactly those snapshots under whichever version of the code is
importable. It uses only APIs that existed before the optimisation, so an older revision checked
out with ``git worktree add`` can be timed the same way:

    # Redis populated by `bastion bench prepare`
    uv run python benchmarks/feature_eval.py fetch artifacts/bench/snapshots.pkl
    uv run python benchmarks/feature_eval.py time artifacts/bench/snapshots.pkl --label after
    PYTHONPATH=<worktree>/src uv run python benchmarks/feature_eval.py \
        time artifacts/bench/snapshots.pkl --label before
    cmp artifacts/bench/snapshots.before.features.json artifacts/bench/snapshots.after.features.json

Each ``time`` run writes its feature values next to the snapshots, so ``cmp`` confirms that the two
versions compute identical features.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

import bastion
from bastion.features.online import features_from_snapshot

LABEL_STRENGTH = 100.0


def fetch(out: Path, payloads: Path, redis_url: str, count: int) -> None:
    import redis

    from bastion.features.online import OnlineFeatureStore
    from bastion.schemas.events import TransactionEvent
    from bastion.streaming.config import load_streaming_config

    client = redis.Redis.from_url(redis_url)
    store = OnlineFeatureStore(client, load_streaming_config().online_store)
    requests = json.loads(payloads.read_text())[:count]
    events = [TransactionEvent.model_validate(request) for request in requests]
    out.write_bytes(pickle.dumps([(event, store.snapshot(event)) for event in events]))
    sys.stdout.write(f"fetched {len(events)} snapshots into {out}\n")


def time_evaluation(snapshots: Path, label: str, repeats: int) -> None:
    pairs = pickle.loads(snapshots.read_bytes())  # written by `fetch` above, not untrusted input
    outputs = [features_from_snapshot(e, s, label_strength=LABEL_STRENGTH) for e, s in pairs]
    per_call = []
    for _ in range(repeats):
        for event, snapshot in pairs:
            started = time.perf_counter_ns()
            features_from_snapshot(event, snapshot, label_strength=LABEL_STRENGTH)
            per_call.append((time.perf_counter_ns() - started) / 1e6)
    samples = np.asarray(per_call)
    p50, p95, p99 = np.percentile(samples, [50, 95, 99])
    result = {
        "label": label,
        "code": bastion.__file__,
        "snapshots": len(pairs),
        "calls": int(samples.size),
        "mean_ms": round(float(samples.mean()), 4),
        "p50_ms": round(float(p50), 4),
        "p95_ms": round(float(p95), 4),
        "p99_ms": round(float(p99), 4),
    }
    stem = snapshots.with_suffix("")
    Path(f"{stem}.{label}.features.json").write_text(
        json.dumps(outputs, default=str, sort_keys=True)
    )
    Path(f"{stem}.{label}.timing.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(json.dumps(result) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fetch_cmd = commands.add_parser("fetch", help="Read snapshots for benchmark payloads.")
    fetch_cmd.add_argument("out", type=Path)
    fetch_cmd.add_argument("--payloads", type=Path, default=Path("artifacts/bench/payloads.json"))
    fetch_cmd.add_argument("--redis-url", default="redis://localhost:6380/0")
    fetch_cmd.add_argument("--count", type=int, default=2000)
    time_cmd = commands.add_parser("time", help="Time feature evaluation on saved snapshots.")
    time_cmd.add_argument("snapshots", type=Path)
    time_cmd.add_argument("--label", required=True)
    time_cmd.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch(args.out, args.payloads, args.redis_url, args.count)
    else:
        time_evaluation(args.snapshots, args.label, args.repeats)


if __name__ == "__main__":
    main()

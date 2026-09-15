"""Scoring-service latency benchmark (ROADMAP Phase 3).

Measures ``POST /v1/score`` end to end with k6 at fixed arrival rates, and joins the client-side
latencies with the service's own per-stage timings from its decision log. Every number in the report
comes from one run, on the machine the report describes.

The setup this assumes (driven by ``bastion bench``):

1. Redis holds the online state for the first 90% of an event table: the events, plus the labels
   that had arrived by the cutoff. The last 10% become request payloads, each scored as of its own
   event time, so reads see realistic history sizes.
2. The scoring service runs against that Redis and writes a JSONL decision log.
3. k6 runs in a container on the host network, one run per arrival rate, after a warm-up run.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
from matplotlib.axis import Axis
from matplotlib.ticker import NullFormatter

from bastion.data.labels import LabelDelayConfig, label_events
from bastion.features.online import OnlineFeatureStore
from bastion.plotting import BASELINE, INK_MUTED, SERIES, label_axes, new_figure, style_axes
from bastion.provenance import git_revision
from bastion.schemas.decisions import DecisionRecord
from bastion.schemas.tables import row_to_event
from bastion.streaming.replay import StoreSink, replay, timeline

K6_IMAGE = "docker.io/grafana/k6:1.8.1"
K6_SCRIPT = Path("benchmarks/k6/score.js")
BUDGET_P99_MS = 50.0  # ARCHITECTURE.md §3.3
STAGE_BUDGETS_MS = {"redis_ms": 12.0, "features_ms": 4.0, "model_ms": 8.0}
REPORT_STEM = "latency"
FIGURES = ("latency_vs_rate.png", "latency_histogram.png")
LOG_TICKS_MS = (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)


def prepare_benchmark(
    events: pl.DataFrame,
    store: OnlineFeatureStore,
    label_config: LabelDelayConfig,
    *,
    request_share: float = 0.1,
    max_requests: int = 20_000,
) -> tuple[int, int, list[dict[str, Any]]]:
    """Ingest history into the store; return (history events, history labels, request payloads)."""
    cutoff = int(events.height * (1 - request_share))
    history = events.head(cutoff)
    cutoff_ts = events["event_ts"][cutoff]
    labels = label_events(history, label_config).filter(pl.col("label_ts") < cutoff_ts)
    replay(timeline(history, labels), StoreSink(store))
    requests = events.slice(cutoff).head(max_requests)
    payloads = [row_to_event(row).model_dump(mode="json") for row in requests.iter_rows(named=True)]
    return history.height, labels.height, payloads


def container_runtime() -> str:
    for candidate in ("docker", "podman"):
        if shutil.which(candidate):
            return candidate
    raise RuntimeError("the latency benchmark runs k6 in a container: install docker or podman")


@dataclass(frozen=True)
class RateResult:
    target_rps: int
    duration_s: int
    achieved_rps: float
    requests: int
    failed_rate: float
    dropped_iterations: int
    client_ms: dict[str, float]  # from k6: avg, med, p(90), p(95), p(99), max
    server_ms: dict[str, dict[str, float]]  # stage -> p50 / p95 / p99 / max
    # HTTP status -> requests, from k6's samples; "0" means no response.
    status_counts: dict[str, int] = field(default_factory=dict)

    @property
    def sustained(self) -> bool:
        """Every scheduled request started, none failed, and the arrival rate held."""
        return (
            self.dropped_iterations == 0
            and self.failed_rate == 0
            and self.achieved_rps >= 0.98 * self.target_rps
        )


def _run_k6(
    runtime: str,
    *,
    target: str,
    payloads: Path,
    rate: int,
    duration_s: int,
    raw_dir: Path,
    name: str,
) -> dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    command = [
        runtime, "run", "--rm", "--network", "host", "--user", "root",
        "-v", f"{K6_SCRIPT.parent.resolve()}:/scripts:ro",
        "-v", f"{payloads.parent.resolve()}:/data:ro",
        "-v", f"{raw_dir.resolve()}:/out",
        "-e", f"TARGET={target}", "-e", f"PAYLOADS=/data/{payloads.name}",
        "-e", f"RATE={rate}", "-e", f"DURATION={duration_s}s",
        "-e", f"SUMMARY=/out/{name}.json",
        K6_IMAGE, "run", "--quiet", "--out", f"csv=/out/{name}.csv", f"/scripts/{K6_SCRIPT.name}",
    ]  # fmt: skip
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    summary: dict[str, Any] = json.loads((raw_dir / f"{name}.json").read_text())
    return summary


def _client_latencies(csv_path: Path) -> npt.NDArray[np.float64]:
    with csv_path.open() as fh:
        values = [
            float(row["metric_value"])
            for row in csv.DictReader(fh)
            if row["metric_name"] == "http_req_duration"
        ]
    return np.asarray(values, dtype=np.float64)


def _status_counts(csv_path: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            if row["metric_name"] == "http_reqs":
                counts[row["status"]] += 1
    return dict(sorted(counts.items()))


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as fh:
        return sum(1 for _ in fh)


def _decisions_after(path: Path, skip: int) -> list[DecisionRecord]:
    with path.open() as fh:
        lines = fh.readlines()[skip:]
    return [DecisionRecord.model_validate_json(line) for line in lines]


def _percentiles(values: npt.ArrayLike) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {}
    p50, p95, p99 = np.percentile(array, [50, 95, 99])
    return {"p50": float(p50), "p95": float(p95), "p99": float(p99), "max": float(array.max())}


def run_latency_benchmark(
    *,
    target: str,
    payloads: Path,
    decision_log: Path,
    rates: list[int],
    duration_s: int,
    warmup_s: int,
    out_dir: Path,
) -> list[RateResult]:
    runtime = container_runtime()
    raw_dir = out_dir / "raw"
    _run_k6(runtime, target=target, payloads=payloads, rate=rates[0], duration_s=warmup_s,
            raw_dir=raw_dir, name="warmup")  # fmt: skip
    results = []
    for rate in rates:
        before = _line_count(decision_log)
        summary = _run_k6(runtime, target=target, payloads=payloads, rate=rate,
                          duration_s=duration_s, raw_dir=raw_dir, name=f"rate_{rate}")  # fmt: skip
        time.sleep(2.0)  # the decision logger flushes in the background every 0.5 s
        records = _decisions_after(decision_log, before)
        metrics = summary["metrics"]
        results.append(
            RateResult(
                target_rps=rate,
                duration_s=duration_s,
                achieved_rps=float(metrics["http_reqs"]["values"]["rate"]),
                requests=int(metrics["http_reqs"]["values"]["count"]),
                failed_rate=float(metrics["http_req_failed"]["values"]["rate"]),
                dropped_iterations=int(
                    metrics.get("dropped_iterations", {}).get("values", {}).get("count", 0)
                ),
                client_ms={k: float(v) for k, v in metrics["http_req_duration"]["values"].items()},
                server_ms={
                    stage: _percentiles([getattr(r.timings, stage) for r in records])
                    for stage in ("redis_ms", "features_ms", "model_ms", "total_ms")
                },
                status_counts=_status_counts(raw_dir / f"rate_{rate}.csv"),
            )
        )
    return results


def hardware_description() -> dict[str, str]:
    model = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    memory = "unknown"
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        kib = int(meminfo.read_text().split()[1])
        memory = f"{kib / 1024**2:.1f} GiB"
    description = {
        "cpu": model,
        "logical_cpus": str(os.cpu_count()),
        "memory": memory,
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }
    governor = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if governor.exists():
        description["cpu_governor"] = governor.read_text().strip()
    for supply in sorted(Path("/sys/class/power_supply").glob("A*/online")):
        description["on_ac_power"] = "yes" if supply.read_text().strip() == "1" else "no"
    return description


# ------------------------------------------------------------------------------ report


def _plain_log_ticks(axis: Axis, lower: float, upper: float) -> None:
    """1-2-5 ticks labelled in plain milliseconds on a log axis, not scientific notation."""
    ticks = [t for t in LOG_TICKS_MS if lower <= t <= upper]
    axis.set_ticks(ticks, [f"{t:g}" for t in ticks])
    axis.set_minor_formatter(NullFormatter())


def _plot_latency_vs_rate(results: list[RateResult], path: Path) -> None:
    fig = new_figure(7, 4.5)
    ax = fig.add_subplot()
    rates = [r.target_rps for r in results]
    series = (("med", "p50", SERIES[0]), ("p(95)", "p95", SERIES[1]), ("p(99)", "p99", SERIES[2]))
    for key, label, color in series:
        ax.plot(rates, [r.client_ms[key] for r in results], color=color, linewidth=2, marker="o",
                markersize=6, label=label)  # fmt: skip
    ax.axhline(BUDGET_P99_MS, color=BASELINE, linewidth=1)
    ax.text(rates[0], BUDGET_P99_MS, " p99 budget 50 ms", color=INK_MUTED, fontsize=8, va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xticks(rates, [str(r) for r in rates])
    ax.set_yscale("log")
    values = [r.client_ms[key] for r in results for key, _, _ in series]
    lower, upper = min(values) * 0.8, max(*values, BUDGET_P99_MS) * 1.25
    ax.set_ylim(lower, upper)
    _plain_log_ticks(ax.yaxis, lower, upper)
    label_axes(ax, "Target arrival rate (requests/s)", "Client-side latency (ms, log scale)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED, loc="upper left")
    style_axes(ax, "POST /v1/score latency by load", grid="both")
    fig.savefig(path)


def _plot_histogram(samples: npt.NDArray[np.float64], rate: int, path: Path) -> None:
    fig = new_figure(7, 4)
    ax = fig.add_subplot()
    # Log-spaced bins on a log axis show the body, the whole tail and the budget in one frame; on a
    # linear axis reaching the budget, a millisecond-scale body collapses into a few bars.
    lower = max(float(samples.min()) * 0.9, 0.1)
    upper = max(float(samples.max()), BUDGET_P99_MS) * 1.15
    counts = ax.hist(samples, bins=np.geomspace(lower, upper, 70).tolist(), color=SERIES[0],
                     edgecolor="white", linewidth=0.4)[0]  # fmt: skip
    ax.set_ylim(0, float(np.max(counts)) * 1.3)  # headroom for the marker labels
    ax.set_xscale("log")
    p50, p99 = (float(v) for v in np.percentile(samples, [50, 99]))
    markers = (
        (p50, f" p50 {p50:.1f} ms", INK_MUTED, 0.95),
        (p99, f" p99 {p99:.1f} ms", INK_MUTED, 0.89),
        (BUDGET_P99_MS, " budget 50 ms", BASELINE, 0.83),  # own height: p50 or p99 can be close
    )
    for value, label, color, height in markers:
        ax.axvline(value, color=color, linewidth=1)
        ax.text(value, height, label, transform=ax.get_xaxis_transform(), color=INK_MUTED,
                fontsize=8)  # fmt: skip
    _plain_log_ticks(ax.xaxis, lower, upper)
    label_axes(ax, "Client-side latency (ms, log scale)", "Requests per bin")
    style_axes(ax, f"Latency distribution at {rate} requests/s ({samples.size:,} requests)")
    fig.savefig(path)


def write_latency_report(
    results: list[RateResult],
    out_dir: Path,
    *,
    setup: dict[str, Any],
    hardware: dict[str, str],
    revision: str | None = None,
) -> Path:
    """Write ``latency.md``, ``latency.json`` and figures; ``revision`` defaults to HEAD's."""
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    sustained = [r for r in results if r.sustained]
    headline = sustained[-1] if sustained else results[0]
    samples = _client_latencies(out_dir / "raw" / f"rate_{headline.target_rps}.csv")
    _plot_latency_vs_rate(results, figures / FIGURES[0])
    _plot_histogram(samples, headline.target_rps, figures / FIGURES[1])

    revision = revision or git_revision()
    payload = {
        "git_revision": revision,
        "hardware": hardware,
        "setup": setup,
        "results": [asdict(r) | {"sustained": r.sustained} for r in results],
    }
    (out_dir / f"{REPORT_STEM}.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")

    lines = [
        "# Scoring latency: POST /v1/score",
        "",
        f"Measured by `bastion bench latency` at git revision `{revision}`. Client and server ran"
        " on the same laptop, so the client competes with the service for CPU.",
        "",
        "## Machine",
        "",
        "| | |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in hardware.items()],
        "",
        "## Setup",
        "",
        "| | |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in setup.items()],
        "",
        "## Results by arrival rate",
        "",
        "Client latency is measured by k6 from request start to response end; server stages come"
        " from the decision log of the same requests. **Sustained** means every scheduled"
        " request started, none failed, and the achieved rate stayed within 2% of the target. HTTP"
        " status 0 means k6 received no response (connection reset or timeout).",
        "",
        "| Target rps | Achieved rps | Sustained | Dropped | Errors | HTTP status | p50 ms | p95 ms"
        " | p99 ms | max ms | server p99 ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        c = r.client_ms
        statuses = " · ".join(f"{code}: {n:,}" for code, n in r.status_counts.items()) or "n/a"
        lines.append(
            f"| {r.target_rps} | {r.achieved_rps:.1f} | {'yes' if r.sustained else 'no'}"
            f" | {r.dropped_iterations:,} | {100 * r.failed_rate:.2f}% | {statuses}"
            f" | {c['med']:.2f} | {c['p(95)']:.2f} | {c['p(99)']:.2f} | {c['max']:.2f}"
            f" | {r.server_ms['total_ms'].get('p99', float('nan')):.2f} |"
        )
    within = headline.client_ms["p(99)"] < BUDGET_P99_MS
    lines += [
        "",
        f"![Latency by arrival rate](figures/{FIGURES[0]})",
        "",
        f"## Stage breakdown at {headline.target_rps} requests/s",
        "",
        f"Client p99 **{headline.client_ms['p(99)']:.2f} ms** against the 50 ms budget:"
        f" {'within' if within else 'over'} budget at this rate.",
        "",
        "| Stage | Budget (p99) | p50 ms | p95 ms | p99 ms | max ms |",
        "|---|---|---|---|---|---|",
    ]
    names = {
        "redis_ms": "Redis round trip + event-loop wait",
        "features_ms": "Feature evaluation + row",
        "model_ms": "LightGBM + calibration",
        "total_ms": "Handler total",
    }
    for stage, label in names.items():
        s = headline.server_ms[stage]
        budget = STAGE_BUDGETS_MS.get(stage)
        budget_cell = f"{budget:g} ms" if budget is not None else "n/a"
        lines.append(
            f"| {label} | {budget_cell} | {s['p50']:.2f} | {s['p95']:.2f} | {s['p99']:.2f}"
            f" | {s['max']:.2f} |"
        )
    lines += [
        "",
        "The Redis stage runs from the start of the handler until the pipeline's replies have been"
        " processed. With one event-loop thread per worker, it therefore also includes time a"
        " request waits while other requests run feature evaluation, which is why its tail grows"
        " with load.",
        "",
        f"![Latency distribution](figures/{FIGURES[1]})",
        "",
        "The handler total excludes HTTP parsing, validation and response serialisation; the gap"
        " between it and the client-side latency is the rest of the request path, plus the"
        " client's own overhead on a shared machine.",
    ]
    path = out_dir / f"{REPORT_STEM}.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def rebuild_latency_report(out_dir: Path) -> Path:
    """Re-render a recorded run's report and figures from its JSON and raw k6 samples.

    For presentation changes only: every number, and the git revision the run was measured at, comes
    from the recording.
    """
    payload = json.loads((out_dir / f"{REPORT_STEM}.json").read_text())
    names = {item.name for item in fields(RateResult)}
    results = []
    for row in payload["results"]:
        result = RateResult(**{key: value for key, value in row.items() if key in names})
        samples = out_dir / "raw" / f"rate_{result.target_rps}.csv"
        if not result.status_counts and samples.exists():
            # Recorded before reports counted status codes: derive them from the same k6 samples.
            result = replace(result, status_counts=_status_counts(samples))
        results.append(result)
    return write_latency_report(
        results,
        out_dir,
        setup=payload["setup"],
        hardware=payload["hardware"],
        revision=payload["git_revision"],
    )

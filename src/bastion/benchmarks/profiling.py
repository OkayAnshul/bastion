"""Summaries of py-spy profiles recorded with ``--format raw`` (one collapsed stack per line).

Each line is ``frame;frame;...;leaf count``, where a frame reads ``function (path:line)``. py-spy
drops idle samples by default, so a sample means the thread was doing work when it was taken.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Stage -> frame prefix ("function (path"). A sample counts toward a stage when any frame of its
# stack starts with the prefix, once per stack, so nested stages overlap by design.
SCORING_STAGES: dict[str, str] = {
    "score handler": "score (bastion/serving/app.py",
    "Redis pipeline": "execute (redis/asyncio/client.py",
    "features_from_snapshot": "features_from_snapshot (bastion/features/online.py",
    "window aggregates": "window_aggregates",
    "distinct counts": "distinct_in_window",
    "LightGBM predict": "predict (lightgbm/basic.py",
    "decision log writer": "run (bastion/serving/decision_log.py",
}


@dataclass(frozen=True)
class ProfileSummary:
    samples: int
    stage_samples: dict[str, int]
    leaves: list[tuple[str, int]]  # hottest leaf frames, most samples first

    def share(self, stage: str) -> float:
        return self.stage_samples[stage] / self.samples

    def ms_per_request(self, stage: str, *, rate_hz: float, requests: int) -> float:
        """Sampled busy time in ``stage`` per request: samples x sampling interval / requests."""
        return self.stage_samples[stage] * 1000.0 / rate_hz / requests


def read_collapsed(path: Path) -> list[tuple[list[str], int]]:
    stacks = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        stack, _, count = line.rpartition(" ")  # frames may contain spaces; the count cannot
        stacks.append((stack.split(";"), int(count)))
    return stacks


def summarize_profile(
    path: Path, stages: dict[str, str] | None = None, *, top: int = 10
) -> ProfileSummary:
    stages = SCORING_STAGES if stages is None else stages
    stacks = read_collapsed(path)
    total = sum(count for _, count in stacks)
    if total == 0:
        raise ValueError(f"{path} contains no samples")
    stage_samples = dict.fromkeys(stages, 0)
    leaves: Counter[str] = Counter()
    for frames, count in stacks:
        for stage, prefix in stages.items():
            if any(frame.startswith(prefix) for frame in frames):
                stage_samples[stage] += count
        leaves[frames[-1]] += count
    return ProfileSummary(
        samples=total, stage_samples=stage_samples, leaves=leaves.most_common(top)
    )

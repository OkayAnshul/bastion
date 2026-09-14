"""MLflow tracking: every experiment is logged from its first run (ROADMAP Phase 1)."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import mlflow
import polars as pl


def configure_tracking(uri: str) -> None:
    """Point MLflow at ``uri``; for a local sqlite store, create its directory first."""
    if uri.startswith("sqlite:///"):
        Path(uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)


@contextmanager
def tracked_run(
    experiment: str,
    run_name: str,
    tags: Mapping[str, str],
    *,
    artifact_root: Path | None = None,
    nested: bool = False,
) -> Iterator[str]:
    """Start an MLflow run, yielding its id. New experiments keep artifacts in ``artifact_root``."""
    if mlflow.get_experiment_by_name(experiment) is None:
        location = None
        if artifact_root is not None:
            artifact_root.mkdir(parents=True, exist_ok=True)
            location = artifact_root.resolve().as_uri()
        mlflow.create_experiment(experiment, artifact_location=location)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name, tags=dict(tags), nested=nested) as active:
        yield str(active.info.run_id)


def log_metrics(metrics: Mapping[str, object], prefix: str = "") -> None:
    """Log the finite numeric values; undefined metrics (NaN) are skipped, not logged as zero."""
    finite = {
        f"{prefix}{name}": float(value)
        for name, value in metrics.items()
        if isinstance(value, int | float) and math.isfinite(value)
    }
    if finite:
        mlflow.log_metrics(finite)


def data_fingerprint(events: pl.DataFrame) -> str:
    """Cheap identity for an event table: changes if rows, time range, amounts or labels change."""
    summary = events.select(
        pl.len().alias("rows"),
        pl.col("event_ts").min().cast(pl.String).alias("first_event"),
        pl.col("event_ts").max().cast(pl.String).alias("last_event"),
        pl.col("amount").sum().round(2).alias("amount"),
        pl.col("is_fraud").sum().alias("frauds"),
    ).row(0)
    return hashlib.sha256(repr(summary).encode()).hexdigest()[:16]

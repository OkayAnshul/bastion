"""Runtime settings (environment variables) and typed YAML experiment configs.

Two kinds of configuration are kept deliberately separate:

* ``Settings``: where things live and how to reach services. It comes from ``BASTION_*``
  environment variables or ``.env`` and differs per machine or container.
* ``configs/*.yaml``: experiment and policy parameters (split boundaries, cost assumptions,
  rule thresholds). These are versioned in git because results depend on them.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven settings. Paths are relative to the working directory (repo root)."""

    model_config = SettingsConfigDict(env_prefix="BASTION_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    results_dir: Path = Path("docs/results")
    config_dir: Path = Path("configs")

    redis_url: str = "redis://localhost:6379/0"
    kafka_bootstrap: str = "localhost:19092"
    # Local sqlite tracking by default; docker compose points containers at http://mlflow:5000.
    mlflow_tracking_uri: str = "sqlite:///mlruns/mlflow.db"

    # Scoring service (Phase 3). A local bundle directory overrides the MLflow registry.
    model_path: Path | None = None
    model_name: str = "bastion-fraud"
    model_alias: str = "champion"
    # Review threshold tuned by `bastion policy sweep` for the served model; unset means 0.
    policy_path: Path | None = None
    decision_sink: str = "jsonl"  # jsonl | kafka | sqlite | none
    decision_log_path: Path = Path("artifacts/decisions/decisions.jsonl")
    # The analyst console's operational store (bastion.serving.decision_store).
    decision_db_path: Path = Path("artifacts/decisions/decisions.db")
    # Where the analyst console finds the scoring service for model and policy details.
    scoring_url: str = "http://localhost:8000"
    # Concurrent Redis connections per worker. redis-py's asyncio pool raises instead of waiting
    # once all are in use, and the score handler answers 503.
    redis_max_connections: int = 100

    log_level: str = "INFO"
    log_json: bool = False

    @property
    def raw_dir(self) -> Path:
        """Untouched downloads (IEEE-CIS zip and CSVs)."""
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        """Canonical event tables derived from raw data."""
        return self.data_dir / "processed"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def load_config[ModelT: BaseModel](
    name: str, model: type[ModelT], config_dir: Path | None = None
) -> ModelT:
    """Load ``<config_dir>/<name>.yaml`` and validate it against ``model``.

    Validation failures raise ``pydantic.ValidationError``, so a typo in a YAML key fails loudly
    instead of silently falling back to a default.
    """
    directory = config_dir if config_dir is not None else get_settings().config_dir
    path = directory / f"{name}.yaml"
    with path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    return model.model_validate(raw)

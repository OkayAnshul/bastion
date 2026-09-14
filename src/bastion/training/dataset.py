"""Model-ready tables, and the feature specification that travels with a trained model."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from bastion.config import load_config
from bastion.data.splits import SPLIT_COLUMN, SplitConfig, assign_splits
from bastion.schemas.tables import ATTRIBUTE_PREFIX

BASE_COLUMNS = ("txn_id", "event_ts", "amount", "merchant_category", "is_fraud")


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int
    lightgbm: dict[str, Any]
    num_boost_round: int = Field(ge=1)
    early_stopping_rounds: int = Field(ge=1)
    calibration: Literal["select", "isotonic", "platt"]
    calibration_holdout_fraction: float = Field(gt=0, lt=1)
    label_strength: float = Field(gt=0)
    passthrough: tuple[str, ...]
    max_category_levels: int = Field(ge=1)
    bootstrap_resamples: int = Field(ge=0)


def load_model_config(config_dir: Path | None = None) -> ModelConfig:
    return load_config("model", ModelConfig, config_dir)


def build_model_frame(
    events: pl.DataFrame, features: pl.DataFrame, splits: SplitConfig
) -> pl.DataFrame:
    """Labels, amount, category and vendor attributes, joined with features and split labels.

    ``features`` comes from the caller, either point-in-time or naive, so this module never decides
    which feature pipeline is used.
    """
    attributes = [c for c in events.columns if c.startswith(ATTRIBUTE_PREFIX)]
    frame = events.select(*BASE_COLUMNS, *attributes).join(
        features, on="txn_id", how="left", maintain_order="left"
    )
    return assign_splits(frame, splits)


def check_label_maturity(
    frame: pl.DataFrame, labels: pl.DataFrame, *, train_window: str, next_window: str
) -> None:
    """ADR-007: every training label must have arrived before the next evaluated window starts.

    If one arrives later, the model was trained on information from a time it is then evaluated on.
    """
    train_ids = frame.filter(pl.col(SPLIT_COLUMN) == train_window).select("txn_id")
    cutoff = cast(
        datetime | None, frame.filter(pl.col(SPLIT_COLUMN) == next_window)["event_ts"].min()
    )
    latest = labels.join(train_ids, on="txn_id").select(pl.col("label_ts").max()).item()
    if latest is not None and cutoff is not None and latest >= cutoff:
        raise ValueError(
            f"a {train_window!r} label arrives at {latest}, after {next_window!r} starts at "
            f"{cutoff}: widen the maturity gap or shorten the label delay"
        )


@dataclass(frozen=True)
class FeatureSpec:
    """Model inputs in order, plus the vocabulary of every categorical input.

    Its fingerprint identifies the exact input layout. Serving refuses a model whose fingerprint
    does not match the features it can build (ADR-004).
    """

    numeric: tuple[str, ...]
    categorical: tuple[tuple[str, tuple[str, ...]], ...]  # (column, vocabulary); code = index

    @property
    def categorical_columns(self) -> tuple[str, ...]:
        return tuple(column for column, _ in self.categorical)

    @property
    def columns(self) -> tuple[str, ...]:
        return self.numeric + self.categorical_columns

    def to_dict(self) -> dict[str, Any]:
        return {
            "numeric": list(self.numeric),
            "categorical": {column: list(vocab) for column, vocab in self.categorical},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureSpec:
        return cls(
            tuple(data["numeric"]),
            tuple((column, tuple(vocab)) for column, vocab in data["categorical"].items()),
        )

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


def select_passthrough(columns: Sequence[str], patterns: Sequence[str]) -> list[str]:
    """Vendor attribute columns whose unprefixed name matches any pattern."""
    compiled = [re.compile(p) for p in patterns]
    return [
        c
        for c in columns
        if c.startswith(ATTRIBUTE_PREFIX)
        and any(p.search(c.removeprefix(ATTRIBUTE_PREFIX)) for p in compiled)
    ]


def fit_feature_spec(
    train: pl.DataFrame,
    feature_columns: Sequence[str],
    passthrough: Sequence[str],
    max_category_levels: int,
) -> FeatureSpec:
    """Choose model inputs and categorical vocabularies from the training window only."""
    candidates = [
        "amount",
        "merchant_category",
        *feature_columns,
        *select_passthrough(train.columns, passthrough),
    ]
    numeric: list[str] = []
    categorical: list[tuple[str, tuple[str, ...]]] = []
    for column in candidates:
        dtype = train.schema[column]
        if dtype == pl.String:
            levels = (
                train.group_by(column)
                .len()
                .drop_nulls(column)
                .sort(["len", column], descending=[True, False])
            )
            categorical.append((column, tuple(levels[column].head(max_category_levels).to_list())))
        elif dtype.is_numeric() or dtype == pl.Boolean:
            numeric.append(column)
        else:
            raise TypeError(f"unsupported model input dtype {dtype} for column {column!r}")
    return FeatureSpec(tuple(numeric), tuple(categorical))


def to_matrix(frame: pl.DataFrame, spec: FeatureSpec) -> npt.NDArray[np.float32]:
    """Model input matrix in spec order.

    Categorical levels become vocabulary codes. Levels outside the vocabulary (rare or unseen in
    training) share one extra code, and nulls stay NaN, which LightGBM treats as missing.
    """
    exprs = [pl.col(column).cast(pl.Float32) for column in spec.numeric]
    for column, vocab in spec.categorical:
        codes = {level: float(i) for i, level in enumerate(vocab)}
        exprs.append(
            pl.when(pl.col(column).is_null())
            .then(None)
            .otherwise(
                pl.col(column).replace_strict(
                    codes, default=float(len(vocab)), return_dtype=pl.Float32
                )
            )
            .alias(column)
        )
    return frame.select(exprs).to_numpy().astype(np.float32)

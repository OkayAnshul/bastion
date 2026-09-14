"""Rules baseline (Phase 0): five hand-written rules that every model must beat.

Thresholds are derived, not guessed. ``configs/rules.yaml`` fixes hand-chosen quantile levels, and
the numbers they imply come from legitimate transactions in the training window only. They are then
frozen and applied unchanged to later windows: the same discipline a model gets.

The rules use only canonical fields and shared features, so they run identically on IEEE-CIS and on
synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from bastion.config import load_config

RULE_NAMES = (
    "high_amount",
    "velocity_1h",
    "new_device_high_amount",
    "amount_spike",
    "no_history_high_amount",
)


class RuleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    high_amount_quantile: float = Field(gt=0, lt=1)
    velocity_1h_quantile: float = Field(gt=0, lt=1)
    new_device_amount_quantile: float = Field(gt=0, lt=1)
    no_history_amount_quantile: float = Field(gt=0, lt=1)
    spike_multiplier: float = Field(gt=1)
    spike_min_history: int = Field(ge=1)


def load_rule_config(config_dir: Path | None = None) -> RuleConfig:
    return load_config("rules", RuleConfig, config_dir)


@dataclass(frozen=True)
class RuleThresholds:
    high_amount: float
    velocity_1h: float
    new_device_amount: float
    no_history_amount: float
    spike_multiplier: float
    spike_min_history: int


def fit_thresholds(train: pl.DataFrame, config: RuleConfig) -> RuleThresholds:
    """Turn quantile levels into thresholds using legitimate rows of ``train`` only.

    ``train`` must already be restricted to the training window: this function cannot tell.
    """
    legit = train.filter(~pl.col("is_fraud"))
    if legit.is_empty():
        raise ValueError("no legitimate transactions to fit rule thresholds on")

    def quantile(column: str, level: float) -> float:
        value = legit[column].quantile(level, interpolation="higher")
        if value is None:
            raise ValueError(f"cannot take a quantile of empty column {column!r}")
        return float(value)

    return RuleThresholds(
        high_amount=quantile("amount", config.high_amount_quantile),
        velocity_1h=quantile("card_txn_count_1h", config.velocity_1h_quantile),
        new_device_amount=quantile("amount", config.new_device_amount_quantile),
        no_history_amount=quantile("amount", config.no_history_amount_quantile),
        spike_multiplier=config.spike_multiplier,
        spike_min_history=config.spike_min_history,
    )


def apply_rules(frame: pl.DataFrame, thresholds: RuleThresholds) -> pl.DataFrame:
    """Add a boolean ``rule_<name>`` column per rule and ``rules_fired`` (how many fired, 0-5).

    ``frame`` needs ``amount`` plus the feature columns from ``bastion.features.batch``. A rule
    whose inputs are missing (for example no device data) does not fire.
    """
    amount = pl.col("amount")
    count_7d = pl.col("card_txn_count_7d")
    rules = {
        "high_amount": amount > thresholds.high_amount,
        "velocity_1h": pl.col("card_txn_count_1h") > thresholds.velocity_1h,
        "new_device_high_amount": ~pl.col("card_device_seen_before")
        & (amount > thresholds.new_device_amount),
        "amount_spike": (count_7d >= thresholds.spike_min_history)
        & (amount > thresholds.spike_multiplier * pl.col("card_amount_sum_7d") / count_7d),
        "no_history_high_amount": (count_7d == 0) & (amount > thresholds.no_history_amount),
    }
    assert tuple(rules) == RULE_NAMES
    flagged = frame.with_columns(
        expr.fill_null(False).alias(f"rule_{name}") for name, expr in rules.items()
    )
    return flagged.with_columns(
        pl.sum_horizontal(pl.col(f"rule_{name}").cast(pl.Int8) for name in RULE_NAMES)
        .cast(pl.Int8)
        .alias("rules_fired")
    )

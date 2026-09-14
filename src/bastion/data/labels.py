"""When each fraud label becomes known (ADR-003, ADR-007).

IEEE-CIS records whether a transaction was fraud, not when that became known. Bastion simulates it:

* a fraud label arrives after a chargeback delay (lognormal, capped at the maturity window);
* a legitimate label "arrives" when the maturity window passes with no chargeback.

Delays depend only on ``txn_id`` and the seed, never on row order or on which rows are present.
So the batch pipeline, the stream replay, and any subset of the data agree on every ``label_ts``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Self

import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from bastion.config import load_config

MS_PER_DAY = 86_400_000
LABEL_COLUMNS = ("txn_id", "card_id", "merchant_id", "event_ts", "label_ts", "is_fraud")


class LabelDelayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int
    fraud_median_days: float = Field(gt=0)
    fraud_sigma: float = Field(gt=0)
    maturity_days: float = Field(gt=0)

    @model_validator(mode="after")
    def _median_inside_maturity(self) -> Self:
        if self.fraud_median_days >= self.maturity_days:
            raise ValueError("fraud_median_days must be below maturity_days")
        return self


def load_label_delay_config(config_dir: Path | None = None) -> LabelDelayConfig:
    return load_config("labels", LabelDelayConfig, config_dir)


def _unit_uniforms(txn_ids: Sequence[str], seed: int) -> npt.NDArray[np.float64]:
    """A stable Uniform(0, 1) draw per transaction, strictly inside the interval."""
    draws = np.empty(len(txn_ids), dtype=np.float64)
    for i, txn_id in enumerate(txn_ids):
        digest = hashlib.blake2b(f"{seed}:{txn_id}".encode(), digest_size=8).digest()
        draws[i] = (int.from_bytes(digest, "big") + 0.5) / 2.0**64
    return draws


def fraud_delay_days(txn_ids: Sequence[str], config: LabelDelayConfig) -> npt.NDArray[np.float64]:
    """Chargeback delay per fraud transaction: lognormal, capped at ``maturity_days``."""
    normal = NormalDist()
    z = np.array([normal.inv_cdf(u) for u in _unit_uniforms(txn_ids, config.seed)])
    delay = config.fraud_median_days * np.exp(config.fraud_sigma * z)
    return np.minimum(delay, config.maturity_days).astype(np.float64)


def label_events(events: pl.DataFrame, config: LabelDelayConfig) -> pl.DataFrame:
    """One row per labelled transaction, with its simulated ``label_ts``, sorted by ``label_ts``.

    Unlabelled transactions (``is_fraud`` null) produce no label event.
    """
    labelled = events.filter(pl.col("is_fraud").is_not_null()).select(
        "txn_id", "card_id", "merchant_id", "event_ts", "is_fraud"
    )
    fraud = labelled["is_fraud"].to_numpy().astype(np.bool_)
    delay_days = np.full(labelled.height, config.maturity_days, dtype=np.float64)
    if fraud.any():
        fraud_ids = labelled.filter(pl.col("is_fraud"))["txn_id"].to_list()
        delay_days[fraud] = fraud_delay_days(fraud_ids, config)
    delay_ms = pl.Series(np.rint(delay_days * MS_PER_DAY).astype(np.int64))
    return (
        labelled.with_columns(
            (pl.col("event_ts") + pl.duration(milliseconds=pl.lit(delay_ms))).alias("label_ts")
        )
        .select(LABEL_COLUMNS)
        .sort("label_ts", "txn_id")
    )

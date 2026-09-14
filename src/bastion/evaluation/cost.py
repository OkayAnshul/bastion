"""Monetary cost of decisions (ADR-005).

Loss = missed fraud (amount) + false declines (margin + friction) + analyst review time.
The parameters are assumptions recorded in ``configs/costs.yaml``, not measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from bastion.config import load_config


class Action(StrEnum):
    APPROVE = "approve"
    REVIEW = "review"
    BLOCK = "block"


_ACTION_VALUES = frozenset(a.value for a in Action)


class CostModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    currency: str = Field(pattern=r"^[A-Z]{3}$")
    false_decline_margin_rate: float = Field(ge=0, le=1)
    false_decline_fixed_cost: float = Field(ge=0)
    review_cost_per_case: float = Field(ge=0)
    review_catch_rate: float = Field(ge=0, le=1)

    def false_decline_cost(self, amount: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.false_decline_margin_rate * amount + self.false_decline_fixed_cost


def load_cost_model(config_dir: Path | None = None) -> CostModel:
    return load_config("costs", CostModel, config_dir)


@dataclass(frozen=True)
class LossBreakdown:
    missed_fraud: float
    false_declines: float
    review_cost: float

    @property
    def total(self) -> float:
        return self.missed_fraud + self.false_declines + self.review_cost


def aligned_decisions(
    actions: npt.ArrayLike, is_fraud: npt.ArrayLike, amount: npt.ArrayLike
) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
    """Convert and validate decision arrays: equal lengths, known actions, positive amounts."""
    acts = np.asarray(actions, dtype=np.str_)
    fraud = np.asarray(is_fraud, dtype=np.bool_)
    amt = np.asarray(amount, dtype=np.float64)
    if not acts.shape == fraud.shape == amt.shape or acts.ndim != 1:
        raise ValueError(
            f"shape mismatch: actions {acts.shape}, labels {fraud.shape}, amounts {amt.shape}"
        )
    unknown = set(np.unique(acts).tolist()) - _ACTION_VALUES
    if unknown:
        raise ValueError(f"unknown actions: {sorted(unknown)}")
    if amt.size and not np.all(amt > 0):
        raise ValueError("amounts must be positive")
    return acts, fraud, amt


def realized_loss(
    actions: npt.ArrayLike, is_fraud: npt.ArrayLike, amount: npt.ArrayLike, costs: CostModel
) -> LossBreakdown:
    """Loss given the true labels (offline evaluation, not the policy's expected loss).

    - approve + fraud: full amount lost
    - review + fraud: analyst stops ``review_catch_rate`` of it; the rest is lost
    - block + legitimate: false-decline cost
    - review (any label): analyst cost per case
    """
    acts, fraud, amt = aligned_decisions(actions, is_fraud, amount)
    approve = acts == Action.APPROVE.value
    review = acts == Action.REVIEW.value
    block = acts == Action.BLOCK.value

    missed = (
        amt[approve & fraud].sum() + (1.0 - costs.review_catch_rate) * amt[review & fraud].sum()
    )
    false_declines = costs.false_decline_cost(amt[block & ~fraud]).sum()
    review_cost = costs.review_cost_per_case * review.sum()
    return LossBreakdown(float(missed), float(false_declines), float(review_cost))

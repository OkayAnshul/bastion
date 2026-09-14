"""Metrics for fraud decisions and scores.

The headline numbers are monetary: fraud value caught, false-decline rate, and loss. Ranking metrics
(PR-AUC) are reported to compare scorers before any thresholds are chosen. Accuracy and F1 are
deliberately absent (ADR-005).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import SupportsFloat

import numpy as np
import numpy.typing as npt
from sklearn.metrics import average_precision_score, roc_auc_score

from bastion.evaluation.cost import (
    Action,
    CostModel,
    LossBreakdown,
    aligned_decisions,
    realized_loss,
)


def _ratio(numerator: SupportsFloat, denominator: SupportsFloat) -> float:
    """Undefined ratios are NaN, never a misleading 0."""
    den = float(denominator)
    return float(numerator) / den if den else float("nan")


@dataclass(frozen=True)
class DecisionReport:
    n_transactions: int
    n_fraud: int
    block_rate: float
    review_rate: float
    intervention_precision: float  # share of blocked + reviewed transactions that were fraud
    fraud_recall: float  # share of frauds blocked or sent to review
    fraud_value_total: float
    fraud_value_caught: float  # blocked fraud value + review_catch_rate x reviewed fraud value
    fraud_value_caught_rate: float
    false_decline_rate: float  # share of legitimate transactions blocked
    loss: LossBreakdown

    def to_dict(self) -> dict[str, float]:
        values = {k: float(v) for k, v in vars(self).items() if k != "loss"}
        values |= {
            "loss_missed_fraud": self.loss.missed_fraud,
            "loss_false_declines": self.loss.false_declines,
            "loss_review_cost": self.loss.review_cost,
            "loss_total": self.loss.total,
        }
        return values


def evaluate_decisions(
    actions: npt.ArrayLike, is_fraud: npt.ArrayLike, amount: npt.ArrayLike, costs: CostModel
) -> DecisionReport:
    acts, fraud, amt = aligned_decisions(actions, is_fraud, amount)
    block = acts == Action.BLOCK.value
    review = acts == Action.REVIEW.value
    intervene = block | review
    legit = ~fraud

    fraud_value_total = float(amt[fraud].sum())
    fraud_value_caught = float(
        amt[block & fraud].sum() + costs.review_catch_rate * amt[review & fraud].sum()
    )
    return DecisionReport(
        n_transactions=int(acts.size),
        n_fraud=int(fraud.sum()),
        block_rate=_ratio(block.sum(), acts.size),
        review_rate=_ratio(review.sum(), acts.size),
        intervention_precision=_ratio((intervene & fraud).sum(), intervene.sum()),
        fraud_recall=_ratio((intervene & fraud).sum(), fraud.sum()),
        fraud_value_total=fraud_value_total,
        fraud_value_caught=fraud_value_caught,
        fraud_value_caught_rate=_ratio(fraud_value_caught, fraud_value_total),
        false_decline_rate=_ratio((block & legit).sum(), legit.sum()),
        loss=realized_loss(acts, fraud, amt, costs),
    )


def ranking_metrics(is_fraud: npt.ArrayLike, scores: npt.ArrayLike) -> dict[str, float]:
    """PR-AUC (average precision) and ROC-AUC. NaN when only one class is present.

    Under heavy imbalance PR-AUC is the informative one: its no-skill value equals the positive
    rate, while ROC-AUC's is always 0.5, which hides poor precision.
    """
    y = np.asarray(is_fraud, dtype=np.bool_)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape:
        raise ValueError(f"shape mismatch: labels {y.shape}, scores {s.shape}")
    positive_rate = _ratio(y.sum(), y.size)
    if y.size == 0 or y.all() or not y.any():
        return {"pr_auc": float("nan"), "roc_auc": float("nan"), "positive_rate": positive_rate}
    return {
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "positive_rate": positive_rate,
    }

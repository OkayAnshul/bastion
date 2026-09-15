"""Expected cost of each action for a transaction (ADR-005).

For a calibrated fraud probability ``p`` and an amount ``a``, under the assumed costs in
``configs/costs.yaml``:

- **approve:** ``p · a``. An approved fraud is charged back in full.
- **review:** ``r + p · (1 - c) · a``. Every review costs analyst time ``r``. The analyst stops a
  share ``c`` of frauds and clears legitimate transactions, so reviewing a legitimate customer
  declines no one.
- **block:** ``(1 - p) · (m · a + f)``. A blocked legitimate customer costs the margin share
  ``m`` of the sale plus a fixed friction cost ``f``.

These are exactly the charges ``evaluation.cost.realized_loss`` makes once the label is known, so
each expected cost is the realized loss averaged over the model's probability. That is why the
policy needs calibrated probabilities (ADR-006): a "0.7" that is really 0.3 misprices every action.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from bastion.evaluation.cost import Action, CostModel

FloatArray = npt.NDArray[np.float64]
StrArray = npt.NDArray[np.str_]


@dataclass(frozen=True)
class ExpectedCosts:
    approve: FloatArray
    review: FloatArray
    block: FloatArray

    def __len__(self) -> int:
        return int(self.approve.size)

    @property
    def best_without_review(self) -> FloatArray:
        return np.minimum(self.approve, self.block)

    @property
    def action_without_review(self) -> StrArray:
        """Block only when strictly cheaper than approving: when indifferent, decline no one."""
        return np.where(self.block < self.approve, Action.BLOCK.value, Action.APPROVE.value)

    @property
    def review_benefit(self) -> FloatArray:
        """Expected saving of a review over the better of approve and block (may be negative)."""
        return self.best_without_review - self.review


def expected_costs(
    probability: npt.ArrayLike, amount: npt.ArrayLike, costs: CostModel
) -> ExpectedCosts:
    p = np.asarray(probability, dtype=np.float64)
    a = np.asarray(amount, dtype=np.float64)
    if p.shape != a.shape or p.ndim != 1:
        raise ValueError(f"shape mismatch: probabilities {p.shape}, amounts {a.shape}")
    if p.size and (bool(np.isnan(p).any()) or p.min() < 0 or p.max() > 1):
        raise ValueError("probabilities must be in [0, 1]")
    if a.size and not np.all(a > 0):
        raise ValueError("amounts must be positive")
    return ExpectedCosts(
        approve=p * a,
        review=costs.review_cost_per_case + p * (1.0 - costs.review_catch_rate) * a,
        block=(1.0 - p) * costs.false_decline_cost(a),
    )

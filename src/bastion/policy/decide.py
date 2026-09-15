"""Three-way decisions under a daily review budget, and how their thresholds are tuned (ADR-005).

A transaction goes to human review when reviewing it is expected to save more than a threshold τ,
compared with the better of approving and blocking it, and only while its event-time day still has
review capacity. Every other transaction gets its cheaper non-review action. τ = 0 is the
unconstrained expected-loss optimum. A positive τ keeps low-value cases from using up a day's
reviews before higher-value ones arrive, which matters once the budget binds.

The budget is applied exactly as the scoring service applies it: in event order, with one counter
per UTC day. Rows passed here must therefore be sorted by event time.

``decide_by_probability`` is the baseline the expected-loss policy replaces: two probability
thresholds, the same budget, and no use of the amount.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from bastion.evaluation.cost import Action
from bastion.policy.expected_loss import ExpectedCosts, FloatArray, StrArray

BoolArray = npt.NDArray[np.bool_]
NO_OVERRIDE = ""

_APPROVE, _REVIEW, _BLOCK = Action.APPROVE.value, Action.REVIEW.value, Action.BLOCK.value


def within_daily_budget(
    candidate: npt.ArrayLike, day: npt.ArrayLike, reviews_per_day: int
) -> BoolArray:
    """The first ``reviews_per_day`` candidates of each day, taken in row (event) order."""
    wants = np.asarray(candidate, dtype=np.bool_)
    days = np.asarray(day, dtype=np.int64)
    if wants.shape != days.shape or wants.ndim != 1:
        raise ValueError(f"shape mismatch: candidates {wants.shape}, days {days.shape}")
    if reviews_per_day < 0:
        raise ValueError("reviews_per_day must be non-negative")
    if days.size and bool(np.any(days[1:] < days[:-1])):
        raise ValueError("rows must be in event order (days non-decreasing)")
    if not wants.size:
        return wants.copy()
    counts = wants.astype(np.int64)
    running = np.cumsum(counts)
    first_of_day = np.ones(days.size, dtype=np.bool_)
    first_of_day[1:] = days[1:] != days[:-1]
    before_day = (running - counts)[first_of_day]  # candidates on earlier days
    rank_in_day = running - before_day[np.cumsum(first_of_day) - 1]
    return wants & (rank_in_day <= reviews_per_day)


@dataclass(frozen=True)
class Decisions:
    actions: StrArray
    expected_cost: FloatArray  # expected cost of the action taken
    wanted_review: BoolArray  # the policy preferred a review
    capped: BoolArray  # preferred a review, but the day's budget was already used

    @property
    def total_expected_cost(self) -> float:
        return float(self.expected_cost.sum())


def _free_rows(forced: npt.ArrayLike | None, n: int) -> BoolArray:
    if forced is None:
        return np.ones(n, dtype=np.bool_)
    actions = np.asarray(forced, dtype=np.str_)
    if actions.shape != (n,):
        raise ValueError(f"expected {n} override actions, got shape {actions.shape}")
    unknown = set(np.unique(actions).tolist()) - {NO_OVERRIDE, _APPROVE, _BLOCK}
    if unknown:
        raise ValueError(f"overrides may only approve or block, got {sorted(unknown)}")
    free: BoolArray = actions == NO_OVERRIDE
    return free


def _decide(
    costs: ExpectedCosts,
    day: npt.ArrayLike,
    reviews_per_day: int,
    wants_review: BoolArray,
    fallback: StrArray,
    forced: npt.ArrayLike | None,
) -> Decisions:
    free = _free_rows(forced, len(costs))
    wants = wants_review & free  # overridden rows never use review capacity
    granted = within_daily_budget(wants, day, reviews_per_day)
    actions = np.where(granted, _REVIEW, fallback)
    if forced is not None:
        actions = np.where(free, actions, np.asarray(forced, dtype=np.str_))
    expected = np.select(
        [actions == _APPROVE, actions == _REVIEW], [costs.approve, costs.review], costs.block
    )
    return Decisions(actions.astype(np.str_), expected.astype(np.float64), wants, wants & ~granted)


def decide_expected_loss(
    costs: ExpectedCosts,
    day: npt.ArrayLike,
    *,
    threshold: float,
    reviews_per_day: int,
    forced: npt.ArrayLike | None = None,
) -> Decisions:
    """Review when the expected saving exceeds ``threshold`` and the day has capacity."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    return _decide(
        costs,
        day,
        reviews_per_day,
        costs.review_benefit > threshold,
        costs.action_without_review,
        forced,
    )


def decide_by_probability(
    probability: npt.ArrayLike,
    costs: ExpectedCosts,
    day: npt.ArrayLike,
    *,
    review_at: float,
    block_at: float,
    reviews_per_day: int,
    forced: npt.ArrayLike | None = None,
) -> Decisions:
    """Baseline: block at ``p ≥ block_at``, review at ``review_at ≤ p < block_at``, else approve.

    Reviews over the day's budget are approved. Amounts are ignored, so a doubtful 5 USD payment and
    a doubtful 5,000 USD payment get the same action.
    """
    p = np.asarray(probability, dtype=np.float64)
    if p.shape != (len(costs),):
        raise ValueError(f"expected {len(costs)} probabilities, got shape {p.shape}")
    if not 0 <= review_at <= block_at:
        raise ValueError("thresholds must satisfy 0 <= review_at <= block_at")
    block = p >= block_at
    fallback = np.where(block, _BLOCK, _APPROVE)
    return _decide(costs, day, reviews_per_day, (p >= review_at) & ~block, fallback, forced)


# ------------------------------------------------------------------------------ tuning


@dataclass(frozen=True)
class TunedThreshold:
    review_threshold: float
    expected_loss: float  # total expected cost on the tuning rows at this threshold
    candidates: int


@dataclass(frozen=True)
class TunedProbabilityThresholds:
    review_at: float
    block_at: float
    expected_loss: float
    candidates: int


def _improves(loss: float, best: float) -> bool:
    return math.isinf(best) or loss < best - 1e-9 * max(1.0, abs(best))


def tune_review_threshold(
    costs: ExpectedCosts,
    day: npt.ArrayLike,
    *,
    reviews_per_day: int,
    candidates: int,
    forced: npt.ArrayLike | None = None,
) -> TunedThreshold:
    """The threshold with the lowest total expected cost on the tuning rows, under the budget.

    No labels are needed: expected costs come from calibrated probabilities (ADR-005, ADR-006).
    Candidates are 0 and quantiles of the positive review benefits; the largest reviews nothing.
    Ties go to the larger threshold, which uses fewer reviews.
    """
    benefit = costs.review_benefit[_free_rows(forced, len(costs))]
    positive = benefit[benefit > 0]
    grid = np.array([0.0])
    if positive.size:
        levels = np.quantile(positive, np.linspace(0.0, 1.0, candidates))
        grid = np.unique(np.concatenate((grid, levels)))
    best_threshold, best_loss = 0.0, math.inf
    for threshold in grid[::-1]:
        loss = decide_expected_loss(
            costs, day, threshold=float(threshold), reviews_per_day=reviews_per_day, forced=forced
        ).total_expected_cost
        if _improves(loss, best_loss):
            best_threshold, best_loss = float(threshold), loss
    return TunedThreshold(best_threshold, best_loss, int(grid.size))


def tune_probability_thresholds(
    probability: npt.ArrayLike,
    costs: ExpectedCosts,
    day: npt.ArrayLike,
    *,
    reviews_per_day: int,
    levels: int = 30,
    forced: npt.ArrayLike | None = None,
) -> TunedProbabilityThresholds:
    """Baseline tuning: the probability thresholds with the lowest expected cost, same objective.

    Candidate thresholds are probabilities above which a geometric range of row shares falls (from
    half the rows down to 1 in 100,000), plus infinity, which never reviews or blocks.
    """
    p = np.asarray(probability, dtype=np.float64)
    if not p.size:
        raise ValueError("no rows to tune on")
    shares = np.geomspace(0.5, 1e-5, levels)
    values = np.unique(np.concatenate((np.quantile(p, 1.0 - shares), [np.inf])))
    best = (math.inf, math.inf)
    best_loss = math.inf
    for block_at in values[::-1]:
        for review_at in values[values <= block_at][::-1]:
            loss = decide_by_probability(
                p,
                costs,
                day,
                review_at=float(review_at),
                block_at=float(block_at),
                reviews_per_day=reviews_per_day,
                forced=forced,
            ).total_expected_cost
            if _improves(loss, best_loss):
                best, best_loss = (float(review_at), float(block_at)), loss
    pairs = int(values.size * (values.size + 1) // 2)
    return TunedProbabilityThresholds(best[0], best[1], best_loss, pairs)

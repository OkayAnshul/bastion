"""The decision policy as the scoring service runs it: one transaction at a time.

``propose`` applies the rule ``bastion.policy.decide`` applies to a whole window. Overrides come
first; otherwise the transaction gets the cheaper of approve and block, unless a review is expected
to save more than the tuned threshold. Whether a wanted review happens depends on the day's
remaining capacity (``bastion.policy.capacity``). A parity test replays one stream through both.

The threshold comes from ``bastion policy sweep``, which tunes it for one model, one budget and one
set of costs. ``serving_policy`` refuses a tuned file that doesn't match the served model or the
configured budget and costs: a threshold tuned for other probabilities or prices would be silently
wrong.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bastion.evaluation.cost import Action, CostModel
from bastion.policy.config import OverrideConfig, PolicyConfig
from bastion.policy.expected_loss import expected_costs
from bastion.policy.overrides import Override

UNTUNED_THRESHOLD = 0.0


@dataclass(frozen=True)
class TunedPolicy:
    """The threshold ``bastion policy sweep`` tuned for the operating budget, saved as JSON."""

    model_version: str
    spec_fingerprint: str
    reviews_per_day: int
    review_threshold: float
    costs: dict[str, Any]
    tuned_on: str
    git_revision: str

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def load(cls, path: Path) -> TunedPolicy:
        return cls(**json.loads(path.read_text()))


@dataclass(frozen=True)
class PolicyParameters:
    reviews_per_day: int
    review_threshold: float
    costs: CostModel
    overrides: OverrideConfig
    reason_codes_top_k: int
    tuned: bool  # False: the untuned threshold 0


def serving_policy(
    config: PolicyConfig,
    costs: CostModel,
    *,
    tuned: TunedPolicy | None,
    model_version: str,
    spec_fingerprint: str,
) -> PolicyParameters:
    """Combine configuration with a tuned threshold, refusing a threshold tuned for something else.

    Without a tuned file the threshold is 0: the unconstrained expected-loss rule, which spends each
    day's budget on the first transactions that want a review.
    """
    if tuned is not None:
        problems = []
        if tuned.spec_fingerprint != spec_fingerprint:
            problems.append("a different feature spec")
        if tuned.model_version != model_version:
            problems.append(f"model {tuned.model_version}, not {model_version}")
        if tuned.reviews_per_day != config.reviews_per_day:
            problems.append(
                f"{tuned.reviews_per_day} reviews per day, not {config.reviews_per_day}"
            )
        if tuned.costs != costs.model_dump():
            problems.append("different costs")
        if problems:
            raise ValueError(
                f"the tuned policy is for {'; '.join(problems)}: re-run `bastion policy sweep`"
            )
    return PolicyParameters(
        reviews_per_day=config.reviews_per_day,
        review_threshold=UNTUNED_THRESHOLD if tuned is None else tuned.review_threshold,
        costs=costs,
        overrides=config.overrides,
        reason_codes_top_k=config.reason_codes_top_k,
        tuned=tuned is not None,
    )


@dataclass(frozen=True)
class Proposal:
    action: Action  # what the policy wants, before the day's review capacity is checked
    fallback: Action  # the cheaper of approve and block, used when no review is available
    expected_cost: dict[str, float]
    review_benefit: float
    override_rule: str | None


def propose(
    probability: float,
    amount: float,
    override: Override | None,
    *,
    threshold: float,
    costs: CostModel,
) -> Proposal:
    priced = expected_costs([probability], [amount], costs)
    expected = {
        Action.APPROVE.value: float(priced.approve[0]),
        Action.REVIEW.value: float(priced.review[0]),
        Action.BLOCK.value: float(priced.block[0]),
    }
    benefit = float(priced.review_benefit[0])
    if override is not None:
        return Proposal(override.action, override.action, expected, benefit, override.rule)
    fallback = Action(str(priced.action_without_review[0]))
    action = Action.REVIEW if benefit > threshold else fallback
    return Proposal(action, fallback, expected, benefit, None)

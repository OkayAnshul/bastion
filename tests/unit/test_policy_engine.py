"""The online policy engine: review capacity in Redis, tuned thresholds, and parity with offline."""

import asyncio
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import fakeredis
import numpy as np
import pytest

from bastion.evaluation.cost import Action, CostModel
from bastion.policy.capacity import ReviewCapacity, event_day
from bastion.policy.config import OverrideConfig, PolicyConfig
from bastion.policy.decide import decide_expected_loss
from bastion.policy.engine import TunedPolicy, propose, serving_policy
from bastion.policy.expected_loss import expected_costs
from bastion.policy.overrides import override_for, overrides_for_rows

COSTS = CostModel(
    currency="USD",
    false_decline_margin_rate=0.10,
    false_decline_fixed_cost=2.0,
    review_cost_per_case=5.0,
    review_catch_rate=0.9,
)
CONFIG = PolicyConfig(
    reviews_per_day=3,
    review_budget_sweep=(0, 3),
    threshold_candidates=10,
    overrides=OverrideConfig(blocked_card_ids=frozenset({"c_bad"}), velocity_cap_1h=6),
    reason_codes_top_k=3,
)


def _capacity(reviews_per_day: int) -> ReviewCapacity:
    return ReviewCapacity(fakeredis.FakeAsyncRedis(), "test", reviews_per_day)


def test_event_day_is_the_offline_day_key() -> None:
    ts = datetime(2026, 3, 1, 23, 59, 59, tzinfo=UTC)
    assert event_day(ts) == (date(2026, 3, 1) - date(1970, 1, 1)).days
    assert event_day(ts + timedelta(seconds=1)) == event_day(ts) + 1


def test_concurrent_reservations_never_exceed_the_budget() -> None:
    async def scenario() -> tuple[list[bool], list[bool], int]:
        capacity = _capacity(7)
        today = await asyncio.gather(*(capacity.try_reserve(100) for _ in range(50)))
        tomorrow = await asyncio.gather(*(capacity.try_reserve(101) for _ in range(3)))
        return list(today), list(tomorrow), await capacity.used(100)

    today, tomorrow, used = asyncio.run(scenario())
    assert sum(today) == 7
    assert used == 7  # refused reservations left no trace
    assert all(tomorrow)


def test_a_zero_budget_never_reserves() -> None:
    assert asyncio.run(_capacity(0).try_reserve(1)) is False


def test_online_decisions_equal_offline_decisions_for_the_same_stream() -> None:
    rng = np.random.default_rng(8)
    n = 600
    day = np.sort(rng.integers(0, 4, n))
    p = rng.random(n) ** 3
    amount = rng.lognormal(4, 1.3, n)
    cards = rng.choice(["c1", "c2", "c_bad"], size=n, p=[0.49, 0.49, 0.02]).tolist()
    velocity = rng.integers(0, 8, n).tolist()
    threshold = 3.0

    forced, _ = overrides_for_rows(cards, [None] * n, ["m1"] * n, velocity, CONFIG.overrides)
    offline = decide_expected_loss(
        expected_costs(p, amount, COSTS),
        day,
        threshold=threshold,
        reviews_per_day=CONFIG.reviews_per_day,
        forced=forced,
    )

    async def online() -> list[str]:
        capacity = _capacity(CONFIG.reviews_per_day)
        actions = []
        for i in range(n):
            override = override_for(
                card_id=cards[i],
                device_id=None,
                merchant_id="m1",
                card_txn_count_1h=velocity[i],
                config=CONFIG.overrides,
            )
            proposal = propose(
                float(p[i]), float(amount[i]), override, threshold=threshold, costs=COSTS
            )
            action = proposal.action
            if action is Action.REVIEW and not await capacity.try_reserve(int(day[i])):
                action = proposal.fallback
            actions.append(action.value)
        return actions

    assert asyncio.run(online()) == offline.actions.tolist()
    assert offline.capped.any()  # the budget really bound, so capacity handling was exercised
    assert (forced != "").any()  # and so did the overrides


def _tuned(**changes: Any) -> TunedPolicy:
    fields: dict[str, Any] = {
        "model_version": "local/run1",
        "spec_fingerprint": "abc",
        "reviews_per_day": 3,
        "review_threshold": 4.5,
        "costs": COSTS.model_dump(),
        "tuned_on": "calibration",
        "git_revision": "deadbee",
    }
    return TunedPolicy(**(fields | changes))


def test_a_tuned_policy_round_trips_and_sets_the_threshold(tmp_path: Path) -> None:
    path = _tuned().save(tmp_path / "policy.json")
    parameters = serving_policy(
        CONFIG,
        COSTS,
        tuned=TunedPolicy.load(path),
        model_version="local/run1",
        spec_fingerprint="abc",
    )
    assert parameters.review_threshold == 4.5
    assert parameters.tuned


@pytest.mark.parametrize(
    "changes",
    [
        {"spec_fingerprint": "other"},
        {"model_version": "local/run2"},
        {"reviews_per_day": 200},
        {"costs": {**COSTS.model_dump(), "review_cost_per_case": 9.0}},
    ],
)
def test_a_policy_tuned_for_something_else_is_refused(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="re-run `bastion policy sweep`"):
        serving_policy(
            CONFIG,
            COSTS,
            tuned=_tuned(**changes),
            model_version="local/run1",
            spec_fingerprint="abc",
        )


def test_without_a_tuned_file_the_threshold_is_zero() -> None:
    parameters = serving_policy(
        CONFIG, COSTS, tuned=None, model_version="local/x", spec_fingerprint="y"
    )
    assert parameters.review_threshold == 0.0
    assert not parameters.tuned

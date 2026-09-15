from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

from bastion.evaluation.calibration import IsotonicCalibrator
from bastion.evaluation.cost import CostModel, realized_loss
from bastion.features.batch import feature_names
from bastion.policy.config import OverrideConfig, load_policy_config
from bastion.policy.decide import (
    decide_by_probability,
    decide_expected_loss,
    tune_probability_thresholds,
    tune_review_threshold,
    within_daily_budget,
)
from bastion.policy.expected_loss import expected_costs
from bastion.policy.overrides import override_for, overrides_for_rows
from bastion.policy.reason_codes import describe, top_reasons
from bastion.training.bundle import ModelBundle
from bastion.training.dataset import FeatureSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
COSTS = CostModel(
    currency="USD",
    false_decline_margin_rate=0.10,
    false_decline_fixed_cost=2.0,
    review_cost_per_case=5.0,
    review_catch_rate=0.9,
)
OVERRIDES = OverrideConfig(
    blocked_card_ids=frozenset({"c_bad"}),
    blocked_device_ids=frozenset({"d_bad"}),
    blocked_merchant_ids=frozenset({"m_bad"}),
    allowed_card_ids=frozenset({"c_vip", "c_bad"}),
    velocity_cap_1h=10,
)


def test_policy_config_loads() -> None:
    config = load_policy_config(REPO_ROOT / "configs")
    assert config.reviews_per_day in config.review_budget_sweep


# ------------------------------------------------------------------------------ expected loss


def test_expected_cost_is_the_realized_loss_averaged_over_the_probability() -> None:
    rng = np.random.default_rng(3)
    p = rng.random(40)
    amount = rng.lognormal(3, 1, 40)
    costs = expected_costs(p, amount, COSTS)
    for action in ("approve", "review", "block"):
        expected = getattr(costs, action)
        for i in range(p.size):
            if_fraud = realized_loss([action], [True], [amount[i]], COSTS).total
            if_legit = realized_loss([action], [False], [amount[i]], COSTS).total
            assert expected[i] == pytest.approx(p[i] * if_fraud + (1 - p[i]) * if_legit)


def test_expected_costs_reject_uncalibrated_scores() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        expected_costs([1.5], [10.0], COSTS)


def test_without_a_binding_budget_every_row_gets_its_cheapest_action() -> None:
    rng = np.random.default_rng(4)
    n = 400
    costs = expected_costs(rng.random(n) ** 3, rng.lognormal(4, 1.5, n), COSTS)
    decisions = decide_expected_loss(
        costs, np.zeros(n, dtype=np.int64), threshold=0.0, reviews_per_day=n
    )
    cheapest = np.column_stack([costs.approve, costs.review, costs.block]).min(axis=1)
    np.testing.assert_allclose(decisions.expected_cost, cheapest)
    assert not decisions.capped.any()


# ------------------------------------------------------------------------------ review budget


def test_budget_takes_the_first_candidates_of_each_day_in_event_order() -> None:
    candidate = np.array([True, False, True, True, True, True, False, True])
    day = np.array([0, 0, 0, 0, 1, 1, 1, 2])
    granted = within_daily_budget(candidate, day, 2)
    assert granted.tolist() == [True, False, True, False, True, True, False, True]
    assert not within_daily_budget(candidate, day, 0).any()


def test_budget_needs_rows_in_event_order() -> None:
    with pytest.raises(ValueError, match="event order"):
        within_daily_budget([True, True], [1, 0], 5)


def test_tuned_threshold_saves_reviews_for_higher_value_cases_when_the_budget_binds() -> None:
    # One day: five doubtful 200 USD payments arrive before a doubtful 5,000 USD one. One review.
    p = np.full(6, 0.3)
    amount = np.array([200.0] * 5 + [5_000.0])
    costs = expected_costs(p, amount, COSTS)
    day = np.zeros(6, dtype=np.int64)

    greedy = decide_expected_loss(costs, day, threshold=0.0, reviews_per_day=1)
    assert greedy.actions[0] == "review"
    assert greedy.capped[-1]  # the large payment found the budget used up

    tuned = tune_review_threshold(costs, day, reviews_per_day=1, candidates=50)
    best = decide_expected_loss(costs, day, threshold=tuned.review_threshold, reviews_per_day=1)
    assert best.actions.tolist() == ["block"] * 5 + ["review"]
    assert tuned.expected_loss == pytest.approx(best.total_expected_cost)
    # greedy: 11 + 4 x 15.4 + 351.4 = 424; tuned: 5 x 15.4 + 155 = 232
    assert best.total_expected_cost == pytest.approx(232.0)
    assert greedy.total_expected_cost == pytest.approx(424.0)


def test_tuned_policies_never_cost_more_than_approving_everything() -> None:
    rng = np.random.default_rng(5)
    n = 2_000
    p = rng.random(n) ** 4
    costs = expected_costs(p, rng.lognormal(4, 1.2, n), COSTS)
    day = np.repeat(np.arange(4), n // 4)
    for budget in (0, 10, n):
        by_loss = tune_review_threshold(costs, day, reviews_per_day=budget, candidates=40)
        by_probability = tune_probability_thresholds(
            p, costs, day, reviews_per_day=budget, levels=12
        )
        assert by_loss.expected_loss <= costs.approve.sum()
        assert by_probability.expected_loss <= costs.approve.sum()
        if budget in (0, n):  # budget never binds, so per-row optimal beats any threshold pair
            assert by_loss.expected_loss <= by_probability.expected_loss + 1e-6


def test_probability_thresholds_ignore_amounts_but_share_the_budget() -> None:
    p = np.array([0.95, 0.5, 0.5, 0.1])
    costs = expected_costs(p, [10.0, 10.0, 10_000.0, 10_000.0], COSTS)
    decisions = decide_by_probability(
        p, costs, np.zeros(4, dtype=np.int64), review_at=0.4, block_at=0.9, reviews_per_day=1
    )
    assert decisions.actions.tolist() == ["block", "review", "approve", "approve"]
    assert decisions.capped.tolist() == [False, False, True, False]


# ------------------------------------------------------------------------------ overrides


def _override(
    card: str = "c1",
    device: str | None = "d1",
    merchant: str = "m1",
    count: int | None = 0,
    config: OverrideConfig = OVERRIDES,
) -> tuple[str, str] | None:
    found = override_for(
        card_id=card, device_id=device, merchant_id=merchant, card_txn_count_1h=count, config=config
    )
    return None if found is None else (found.action.value, found.rule)


def test_override_precedence() -> None:
    assert _override(card="c_bad") == ("block", "blocked_card")  # blocklist beats the allowlist
    assert _override(card="c_vip", device="d_bad") == ("block", "blocked_device")
    assert _override(card="c_vip", merchant="m_bad") == ("block", "blocked_merchant")
    assert _override(card="c_vip", count=50) == ("approve", "allowed_card")  # beats the cap
    assert _override(count=10) == ("block", "velocity_cap_1h")
    assert _override(count=9) is None
    assert _override(device=None, count=None) is None
    assert _override(count=500, config=OverrideConfig()) is None  # cap disabled


def test_overridden_rows_keep_their_action_and_use_no_review_capacity() -> None:
    costs = expected_costs(np.full(3, 0.3), np.full(3, 200.0), COSTS)
    forced, rules = overrides_for_rows(
        ["c_bad", "c1", "c2"], ["d1", "d1", None], ["m1"] * 3, [0, 0, 0], OVERRIDES
    )
    decisions = decide_expected_loss(
        costs, np.zeros(3, dtype=np.int64), threshold=0.0, reviews_per_day=1, forced=forced
    )
    assert rules.tolist() == ["blocked_card", "", ""]
    # c1 takes the day's only review; c2 wanted one too and falls back to its cheaper action
    assert decisions.actions.tolist() == ["block", "review", "block"]
    assert decisions.capped.tolist() == [False, False, True]


# ------------------------------------------------------------------------------ reason codes


def test_reason_codes_are_the_largest_positive_treeshap_contributions() -> None:
    rng = np.random.default_rng(6)
    n = 2_000
    x = rng.normal(size=(n, 3)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1] + rng.normal(scale=0.5, size=n)) > 1.5
    spec = FeatureSpec(("amount", "card_txn_count_1h", "hour_of_day"), ())
    params = {"objective": "binary", "verbosity": -1, "num_threads": 1, "deterministic": True}
    booster = lgb.train(
        params, lgb.Dataset(x, label=y.astype(float), feature_name=list(spec.columns)), 30
    )
    raw = booster.predict(x)
    bundle = ModelBundle.from_booster(booster, IsotonicCalibrator.fit(raw, y), spec, {})

    i = int(np.argmax(raw))
    contributions = bundle.contributions_from_matrix(x[i : i + 1])
    assert contributions.shape == (1, 4)
    margin = float(booster.predict(x[i : i + 1], raw_score=True)[0])
    assert contributions.sum() == pytest.approx(margin, abs=1e-5)  # SHAP values add up

    values = dict(zip(spec.columns, x[i].tolist(), strict=True))
    reasons = top_reasons(contributions[0], spec, values, k=2)
    assert 1 <= len(reasons) <= 2
    assert all(r.contribution > 0 for r in reasons)
    assert [r.contribution for r in reasons] == sorted(
        (r.contribution for r in reasons), reverse=True
    )
    assert reasons[0].feature == spec.columns[int(np.argmax(contributions[0, :3]))]
    assert reasons[0].value == pytest.approx(values[reasons[0].feature])


def test_every_model_feature_has_a_readable_description() -> None:
    assert describe("card_txn_count_1h") == "Card transactions in the previous hour"
    assert (
        describe("device_distinct_cards_7d")
        == "Distinct cards on this device in the previous 7 days"
    )
    assert describe("attr_C1") == "Vendor attribute C1"
    assert describe("not_a_feature") == "not_a_feature"
    for name in ("amount", "merchant_category", *feature_names(with_labels=True)):
        assert describe(name) != name, name

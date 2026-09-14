import math
from pathlib import Path

import numpy as np
import pytest

from bastion.evaluation.cost import CostModel, load_cost_model, realized_loss
from bastion.evaluation.metrics import evaluate_decisions, ranking_metrics

REPO_ROOT = Path(__file__).resolve().parents[2]

COSTS = CostModel(
    currency="USD",
    false_decline_margin_rate=0.10,
    false_decline_fixed_cost=2.0,
    review_cost_per_case=5.0,
    review_catch_rate=0.9,
)

# Worked example used by the two tests below.
ACTIONS = ["block", "block", "approve", "review", "review"]
FRAUD = [True, False, True, False, True]
AMOUNT = [100.0, 200.0, 50.0, 400.0, 1000.0]


def test_loss_worked_example() -> None:
    loss = realized_loss(ACTIONS, FRAUD, AMOUNT, COSTS)
    # approved fraud 50 + the 10% of reviewed fraud 1000 that analysts miss
    assert loss.missed_fraud == pytest.approx(50.0 + 100.0)
    # one blocked legitimate customer: 10% of 200 + 2 friction
    assert loss.false_declines == pytest.approx(22.0)
    # two reviews at 5 each, regardless of label
    assert loss.review_cost == pytest.approx(10.0)
    assert loss.total == pytest.approx(182.0)


def test_decision_report_worked_example() -> None:
    report = evaluate_decisions(ACTIONS, FRAUD, AMOUNT, COSTS)
    assert report.n_fraud == 3
    assert report.block_rate == pytest.approx(0.4)
    assert report.review_rate == pytest.approx(0.4)
    assert report.intervention_precision == pytest.approx(2 / 4)
    assert report.fraud_recall == pytest.approx(2 / 3)
    assert report.fraud_value_total == pytest.approx(1150.0)
    assert report.fraud_value_caught == pytest.approx(100.0 + 900.0)
    assert report.false_decline_rate == pytest.approx(1 / 2)
    assert report.to_dict()["loss_total"] == pytest.approx(182.0)


def test_caught_plus_missed_equals_total_fraud_value() -> None:
    rng = np.random.default_rng(0)
    n = 500
    actions = rng.choice(["approve", "review", "block"], size=n)
    fraud = rng.random(n) < 0.2
    amount = rng.lognormal(3, 1, n)
    report = evaluate_decisions(actions, fraud, amount, COSTS)
    assert report.fraud_value_caught + report.loss.missed_fraud == pytest.approx(
        report.fraud_value_total
    )


def test_approving_everything_loses_all_fraud_value_and_declines_nobody() -> None:
    report = evaluate_decisions(["approve"] * 5, FRAUD, AMOUNT, COSTS)
    assert report.loss.total == pytest.approx(report.fraud_value_total)
    assert report.false_decline_rate == 0.0
    assert math.isnan(report.intervention_precision)  # nothing flagged: undefined, not 0


def test_blocking_everything_catches_all_fraud_at_full_false_decline_rate() -> None:
    report = evaluate_decisions(["block"] * 5, FRAUD, AMOUNT, COSTS)
    assert report.fraud_value_caught_rate == pytest.approx(1.0)
    assert report.false_decline_rate == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("actions", "fraud", "amount", "message"),
    [
        (["approve", "decline"], [True, False], [1.0, 2.0], "unknown actions"),
        (["approve"], [True, False], [1.0, 2.0], "shape mismatch"),
        (["approve", "block"], [True, False], [1.0, 0.0], "positive"),
    ],
)
def test_invalid_decisions_are_rejected(
    actions: list[str], fraud: list[bool], amount: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_decisions(actions, fraud, amount, COSTS)


def test_ranking_metrics_perfect_and_no_skill() -> None:
    labels = [False, False, True, False, True]
    perfect = ranking_metrics(labels, [0.1, 0.2, 0.9, 0.3, 0.8])
    assert perfect["pr_auc"] == pytest.approx(1.0)
    assert perfect["roc_auc"] == pytest.approx(1.0)
    constant = ranking_metrics(labels, [0.5] * 5)
    assert constant["pr_auc"] == pytest.approx(0.4)  # no-skill PR-AUC is the positive rate
    assert constant["roc_auc"] == pytest.approx(0.5)


def test_ranking_metrics_are_nan_for_a_single_class() -> None:
    result = ranking_metrics([False, False], [0.1, 0.9])
    assert math.isnan(result["pr_auc"])
    assert math.isnan(result["roc_auc"])


def test_repository_cost_config_is_valid() -> None:
    assert load_cost_model(REPO_ROOT / "configs").currency == "USD"

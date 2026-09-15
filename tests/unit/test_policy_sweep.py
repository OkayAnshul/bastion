"""The offline policy sweep on a small synthetic table and a quickly trained model."""

import json
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from bastion.data.labels import label_events
from bastion.evaluation.calibration import IsotonicCalibrator
from bastion.evaluation.cost import load_cost_model
from bastion.features.batch import compute_features, feature_names
from bastion.policy.config import OverrideConfig, PolicyConfig
from bastion.policy.sweep import (
    APPROVE_ALL,
    EXPECTED_LOSS,
    FIGURE,
    PROBABILITY,
    RULES,
    SweepResult,
    log_policy_sweep,
    run_policy_sweep,
    write_policy_report,
)
from bastion.rules.baseline import load_rule_config
from bastion.training.bundle import ModelBundle
from bastion.training.dataset import build_model_frame
from bastion.training.pipeline import fit_lightgbm_on_frame, labels_of, split_rows
from tests.unit.test_training import LABELS, fast_model_config, small_splits, synthetic_events

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY = PolicyConfig(
    reviews_per_day=5,
    review_budget_sweep=(0, 50),  # the operating budget (5) is added by the sweep
    threshold_candidates=20,
    overrides=OverrideConfig(velocity_cap_1h=8),
    reason_codes_top_k=3,
)
COSTS = load_cost_model(REPO_ROOT / "configs")


@pytest.fixture(scope="module")
def sweep() -> SweepResult:
    events = synthetic_events()
    config = fast_model_config()
    features = compute_features(
        events, label_events(events, LABELS), label_strength=config.label_strength
    )
    frame = build_model_frame(events, features, small_splits())
    fitted = fit_lightgbm_on_frame(frame, feature_names(with_labels=True), config)
    calibration = split_rows(frame, "calibration")
    calibrator = IsotonicCalibrator.fit(fitted.scores(calibration), labels_of(calibration))
    bundle = ModelBundle.from_booster(
        fitted.booster, calibrator, fitted.spec, {"label_strength": config.label_strength}
    )
    return run_policy_sweep(
        events,
        bundle,
        splits=small_splits(),
        labels=LABELS,
        costs=COSTS,
        policy=POLICY,
        rules=load_rule_config(REPO_ROOT / "configs"),
    )


def test_every_policy_is_evaluated_at_every_budget(sweep: SweepResult) -> None:
    assert sweep.budgets == (0, 5, 50)
    evaluated = {(o.policy, o.reviews_per_day) for o in sweep.outcomes}
    for budget in sweep.budgets:
        assert {(EXPECTED_LOSS, budget), (PROBABILITY, budget)} <= evaluated
    assert {(RULES, None), (APPROVE_ALL, None)} <= evaluated


def test_no_policy_exceeds_its_daily_review_budget(sweep: SweepResult) -> None:
    for outcome in sweep.outcomes:
        assert outcome.reviews_per_day_max <= (outcome.reviews_per_day or 0)
    assert sweep.outcome(EXPECTED_LOSS, 0).test["review_rate"] == 0


def test_every_policy_sees_the_same_overrides(sweep: SweepResult) -> None:
    overridden = {o.overridden for o in sweep.outcomes}
    assert overridden == {sum(sweep.override_counts.values())}


def test_approving_everything_misses_all_fraud_not_blocked_by_an_override(
    sweep: SweepResult,
) -> None:
    approve_all = sweep.outcome(APPROVE_ALL)
    assert approve_all.test["review_rate"] == 0
    assert approve_all.test["loss_review_cost"] == 0
    if not sweep.override_counts:
        assert approve_all.test["fraud_value_caught"] == 0


def test_sensitivity_retunes_under_every_cost_variant(sweep: SweepResult) -> None:
    assert sweep.sensitivity[0].variant == "Assumed costs"
    assert len(sweep.sensitivity) == 7
    assert all(row.reviews_per_day_mean <= POLICY.reviews_per_day for row in sweep.sensitivity)


def _reject_constant(name: str) -> None:
    raise ValueError(f"non-standard JSON constant {name}")


def test_report_is_strict_json_and_states_the_headline(sweep: SweepResult, tmp_path: Path) -> None:
    path = write_policy_report(
        sweep, COSTS, tmp_path, dataset="synthetic", source="test", model_version="test/1"
    )
    text = path.read_text()
    assert "At **5 reviews per day**, the expected-loss policy caught" in text
    assert "> **Synthetic data.**" in text
    json.loads(path.with_suffix(".json").read_text(), parse_constant=_reject_constant)
    assert (tmp_path / "figures" / FIGURE).stat().st_size > 0


def test_sweep_is_tracked_in_mlflow(sweep: SweepResult, tmp_path: Path) -> None:
    report = write_policy_report(
        sweep, COSTS, tmp_path, dataset="synthetic", source="test", model_version="test/1"
    )
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    run_id = log_policy_sweep(
        sweep, report, tracking_uri=uri, artifacts_dir=tmp_path, tags={"dataset": "synthetic"}
    )
    run = MlflowClient(tracking_uri=uri).get_run(run_id)
    assert "expected_loss_n5_fraud_value_caught_rate" in run.data.metrics
    assert "rules_loss_total" in run.data.metrics

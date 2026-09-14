import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from mlflow.tracking import MlflowClient

from bastion.data.labels import LabelDelayConfig, label_events
from bastion.data.splits import SplitConfig, SplitWindow
from bastion.evaluation.metrics import ranking_metrics
from bastion.features.batch import compute_features
from bastion.rules.baseline import load_rule_config
from bastion.streaming.synthetic import SyntheticConfig, generate
from bastion.training.bundle import ModelBundle
from bastion.training.dataset import ModelConfig, build_model_frame, load_model_config
from bastion.training.pipeline import labels_of, split_rows
from bastion.training.train import (
    FIGURES,
    SCORER_LABELS,
    TrainingInputs,
    TrainingResult,
    train_model,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS = LabelDelayConfig(seed=11, fraud_median_days=5.0, fraud_sigma=0.6, maturity_days=14.0)


def small_splits(gap_days: int = 14) -> SplitConfig:
    """Short windows for a 50-day synthetic table, same structure as configs/splits.yaml."""
    return SplitConfig(
        windows=(
            SplitWindow(name="train", start_day=0, end_day=14),
            SplitWindow(name="maturity_gap", start_day=14, end_day=14 + gap_days),
            SplitWindow(name="early_stopping", start_day=14 + gap_days, end_day=21 + gap_days),
            SplitWindow(name="calibration", start_day=21 + gap_days, end_day=28 + gap_days),
            SplitWindow(name="test", start_day=28 + gap_days, end_day=None),
        )
    )


def fast_model_config() -> ModelConfig:
    base = load_model_config(REPO_ROOT / "configs")
    return base.model_copy(
        update={
            "num_boost_round": 150,
            "early_stopping_rounds": 20,
            "bootstrap_resamples": 20,
            "lightgbm": {**base.lightgbm, "min_data_in_leaf": 20, "num_threads": 2},
        }
    )


def synthetic_events() -> pl.DataFrame:
    return generate(
        SyntheticConfig(
            days=50,
            n_cards=400,
            n_merchants=40,
            card_testing_attacks_per_day=3.0,
            account_takeovers_per_day=3.0,
        )
    )


@pytest.fixture(scope="module")
def events() -> pl.DataFrame:
    return synthetic_events()


@pytest.fixture(scope="module")
def trained(
    events: pl.DataFrame, tmp_path_factory: pytest.TempPathFactory
) -> tuple[TrainingResult, str]:
    root = tmp_path_factory.mktemp("training")
    uri = f"sqlite:///{root}/mlflow.db"
    inputs = TrainingInputs(
        events,
        small_splits(),
        LABELS,
        fast_model_config(),
        load_rule_config(REPO_ROOT / "configs"),
        "synthetic",
        "unit test",
    )
    result = train_model(
        inputs, results_dir=root / "results", artifacts_dir=root / "artifacts", tracking_uri=uri
    )
    return result, uri


def test_every_scorer_is_evaluated_on_the_test_window(trained: tuple[TrainingResult, str]) -> None:
    result, _ = trained
    assert set(result.metrics) == set(SCORER_LABELS)
    for name, values in result.metrics.items():
        assert 0.0 <= values["pr_auc"] <= 1.0, name
    assert "brier" not in result.metrics["rules"]  # a count of fired rules is not a probability
    assert result.metrics["lightgbm_isotonic"]["brier"] <= 1.0


def test_report_and_figures_are_written(trained: tuple[TrainingResult, str]) -> None:
    result, _ = trained
    text = result.report_path.read_text()
    assert "Test-window results" in text
    for figure in FIGURES:
        assert (result.report_path.parent / "figures" / figure).stat().st_size > 0


def test_saved_bundle_reproduces_the_reported_test_pr_auc(
    trained: tuple[TrainingResult, str], events: pl.DataFrame
) -> None:
    result, _ = trained
    bundle = ModelBundle.load(result.bundle_dir)
    labels = label_events(events, LABELS)
    frame = build_model_frame(
        events, compute_features(events, labels, label_strength=100.0), small_splits()
    )
    test = split_rows(frame, "test")
    probabilities = bundle.predict_proba(test)
    assert ranking_metrics(labels_of(test), probabilities)["pr_auc"] == pytest.approx(
        result.metrics["lightgbm_isotonic"]["pr_auc"]
    )
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_mlflow_run_holds_metrics_bundle_and_report(trained: tuple[TrainingResult, str]) -> None:
    result, uri = trained
    client = MlflowClient(tracking_uri=uri)
    run = client.get_run(result.run_id)
    assert math.isfinite(run.data.metrics["test_lightgbm_isotonic_pr_auc"])
    assert run.data.tags["feature_mode"] == "point_in_time"
    artifacts = {a.path for a in client.list_artifacts(result.run_id)}
    assert {"bundle", "report"} <= artifacts


def test_training_refuses_labels_that_arrive_inside_the_next_window(
    events: pl.DataFrame, tmp_path: Path
) -> None:
    inputs = TrainingInputs(
        events,
        small_splits(gap_days=7),
        LABELS,
        fast_model_config(),
        load_rule_config(REPO_ROOT / "configs"),
        "synthetic",
        "unit test",
    )
    with pytest.raises(ValueError, match="maturity gap"):
        train_model(
            inputs,
            results_dir=tmp_path,
            artifacts_dir=tmp_path,
            tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        )

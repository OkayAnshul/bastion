"""MLflow registration: champion and challenger aliases, and loading the champion (ADR-009)."""

from pathlib import Path

import mlflow
import numpy as np
import pytest

from bastion.data.labels import label_events
from bastion.features.batch import compute_features
from bastion.rules.baseline import load_rule_config
from bastion.training.bundle import ModelBundle
from bastion.training.dataset import build_model_frame
from bastion.training.pipeline import split_rows
from bastion.training.registry import CHALLENGER, CHAMPION, load_registered_bundle, register_bundle
from bastion.training.train import TrainingInputs, train_model
from tests.unit.test_training import LABELS, fast_model_config, small_splits, synthetic_events

REPO_ROOT = Path(__file__).resolve().parents[2]
NAME = "bastion-test"


def test_first_version_is_champion_later_ones_challengers(tmp_path: Path) -> None:
    events = synthetic_events()
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    inputs = TrainingInputs(
        events,
        small_splits(),
        LABELS,
        fast_model_config(),
        load_rule_config(REPO_ROOT / "configs"),
        "synthetic",
        "unit test",
    )
    first = train_model(
        inputs,
        results_dir=tmp_path / "results",
        artifacts_dir=tmp_path / "artifacts",
        tracking_uri=uri,
        register_as=NAME,
    )
    assert first.registered is not None
    assert (first.registered.version, first.registered.alias) == ("1", CHAMPION)

    with mlflow.start_run():
        second = register_bundle(first.bundle_dir, NAME)
    assert (second.version, second.alias) == ("2", CHALLENGER)

    champion, label = load_registered_bundle(NAME, CHAMPION)
    assert label == f"{NAME}/1"
    labels = label_events(events, LABELS)
    frame = build_model_frame(
        events, compute_features(events, labels, label_strength=100.0), small_splits()
    )
    test = split_rows(frame, "test")
    np.testing.assert_array_equal(
        champion.predict_proba(test), ModelBundle.load(first.bundle_dir).predict_proba(test)
    )


def test_loading_a_missing_alias_fails_loudly(tmp_path: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    with pytest.raises(mlflow.exceptions.MlflowException):
        load_registered_bundle("no-such-model", CHAMPION)

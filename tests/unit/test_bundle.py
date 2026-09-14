import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from bastion.evaluation.calibration import IsotonicCalibrator
from bastion.training.bundle import SPEC_FILE, ModelBundle
from bastion.training.dataset import fit_feature_spec, to_matrix
from bastion.training.models import LogisticBaseline, predict_scores, train_lightgbm

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.1,
    "num_leaves": 8,
    "min_data_in_leaf": 10,
    "num_threads": 2,
    "deterministic": True,
    "force_row_wise": True,
    "verbosity": -1,
}


def _frame(n: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    velocity = rng.integers(0, 6, n)
    category = rng.choice(["W", "C", "H"], n)
    risk = 0.02 + 0.12 * velocity + 0.2 * (category == "C")
    return pl.DataFrame(
        {
            "amount": rng.lognormal(3, 1, n),
            "merchant_category": category,
            "card_txn_count_1h": velocity,
            "is_fraud": rng.random(n) < np.minimum(risk, 1.0),
        }
    )


@pytest.fixture(scope="module")
def trained() -> tuple[ModelBundle, pl.DataFrame]:
    train, valid, test = _frame(2_000, 0), _frame(600, 1), _frame(600, 2)
    spec = fit_feature_spec(train, ["card_txn_count_1h"], [], 10)
    booster = train_lightgbm(
        to_matrix(train, spec),
        train["is_fraud"].to_numpy(),
        to_matrix(valid, spec),
        valid["is_fraud"].to_numpy(),
        feature_names=spec.columns,
        categorical_features=spec.categorical_columns,
        params=PARAMS,
        seed=7,
        num_boost_round=200,
        early_stopping_rounds=20,
    )
    calibrator = IsotonicCalibrator.fit(
        predict_scores(booster, to_matrix(valid, spec)), valid["is_fraud"].to_numpy()
    )
    return ModelBundle.from_booster(booster, calibrator, spec, {"feature_mode": "test"}), test


def test_bundle_round_trips_with_identical_predictions(
    trained: tuple[ModelBundle, pl.DataFrame], tmp_path: Path
) -> None:
    bundle, test = trained
    restored = ModelBundle.load(bundle.save(tmp_path / "bundle"))
    np.testing.assert_array_equal(restored.predict_proba(test), bundle.predict_proba(test))
    assert restored.metadata["spec_fingerprint"] == bundle.spec.fingerprint()


def test_calibrated_probabilities_are_valid_and_rank_risk(
    trained: tuple[ModelBundle, pl.DataFrame],
) -> None:
    bundle, test = trained
    probabilities = bundle.predict_proba(test)
    assert probabilities.min() >= 0.0
    assert probabilities.max() <= 1.0
    high = test["card_txn_count_1h"].to_numpy() >= 4
    assert probabilities[high].mean() > probabilities[~high].mean()


def test_a_tampered_feature_spec_is_refused(
    trained: tuple[ModelBundle, pl.DataFrame], tmp_path: Path
) -> None:
    directory = trained[0].save(tmp_path / "bundle")
    spec = json.loads((directory / SPEC_FILE).read_text())
    spec["numeric"].reverse()
    (directory / SPEC_FILE).write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="fingerprint"):
        ModelBundle.load(directory)


def test_logistic_baseline_ignores_categorical_codes() -> None:
    train = _frame(1_000, 3)
    spec = fit_feature_spec(train, ["card_txn_count_1h"], [], 10)
    x = to_matrix(train, spec)
    model = LogisticBaseline(numeric_count=len(spec.numeric), seed=7).fit(
        x, train["is_fraud"].to_numpy()
    )
    scrambled = x.copy()
    scrambled[:, len(spec.numeric) :] = 99.0
    np.testing.assert_array_equal(model.predict(scrambled), model.predict(x))

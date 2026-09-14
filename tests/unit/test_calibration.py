import json
import math

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from bastion.evaluation.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    brier_score,
    calibrator_from_dict,
    expected_calibration_error,
    fit_calibrator,
    log_loss,
    reliability_bins,
    reliability_bins_log,
    select_calibration_method,
)


def test_brier_and_log_loss_worked_example() -> None:
    labels = [1, 0, 1, 0]
    probabilities = [0.9, 0.2, 0.6, 0.1]
    assert brier_score(labels, probabilities) == pytest.approx((0.01 + 0.04 + 0.16 + 0.01) / 4)
    expected = -(math.log(0.9) + math.log(0.8) + math.log(0.6) + math.log(0.9)) / 4
    assert log_loss(labels, probabilities) == pytest.approx(expected)


def test_ece_is_zero_when_every_bin_matches_its_observed_rate() -> None:
    probabilities = [0.2] * 10 + [0.8] * 10
    labels = [1, 1] + [0] * 8 + [1] * 8 + [0, 0]
    assert expected_calibration_error(labels, probabilities, n_bins=2) == pytest.approx(0.0)


def test_ece_measures_overconfidence() -> None:
    # Says 90%, happens 50% of the time.
    assert expected_calibration_error([1, 0] * 5, [0.9] * 10, n_bins=1) == pytest.approx(0.4)


def test_reliability_bins_hold_equal_numbers_of_transactions() -> None:
    rng = np.random.default_rng(0)
    bins = reliability_bins(rng.integers(0, 2, 100), rng.random(100), n_bins=15)
    counts = [b.count for b in bins]
    assert sum(counts) == 100
    assert max(counts) - min(counts) <= 1


def test_probabilities_outside_the_unit_interval_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        brier_score([1, 0], [1.2, 0.1])


def test_isotonic_calibrator_matches_scikit_learn_including_out_of_range_scores() -> None:
    rng = np.random.default_rng(1)
    scores = rng.random(500) * 0.8 + 0.1
    labels = rng.random(500) < scores**2
    reference = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(scores, labels)
    new_scores = np.concatenate([[0.0, 0.05, 0.95, 1.0], rng.random(100)])
    np.testing.assert_allclose(
        IsotonicCalibrator.fit(scores, labels).predict(new_scores), reference.predict(new_scores)
    )


@pytest.mark.parametrize("method", ["isotonic", "platt"])
def test_calibrators_round_trip_through_json(method: str) -> None:
    rng = np.random.default_rng(2)
    scores = rng.random(300)
    labels = rng.random(300) < scores
    calibrator = fit_calibrator(method, scores, labels)  # type: ignore[arg-type]
    restored = calibrator_from_dict(json.loads(json.dumps(calibrator.to_dict())))
    probe = np.linspace(0, 1, 50)
    np.testing.assert_array_equal(restored.predict(probe), calibrator.predict(probe))


@pytest.mark.parametrize("method", ["isotonic", "platt"])
def test_calibration_fixes_overconfident_scores(method: str) -> None:
    rng = np.random.default_rng(3)
    true_risk = rng.random(20_000) * 0.2
    labels = rng.random(20_000) < true_risk
    scores = np.sqrt(true_risk)  # ranks correctly but overstates every risk
    calibrator = fit_calibrator(method, scores[:10_000], labels[:10_000])  # type: ignore[arg-type]
    held_out = calibrator.predict(scores[10_000:])
    assert brier_score(labels[10_000:], held_out) < brier_score(labels[10_000:], scores[10_000:])
    # Equal-mass bins of ~667 rows carry about 0.01 of sampling noise on their own, so compare with
    # the raw scores' error rather than an absolute threshold below that noise floor.
    raw_ece = expected_calibration_error(labels[10_000:], scores[10_000:])
    assert expected_calibration_error(labels[10_000:], held_out) < 0.25 * raw_ece


def test_platt_calibrator_is_monotone() -> None:
    calibrator = PlattCalibrator(slope=1.3, intercept=-2.0)
    predictions = calibrator.predict(np.linspace(0.001, 0.999, 100))
    assert np.all(np.diff(predictions) > 0)


def test_selection_rejects_isotonic_when_it_assigns_zero_risk_to_real_fraud() -> None:
    rng = np.random.default_rng(4)
    scores = rng.random(4_000)
    early = np.arange(4_000) < 2_800
    # Early rows: fraud only at high scores. Later rows: some fraud at every score.
    labels = np.where(early, (scores > 0.8) & (rng.random(4_000) < 0.5), rng.random(4_000) < 0.1)
    method, losses = select_calibration_method(scores, labels, holdout_fraction=0.3)
    assert method == "platt"
    assert losses["isotonic"] > losses["platt"]


def test_selection_needs_fraud_on_both_sides_of_the_holdout_cut() -> None:
    scores = np.linspace(0, 1, 100)
    labels = np.arange(100) < 5  # all fraud in the fitting part
    with pytest.raises(ValueError, match="both"):
        select_calibration_method(scores, labels, holdout_fraction=0.3)


def test_log_bins_give_the_rare_high_risk_tail_its_own_bins() -> None:
    rng = np.random.default_rng(5)
    probabilities = np.concatenate([rng.random(9_900) * 0.01, 0.5 + rng.random(100) * 0.4])
    labels = rng.random(10_000) < probabilities
    equal_mass = reliability_bins(labels, probabilities, n_bins=15)
    log_spaced = reliability_bins_log(labels, probabilities, min_count=20)
    assert sum(b.mean_predicted > 0.1 for b in equal_mass) <= 1  # the tail is diluted into one bin
    assert max(b.mean_predicted for b in log_spaced) > 0.5  # log spacing keeps it separate


def test_log_bins_drop_bins_too_sparse_to_estimate() -> None:
    assert reliability_bins_log([1, 0, 1], [0.9, 0.5, 0.2], min_count=50) == []

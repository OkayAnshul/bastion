"""Probability calibration (ADR-006).

The policy engine multiplies probabilities by amounts, so a score of 0.7 has to mean a 70% fraud
risk. Boosted trees are not calibrated out of the box. This module measures calibration (Brier
score, log loss, expected calibration error, reliability bins) and fits isotonic and Platt
calibrators. The calibrators serialise to plain JSON, so serving needs neither pickles nor
scikit-learn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

FloatArray = npt.NDArray[np.float64]
CalibrationMethod = Literal["isotonic", "platt"]
_LOG_EPS = 1e-15
_LOGIT_EPS = 1e-9


def _checked(labels: npt.ArrayLike, probabilities: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError(f"shape mismatch: labels {y.shape}, probabilities {p.shape}")
    if p.size and (np.nanmin(p) < 0 or np.nanmax(p) > 1):
        raise ValueError("probabilities must lie in [0, 1]")
    return y, p


def brier_score(labels: npt.ArrayLike, probabilities: npt.ArrayLike) -> float:
    """Mean squared error between probability and outcome. Lower is better."""
    y, p = _checked(labels, probabilities)
    return float(np.mean((p - y) ** 2)) if y.size else float("nan")


def log_loss(labels: npt.ArrayLike, probabilities: npt.ArrayLike) -> float:
    y, p = _checked(labels, probabilities)
    if not y.size:
        return float("nan")
    p = np.clip(p, _LOG_EPS, 1 - _LOG_EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


@dataclass(frozen=True)
class ReliabilityBin:
    mean_predicted: float
    observed_rate: float
    count: int


def reliability_bins(
    labels: npt.ArrayLike, probabilities: npt.ArrayLike, n_bins: int = 15
) -> list[ReliabilityBin]:
    """Equal-mass bins of transactions sorted by predicted probability.

    With fraud this rare, fixed-width bins leave every bin above ~0.1 almost empty. Equal-mass bins
    give the high-risk tail bins of its own, each with enough transactions to estimate a rate.
    """
    y, p = _checked(labels, probabilities)
    if not y.size:
        return []
    order = np.argsort(p, kind="stable")
    return [
        ReliabilityBin(float(p[idx].mean()), float(y[idx].mean()), int(idx.size))
        for idx in np.array_split(order, min(n_bins, y.size))
        if idx.size
    ]


def expected_calibration_error(
    labels: npt.ArrayLike, probabilities: npt.ArrayLike, n_bins: int = 15
) -> float:
    """Count-weighted mean gap between predicted and observed rates over equal-mass bins."""
    bins = reliability_bins(labels, probabilities, n_bins)
    total = sum(b.count for b in bins)
    if not total:
        return float("nan")
    return sum(b.count / total * abs(b.mean_predicted - b.observed_rate) for b in bins)


def _logit(scores: npt.ArrayLike) -> FloatArray:
    p = np.clip(np.asarray(scores, dtype=np.float64), _LOGIT_EPS, 1 - _LOGIT_EPS)
    return np.log(p / (1 - p))


@dataclass(frozen=True)
class IsotonicCalibrator:
    """Monotone step mapping: linear between fitted thresholds, clipped outside them."""

    x_thresholds: tuple[float, ...]
    y_thresholds: tuple[float, ...]
    method: Literal["isotonic"] = "isotonic"
    _x: FloatArray = field(init=False, repr=False, compare=False)
    _y: FloatArray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_x", np.asarray(self.x_thresholds, dtype=np.float64))
        object.__setattr__(self, "_y", np.asarray(self.y_thresholds, dtype=np.float64))

    @classmethod
    def fit(cls, scores: npt.ArrayLike, labels: npt.ArrayLike) -> IsotonicCalibrator:
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.float64))
        return cls(
            tuple(float(v) for v in model.X_thresholds_),
            tuple(float(v) for v in model.y_thresholds_),
        )

    def predict(self, scores: npt.ArrayLike) -> FloatArray:
        # np.interp clamps outside the fitted range, like scikit-learn's out_of_bounds="clip".
        return np.interp(np.asarray(scores, dtype=np.float64), self._x, self._y)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "x_thresholds": list(self.x_thresholds),
            "y_thresholds": list(self.y_thresholds),
        }


@dataclass(frozen=True)
class PlattCalibrator:
    """Logistic regression on the logit of the raw score: two parameters, smooth, monotone."""

    slope: float
    intercept: float
    method: Literal["platt"] = "platt"

    @classmethod
    def fit(cls, scores: npt.ArrayLike, labels: npt.ArrayLike) -> PlattCalibrator:
        # Large C: effectively unregularised, as in Platt's original method.
        model = LogisticRegression(C=1e6, max_iter=1_000)
        model.fit(_logit(scores).reshape(-1, 1), np.asarray(labels, dtype=np.int64))
        return cls(float(model.coef_[0, 0]), float(model.intercept_[0]))

    def predict(self, scores: npt.ArrayLike) -> FloatArray:
        return 1.0 / (1.0 + np.exp(-(self.slope * _logit(scores) + self.intercept)))

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "slope": self.slope, "intercept": self.intercept}


Calibrator = IsotonicCalibrator | PlattCalibrator


def fit_calibrator(
    method: CalibrationMethod, scores: npt.ArrayLike, labels: npt.ArrayLike
) -> Calibrator:
    if method == "isotonic":
        return IsotonicCalibrator.fit(scores, labels)
    return PlattCalibrator.fit(scores, labels)


def calibrator_from_dict(data: dict[str, Any]) -> Calibrator:
    if data["method"] == "isotonic":
        return IsotonicCalibrator(tuple(data["x_thresholds"]), tuple(data["y_thresholds"]))
    if data["method"] == "platt":
        return PlattCalibrator(float(data["slope"]), float(data["intercept"]))
    raise ValueError(f"unknown calibration method {data['method']!r}")


def select_calibration_method(
    scores: npt.ArrayLike,
    labels: npt.ArrayLike,
    *,
    holdout_fraction: float,
    candidates: tuple[CalibrationMethod, ...] = ("isotonic", "platt"),
) -> tuple[CalibrationMethod, dict[str, float]]:
    """Choose a calibrator by log loss on the most recent part of the calibration rows.

    Rows must be in time order. Each candidate is fit on the earlier ``1 - holdout_fraction`` and
    scored on the rest; the caller refits the winner on all rows. Log loss punishes the failure that
    matters to a cost-based policy: isotonic regression assigning a probability of exactly zero to a
    score region where fraud does happen. Ties go to the earlier candidate.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.bool_)
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must lie strictly between 0 and 1")
    cut = round(s.size * (1 - holdout_fraction))
    if not y[:cut].any() or not y[cut:].any():
        raise ValueError("calibration rows need fraud in both the fitting part and the holdout")
    losses: dict[str, float] = {
        method: log_loss(y[cut:], fit_calibrator(method, s[:cut], y[:cut]).predict(s[cut:]))
        for method in candidates
    }
    return min(candidates, key=lambda method: losses[method]), losses


def reliability_bins_log(
    labels: npt.ArrayLike,
    probabilities: npt.ArrayLike,
    *,
    floor: float = 1e-4,
    bins_per_decade: int = 4,
    min_count: int = 50,
) -> list[ReliabilityBin]:
    """Bins evenly spaced in log-probability, for plotting the calibration of rare events.

    With fraud near 1%, equal-mass bins (used for ECE) put almost every bin in the low-risk bulk.
    Log spacing gives the high-risk tail bins of its own. Probabilities below ``floor``, including
    exact zeros, share the first bin. Bins with fewer than ``min_count`` rows are dropped as too
    noisy to plot.
    """
    y, p = _checked(labels, probabilities)
    decades = int(np.ceil(-np.log10(floor)))
    edges = np.logspace(np.log10(floor), 0.0, decades * bins_per_decade + 1)
    edges[0] = 0.0
    index = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, edges.size - 2)
    bins = []
    for b in range(edges.size - 1):
        mask = index == b
        count = int(mask.sum())
        if count >= min_count:
            bins.append(ReliabilityBin(float(p[mask].mean()), float(y[mask].mean()), count))
    return bins

"""Trainers: the logistic-regression baseline and LightGBM, the primary model (ADR-002)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

FloatArray = npt.NDArray[np.float64]


def train_lightgbm(
    x_train: npt.NDArray[np.float32],
    y_train: npt.NDArray[np.bool_],
    x_valid: npt.NDArray[np.float32],
    y_valid: npt.NDArray[np.bool_],
    *,
    feature_names: Sequence[str],
    categorical_features: Sequence[str],
    params: Mapping[str, Any],
    seed: int,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> lgb.Booster:
    """Gradient-boosted trees with early stopping on PR-AUC (average precision) of ``x_valid``.

    No class weighting: reweighting distorts probabilities, and cost asymmetry is handled downstream
    by calibration and the expected-loss policy (ADR-005, ADR-006).
    """
    train_set = lgb.Dataset(
        x_train,
        label=y_train.astype(np.float64),
        feature_name=list(feature_names),
        categorical_feature=list(categorical_features),
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(x_valid, label=y_valid.astype(np.float64), reference=train_set)
    return lgb.train(
        {**params, "seed": seed, "metric": "average_precision"},
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[valid_set],
        valid_names=["early_stopping"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, first_metric_only=True, verbose=False)
        ],
    )


def predict_scores(booster: lgb.Booster, x: npt.NDArray[np.float32]) -> FloatArray:
    """Uncalibrated fraud scores from the best iteration."""
    return np.asarray(booster.predict(x, num_iteration=booster.best_iteration), dtype=np.float64)


class LogisticBaseline:
    """Logistic regression on the numeric inputs: the simpler model LightGBM must beat.

    Categorical codes are excluded because a linear model would read them as magnitudes.
    """

    def __init__(self, numeric_count: int, seed: int) -> None:
        self.numeric_count = numeric_count
        self.pipeline: Pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2_000, random_state=seed),
        )

    def fit(self, x: npt.NDArray[np.float32], y: npt.NDArray[np.bool_]) -> LogisticBaseline:
        self.pipeline.fit(x[:, : self.numeric_count], y.astype(np.int64))
        return self

    def predict(self, x: npt.NDArray[np.float32]) -> FloatArray:
        return np.asarray(
            self.pipeline.predict_proba(x[:, : self.numeric_count])[:, 1], dtype=np.float64
        )

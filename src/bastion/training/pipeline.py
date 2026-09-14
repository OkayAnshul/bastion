"""Model-building steps shared by the training run and the leakage experiment."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.metrics import average_precision_score

from bastion.data.splits import SPLIT_COLUMN
from bastion.training.dataset import FeatureSpec, ModelConfig, fit_feature_spec, to_matrix
from bastion.training.models import predict_scores, train_lightgbm


def split_rows(frame: pl.DataFrame, name: str) -> pl.DataFrame:
    rows = frame.filter(pl.col(SPLIT_COLUMN) == name)
    if rows.is_empty():
        raise ValueError(f"split window {name!r} has no rows")
    return rows


def labels_of(frame: pl.DataFrame) -> npt.NDArray[np.bool_]:
    if frame["is_fraud"].null_count():
        raise ValueError("model rows must all be labelled")
    return frame["is_fraud"].to_numpy().astype(np.bool_)


@dataclass(frozen=True)
class FittedModel:
    booster: lgb.Booster
    spec: FeatureSpec

    def scores(self, frame: pl.DataFrame) -> npt.NDArray[np.float64]:
        return predict_scores(self.booster, to_matrix(frame, self.spec))


def fit_lightgbm_on_frame(
    frame: pl.DataFrame,
    feature_columns: Sequence[str],
    config: ModelConfig,
    *,
    train_window: str = "train",
    valid_window: str = "early_stopping",
) -> FittedModel:
    """Fit the feature spec and LightGBM on ``train_window``; early-stop on ``valid_window``."""
    train = split_rows(frame, train_window)
    valid = split_rows(frame, valid_window)
    spec = fit_feature_spec(train, feature_columns, config.passthrough, config.max_category_levels)
    booster = train_lightgbm(
        to_matrix(train, spec),
        labels_of(train),
        to_matrix(valid, spec),
        labels_of(valid),
        feature_names=spec.columns,
        categorical_features=spec.categorical_columns,
        params=config.lightgbm,
        seed=config.seed,
        num_boost_round=config.num_boost_round,
        early_stopping_rounds=config.early_stopping_rounds,
    )
    return FittedModel(booster, spec)


def bootstrap_pr_auc(
    labels: npt.ArrayLike, scores: npt.ArrayLike, *, resamples: int, seed: int
) -> tuple[float, float]:
    """95% percentile interval of PR-AUC over bootstrap resamples of the evaluated rows.

    Resamples without both classes are skipped. With no resamples or one class, returns NaNs.
    """
    y = np.asarray(labels, dtype=np.bool_)
    s = np.asarray(scores, dtype=np.float64)
    if resamples == 0 or not y.any() or y.all():
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        idx = rng.integers(0, y.size, y.size)
        if y[idx].any() and not y[idx].all():
            values.append(float(average_precision_score(y[idx], s[idx])))
    if not values:
        return float("nan"), float("nan")
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)

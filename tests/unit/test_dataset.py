import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from bastion.training.dataset import (
    FeatureSpec,
    check_label_maturity,
    fit_feature_spec,
    load_model_config,
    select_passthrough,
    to_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "txn_id": ["t0", "t1", "t2", "t3", "t4"],
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0],
            "merchant_category": ["W", "W", "C", "W", "H"],
            "card_txn_count_1h": [0, 1, 2, 0, 1],
            "card_device_seen_before": [True, None, False, True, True],
            "attr_C1": [1.0, 2.0, None, 4.0, 5.0],
            "attr_M4": ["M0", "M0", "M1", "M2", None],
            "attr_V1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "split": ["train", "train", "train", "test", "test"],
        }
    )


def test_passthrough_selects_attributes_by_pattern() -> None:
    columns = _frame().columns
    assert select_passthrough(columns, ["^C\\d+$", "^M\\d+$"]) == ["attr_C1", "attr_M4"]


def test_spec_types_and_vocabularies_come_from_the_train_window() -> None:
    frame = _frame()
    train = frame.filter(pl.col("split") == "train")
    spec = fit_feature_spec(
        train, ["card_txn_count_1h", "card_device_seen_before"], ["^C1$", "^M4$"], 10
    )
    assert spec.numeric == ("amount", "card_txn_count_1h", "card_device_seen_before", "attr_C1")
    assert dict(spec.categorical) == {"merchant_category": ("W", "C"), "attr_M4": ("M0", "M1")}

    matrix = to_matrix(frame.filter(pl.col("split") == "test"), spec)
    columns = list(spec.columns)
    m4 = matrix[:, columns.index("attr_M4")]
    assert m4[0] == 2.0  # "M2" never appeared in training: the shared "other" code
    assert math.isnan(m4[1])  # null stays missing
    assert matrix[0, columns.index("merchant_category")] == 0.0  # "W"
    assert matrix[1, columns.index("merchant_category")] == 2.0  # "H" unseen in training


def test_vocabulary_keeps_the_most_frequent_levels() -> None:
    train = _frame().filter(pl.col("split") == "train")
    spec = fit_feature_spec(train, [], ["^M4$"], max_category_levels=1)
    assert dict(spec.categorical)["attr_M4"] == ("M0",)


def test_fingerprint_changes_with_the_input_layout() -> None:
    train = _frame().filter(pl.col("split") == "train")
    wide = fit_feature_spec(train, [], ["^M4$"], 10)
    narrow = fit_feature_spec(train, [], ["^M4$"], 1)
    assert wide.fingerprint() != narrow.fingerprint()
    assert FeatureSpec.from_dict(wide.to_dict()).fingerprint() == wide.fingerprint()


def test_label_maturity_check_rejects_labels_arriving_inside_the_next_window() -> None:
    ts = pl.Datetime("ms", "UTC")
    frame = pl.DataFrame(
        {
            "txn_id": ["a", "b"],
            "event_ts": pl.Series([T0, T0 + timedelta(days=20)], dtype=ts),
            "split": ["train", "early_stopping"],
        }
    )

    def labels(arrival: timedelta) -> pl.DataFrame:
        return pl.DataFrame({"txn_id": ["a"], "label_ts": pl.Series([T0 + arrival], dtype=ts)})

    check_label_maturity(
        frame, labels(timedelta(days=14)), train_window="train", next_window="early_stopping"
    )
    with pytest.raises(ValueError, match="maturity gap"):
        check_label_maturity(
            frame, labels(timedelta(days=21)), train_window="train", next_window="early_stopping"
        )


def test_repository_model_config_is_valid() -> None:
    config = load_model_config(REPO_ROOT / "configs")
    assert config.calibration == "select"
    assert config.lightgbm["objective"] == "binary"

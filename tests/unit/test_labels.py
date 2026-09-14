from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from bastion.data.labels import LabelDelayConfig, label_events, load_label_delay_config
from bastion.data.splits import load_split_config
from bastion.schemas.events import LabelEvent

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = LabelDelayConfig(seed=11, fraud_median_days=5.0, fraud_sigma=0.6, maturity_days=14.0)
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _events(n: int, fraud_every: int = 3) -> pl.DataFrame:
    frame = pl.DataFrame(
        {
            "txn_id": [f"t{i}" for i in range(n)],
            "card_id": [f"c{i % 7}" for i in range(n)],
            "merchant_id": [f"m{i % 5}" for i in range(n)],
            "event_ts": pl.Series(
                [T0 + timedelta(minutes=i) for i in range(n)], dtype=pl.Datetime("ms", "UTC")
            ),
            "is_fraud": [i % fraud_every == 0 for i in range(n)],
        }
    )
    return frame


def _delays(labels: pl.DataFrame) -> pl.DataFrame:
    return labels.select(
        "txn_id",
        "is_fraud",
        ((pl.col("label_ts") - pl.col("event_ts")).dt.total_milliseconds() / 86_400_000).alias(
            "days"
        ),
    )


def test_legitimate_labels_mature_exactly_after_the_window() -> None:
    delays = _delays(label_events(_events(30), CONFIG)).filter(~pl.col("is_fraud"))
    assert delays["days"].to_list() == [14.0] * delays.height


def test_fraud_labels_arrive_after_a_positive_capped_delay() -> None:
    delays = _delays(label_events(_events(300), CONFIG)).filter(pl.col("is_fraud"))["days"]
    assert delays.min() > 0  # type: ignore[operator]
    assert delays.max() <= 14.0  # type: ignore[operator]


def test_label_times_do_not_depend_on_row_order_or_on_other_rows() -> None:
    events = _events(60)
    full = label_events(events, CONFIG).sort("txn_id")
    shuffled = label_events(events.sample(fraction=1.0, shuffle=True, seed=1), CONFIG).sort(
        "txn_id"
    )
    subset = label_events(events.filter(pl.col("txn_id").is_in(["t3", "t9", "t10"])), CONFIG)
    assert full.equals(shuffled)
    assert subset.sort("txn_id").equals(full.filter(pl.col("txn_id").is_in(["t3", "t9", "t10"])))


def test_seed_changes_fraud_delays() -> None:
    events = _events(60)
    other = CONFIG.model_copy(update={"seed": 12})
    assert not label_events(events, CONFIG).equals(label_events(events, other))


def test_unlabelled_transactions_produce_no_label() -> None:
    events = _events(6).with_columns(
        pl.when(pl.col("txn_id") == "t1").then(None).otherwise(pl.col("is_fraud")).alias("is_fraud")
    )
    assert "t1" not in label_events(events, CONFIG)["txn_id"].to_list()


def test_delay_distribution_matches_the_configuration() -> None:
    delays = _delays(label_events(_events(6_000, fraud_every=1), CONFIG))["days"].to_numpy()
    assert np.median(delays) == pytest.approx(5.0, rel=0.05)
    # P(lognormal > 14) = P(z > ln(14/5)/0.6) ≈ 4.3%; those draws are capped at exactly 14 days.
    assert 0.02 < np.mean(delays == 14.0) < 0.07


def test_every_label_satisfies_the_label_event_contract() -> None:
    for row in label_events(_events(40), CONFIG).iter_rows(named=True):
        LabelEvent.model_validate(row)


def test_repository_label_config_fits_inside_the_split_maturity_gap() -> None:
    labels = load_label_delay_config(REPO_ROOT / "configs")
    gap = load_split_config(REPO_ROOT / "configs").window("maturity_gap")
    assert gap.end_day is not None
    assert labels.maturity_days <= gap.end_day - gap.start_day

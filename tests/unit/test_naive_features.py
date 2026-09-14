import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from bastion.data.labels import LabelDelayConfig, label_events
from bastion.features.batch import compute_features
from bastion.features.naive import compute_naive_features
from bastion.streaming.synthetic import SyntheticConfig, generate

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src" / "bastion"
LABELS = LabelDelayConfig(seed=11, fraud_median_days=5.0, fraud_sigma=0.6, maturity_days=14.0)
T0 = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)


def _events(rows: list[tuple[str, int, str, bool]]) -> pl.DataFrame:
    """rows: (card_id, seconds after T0, merchant_id, is_fraud)."""
    return pl.DataFrame(
        {
            "txn_id": [f"t{i}" for i in range(len(rows))],
            "event_ts": pl.Series(
                [T0 + timedelta(seconds=s) for _, s, _, _ in rows], dtype=pl.Datetime("ms", "UTC")
            ),
            "card_id": [r[0] for r in rows],
            "device_id": pl.Series([None] * len(rows), dtype=pl.String),
            "merchant_id": [r[2] for r in rows],
            "merchant_category": ["W"] * len(rows),
            "amount": [10.0] * len(rows),
            "is_fraud": [r[3] for r in rows],
        }
    )


def test_naive_pipeline_emits_the_same_columns_and_types() -> None:
    events = generate(SyntheticConfig(days=10, n_cards=100, n_merchants=20))
    pit = compute_features(events, label_events(events, LABELS))
    naive = compute_naive_features(events, with_labels=True)
    assert naive.schema == pit.schema
    assert naive.height == pit.height


def test_naive_features_change_when_future_events_are_deleted() -> None:
    events = _events([("c", 0, "m1", False), ("c", 30, "m1", False)])
    first_only = events.head(1)

    def first_count(features: pl.DataFrame) -> object:
        return features["card_txn_count_1h"][0]

    # The honest pipeline gives the first transaction the same value either way...
    assert first_count(compute_features(events)) == first_count(compute_features(first_only)) == 0
    # ...the naive one counts the later transaction (and itself) inside the bucket.
    assert first_count(compute_naive_features(events)) == 2
    assert first_count(compute_naive_features(first_only)) == 1


def test_naive_label_features_read_the_rows_own_label() -> None:
    events = _events([("c", 0, "m_only_fraud", True), ("k", 60, "m_other", False)])
    naive = compute_naive_features(events, with_labels=True).row(0, named=True)
    honest = compute_features(events, label_events(events, LABELS)).row(0, named=True)
    assert naive["card_prior_fraud_labels"] == 1  # its own chargeback, before it could exist
    assert naive["merchant_known_fraud_rate"] == 1.0
    assert honest["card_prior_fraud_labels"] == 0
    assert honest["merchant_known_labels"] == 0


def test_only_the_leakage_experiment_imports_the_naive_pipeline() -> None:
    pattern = re.compile(r"features\.naive|from bastion\.features import .*\bnaive\b")
    offenders = [
        path.relative_to(REPO_ROOT)
        for path in SRC.rglob("*.py")
        if path.name != "naive.py"
        and "experiments" not in path.relative_to(SRC).parts
        and pattern.search(path.read_text())
    ]
    assert offenders == []

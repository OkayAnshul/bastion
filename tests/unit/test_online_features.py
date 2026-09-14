"""Batch/online parity on an in-memory Redis (ADR-004).

The same checks run against a real Redis fed through Redpanda in tests/integration.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import fakeredis
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bastion.data.labels import LabelDelayConfig, label_events
from bastion.features.batch import compute_features, feature_schema
from bastion.features.online import (
    CARD_HISTORY_MS,
    OnlineFeatureStore,
    OnlineStoreConfig,
    epoch_ms,
)
from bastion.schemas.events import LabelEvent, TransactionEvent
from bastion.schemas.tables import row_to_event
from bastion.streaming.config import load_streaming_config
from bastion.streaming.synthetic import SyntheticConfig, generate

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS = LabelDelayConfig(seed=11, fraud_median_days=5.0, fraud_sigma=0.6, maturity_days=14.0)
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _store(**overrides: object) -> OnlineFeatureStore:
    config = OnlineStoreConfig(**overrides)  # type: ignore[arg-type]
    return OnlineFeatureStore(fakeredis.FakeRedis(decode_responses=True), config)


def _transactions(events: pl.DataFrame) -> list[TransactionEvent]:
    return [row_to_event(row) for row in events.iter_rows(named=True)]


def _labels(labels: pl.DataFrame) -> list[LabelEvent]:
    return [LabelEvent.model_validate(row) for row in labels.iter_rows(named=True)]


def _online_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={"txn_id": pl.String, **feature_schema(with_labels=True)})


def _bulk_parity(events: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Write everything first, then read each transaction as of its own time."""
    labels = label_events(events, LABELS)
    store = _store(trim_history=False)
    for label in _labels(labels):
        store.write_label(label)
    transactions = _transactions(events)
    for event in transactions:
        store.write_transaction(event)
    online = _online_frame([{"txn_id": e.txn_id, **store.read_features(e)} for e in transactions])
    return online, compute_features(events, labels)


@pytest.fixture(scope="module")
def events() -> pl.DataFrame:
    return generate(
        SyntheticConfig(
            days=40,
            n_cards=60,
            n_merchants=15,
            card_testing_attacks_per_day=2.0,
            account_takeovers_per_day=2.0,
        )
    )


def test_bulk_ingest_then_as_of_reads_match_batch_exactly(events: pl.DataFrame) -> None:
    online, batch = _bulk_parity(events)
    assert online.equals(batch)


def test_interleaved_replay_with_trimming_matches_batch_exactly(events: pl.DataFrame) -> None:
    """Production order: read a transaction's features, then ingest it; labels arrive in between."""
    labels = label_events(events, LABELS)
    timeline: list[tuple[int, int, int, TransactionEvent | LabelEvent]] = []
    for i, label in enumerate(_labels(labels)):
        timeline.append((epoch_ms(label.label_ts), 0, i, label))
    for i, event in enumerate(_transactions(events)):
        timeline.append((epoch_ms(event.event_ts), 1, i, event))
    timeline.sort(key=lambda item: item[:3])

    store = _store(trim_history=True)
    rows: list[dict[str, object]] = []
    for _, _, _, item in timeline:
        if isinstance(item, LabelEvent):
            store.write_label(item)
        else:
            rows.append({"txn_id": item.txn_id, **store.read_features(item)})
            store.write_transaction(item)
    assert _online_frame(rows).equals(compute_features(events, labels))


def test_card_history_is_trimmed_to_the_longest_window() -> None:
    store = _store(trim_history=True)
    old = _event("t1", T0)
    new = _event("t2", T0 + timedelta(milliseconds=CARD_HISTORY_MS + 1))
    store.write_transaction(old)
    store.write_transaction(new)
    assert store.client.zcard("bastion:card:c1:events") == 1


def test_history_keys_expire_after_the_window_plus_slack() -> None:
    store = _store(history_ttl_slack_days=2)
    store.write_transaction(_event("t1", T0))
    ttl = store.client.ttl("bastion:card:c1:events")
    assert 0 < ttl <= CARD_HISTORY_MS // 1000 + 2 * 86_400


def test_writes_are_idempotent() -> None:
    once, twice = _store(), _store()
    event = _event("t1", T0, device="d1")
    label = LabelEvent(
        txn_id="t1",
        card_id="c1",
        merchant_id="m1",
        event_ts=T0,
        label_ts=T0 + timedelta(days=1),
        is_fraud=True,
    )
    for store, repeats in ((once, 1), (twice, 2)):
        for _ in range(repeats):
            store.write_transaction(event)
            store.write_label(label)
    probe = _event("t2", T0 + timedelta(days=3), device="d1")
    assert once.read_features(probe) == twice.read_features(probe)


def test_first_seen_keeps_the_earliest_time_when_writes_arrive_out_of_order() -> None:
    store = _store()
    store.write_transaction(_event("late", T0 + timedelta(hours=2), card="c2", device="d1"))
    store.write_transaction(_event("early", T0, card="c1", device="d1"))
    assert store.client.zscore("bastion:first_seen:device", "d1") == epoch_ms(T0)


def test_repository_streaming_config_is_valid() -> None:
    config = load_streaming_config(REPO_ROOT / "configs")
    assert config.topics.transactions.name == "transactions.raw"
    assert config.online_store.trim_history


def _event(
    txn_id: str, ts: datetime, *, card: str = "c1", device: str | None = None, amount: float = 10.0
) -> TransactionEvent:
    return TransactionEvent(
        txn_id=txn_id,
        event_ts=ts,
        card_id=card,
        device_id=device,
        merchant_id="m1",
        merchant_category="W",
        amount=amount,
        currency="USD",
        channel="ecom",
    )


DAY = 86_400
random_rows = st.lists(
    st.tuples(
        st.sampled_from(["c1", "c2"]),
        st.integers(min_value=0, max_value=45 * DAY),
        st.floats(min_value=0.01, max_value=5_000, allow_nan=False),
        st.sampled_from([None, "d1", "d2"]),
        st.booleans(),
        st.sampled_from(["m1", "m2"]),
    ),
    min_size=1,
    max_size=40,
)


@settings(max_examples=40, deadline=None)
@given(rows=random_rows)
def test_online_features_equal_batch_features_on_random_logs(
    rows: list[tuple[str, int, float, str | None, bool, str]],
) -> None:
    ordered = sorted(enumerate(rows), key=lambda item: (item[1][1], item[0]))
    events = pl.DataFrame(
        {
            "txn_id": [f"t{i}" for i, _ in ordered],
            "event_ts": pl.Series(
                [T0 + timedelta(seconds=r[1]) for _, r in ordered], dtype=pl.Datetime("ms", "UTC")
            ),
            "card_id": [r[0] for _, r in ordered],
            "device_id": pl.Series([r[3] for _, r in ordered], dtype=pl.String),
            "merchant_id": [r[5] for _, r in ordered],
            "merchant_category": ["W"] * len(ordered),
            "ip": pl.Series([None] * len(ordered), dtype=pl.String),
            "amount": [r[2] for _, r in ordered],
            "currency": ["USD"] * len(ordered),
            "channel": ["ecom"] * len(ordered),
            "is_fraud": [r[4] for _, r in ordered],
        }
    )
    online, batch = _bulk_parity(events)
    assert online.equals(batch)

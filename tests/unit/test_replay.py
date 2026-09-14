from datetime import UTC, datetime, timedelta

import fakeredis
import polars as pl
import pytest

from bastion.data.labels import LabelDelayConfig, label_events
from bastion.features.online import OnlineFeatureStore, OnlineStoreConfig
from bastion.schemas.events import LabelEvent, TransactionEvent
from bastion.streaming.config import TopicConfig, Topics
from bastion.streaming.feature_builder import decode_message
from bastion.streaming.replay import StoreSink, replay, timeline
from bastion.streaming.synthetic import SyntheticConfig, generate

LABELS = LabelDelayConfig(seed=11, fraud_median_days=5.0, fraud_sigma=0.6, maturity_days=14.0)
TOPICS = Topics(
    transactions=TopicConfig(name="transactions.raw", partitions=6),
    labels=TopicConfig(name="transactions.labels", partitions=6),
)


class ListSink:
    def __init__(self) -> None:
        self.items: list[TransactionEvent | LabelEvent] = []
        self.flushed = False

    def send_transaction(self, event: TransactionEvent) -> None:
        self.items.append(event)

    def send_label(self, label: LabelEvent) -> None:
        self.items.append(label)

    def flush(self) -> None:
        self.flushed = True


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture(scope="module")
def events() -> pl.DataFrame:
    return generate(SyntheticConfig(days=20, n_cards=40, n_merchants=10))


def _time(item: TransactionEvent | LabelEvent) -> datetime:
    return item.event_ts if isinstance(item, TransactionEvent) else item.label_ts


def test_timeline_interleaves_labels_at_their_arrival_time(events: pl.DataFrame) -> None:
    labels = label_events(events, LABELS)
    sink = ListSink()
    stats = replay(timeline(events, labels), sink)
    assert stats.transactions == events.height
    assert stats.labels == labels.height
    times = [_time(item) for item in sink.items]
    assert times == sorted(times)
    assert sink.flushed


def test_labels_never_precede_their_transaction(events: pl.DataFrame) -> None:
    sink = ListSink()
    replay(timeline(events, label_events(events, LABELS)), sink)
    seen: set[str] = set()
    for item in sink.items:
        if isinstance(item, TransactionEvent):
            seen.add(item.txn_id)
        else:
            assert item.txn_id in seen


def test_replay_paces_by_event_time() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [("a", t0), ("b", t0 + timedelta(hours=1)), ("c", t0 + timedelta(hours=3))]
    events = pl.DataFrame(
        {
            "txn_id": [r[0] for r in rows],
            "event_ts": pl.Series([r[1] for r in rows], dtype=pl.Datetime("ms", "UTC")),
            "card_id": ["c"] * 3,
            "device_id": pl.Series([None] * 3, dtype=pl.String),
            "merchant_id": ["m"] * 3,
            "merchant_category": ["W"] * 3,
            "ip": pl.Series([None] * 3, dtype=pl.String),
            "amount": [1.0] * 3,
            "currency": ["USD"] * 3,
            "channel": ["ecom"] * 3,
            "is_fraud": [False] * 3,
        }
    )
    clock = FakeClock()
    stats = replay(timeline(events), ListSink(), speedup=3600, clock=clock.clock, sleep=clock.sleep)
    # One event-hour per wall-second: the gaps of 1 h and 2 h become sleeps of 1 s and 2 s.
    assert clock.sleeps == pytest.approx([1.0, 2.0])
    assert stats.event_span_ms == 3 * 3_600_000


def test_limit_caps_transactions(events: pl.DataFrame) -> None:
    sink = ListSink()
    stats = replay(timeline(events), sink, limit=25)
    assert stats.transactions == 25
    assert sum(isinstance(i, TransactionEvent) for i in sink.items) == 25


def test_store_sink_feeds_the_online_store(events: pl.DataFrame) -> None:
    store = OnlineFeatureStore(fakeredis.FakeRedis(decode_responses=True), OnlineStoreConfig())
    replay(timeline(events, label_events(events, LABELS)), StoreSink(store))
    assert store.client.zcard("bastion:first_seen:card") == events["card_id"].n_unique()


def test_messages_round_trip_through_their_topic(events: pl.DataFrame) -> None:
    sink = ListSink()
    replay(timeline(events.head(3), label_events(events.head(3), LABELS)), sink)
    for item in sink.items:
        topic = (
            TOPICS.transactions.name if isinstance(item, TransactionEvent) else TOPICS.labels.name
        )
        assert decode_message(topic, item.model_dump_json().encode(), TOPICS) == item
    with pytest.raises(ValueError, match="unexpected topic"):
        decode_message("other", b"{}", TOPICS)

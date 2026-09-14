"""Batch/stream parity through real Redpanda and Redis (ADR-004, ROADMAP Phase 2 exit criterion).

Run with ``make up`` then ``make test-integration``; CI runs it on every push. Each test uses its
own topics, consumer group and Redis key prefix, so runs never see each other's data.
"""

import os
import threading
import uuid

import polars as pl
import pytest
import redis

from bastion.data.labels import LabelDelayConfig, label_events
from bastion.features.batch import compute_features, feature_schema
from bastion.features.online import OnlineFeatureStore, OnlineStoreConfig
from bastion.schemas.events import TransactionEvent
from bastion.streaming.config import TopicConfig, Topics
from bastion.streaming.feature_builder import run_feature_builder
from bastion.streaming.replay import KafkaSink, replay, timeline
from bastion.streaming.synthetic import SyntheticConfig, generate
from bastion.streaming.topics import ensure_topics

pytestmark = pytest.mark.integration

BOOTSTRAP = os.environ.get("BASTION_KAFKA_BOOTSTRAP", "localhost:19092")
REDIS_URL = os.environ.get("BASTION_REDIS_URL", "redis://localhost:6379/0")
LABELS = LabelDelayConfig(seed=11, fraud_median_days=5.0, fraud_sigma=0.6, maturity_days=14.0)


def _isolated(trim_history: bool) -> tuple[Topics, OnlineFeatureStore, str]:
    run = uuid.uuid4().hex[:10]
    topics = Topics(
        transactions=TopicConfig(name=f"it-{run}-transactions", partitions=3),
        labels=TopicConfig(name=f"it-{run}-labels", partitions=3),
    )
    ensure_topics(BOOTSTRAP, [topics.transactions, topics.labels])
    store = OnlineFeatureStore(
        redis.Redis.from_url(REDIS_URL, decode_responses=True),
        OnlineStoreConfig(prefix=f"it-{run}", trim_history=trim_history),
    )
    return topics, store, f"it-{run}"


def _online_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={"txn_id": pl.String, **feature_schema(with_labels=True)})


def test_features_built_from_the_stream_equal_batch_features() -> None:
    """Replay through Redpanda, consume everything, then read each transaction as of its time."""
    events = generate(SyntheticConfig(days=30, n_cards=80, n_merchants=15))
    labels = label_events(events, LABELS)
    topics, store, group = _isolated(trim_history=False)

    stats = replay(timeline(events, labels), KafkaSink(BOOTSTRAP, topics))
    expected = stats.transactions + stats.labels
    processed = run_feature_builder(
        BOOTSTRAP, group, topics, store, max_messages=expected, idle_timeout_s=60
    )
    assert processed == expected

    transactions = [
        item.payload for item in timeline(events) if isinstance(item.payload, TransactionEvent)
    ]
    online = _online_frame([{"txn_id": e.txn_id, **store.read_features(e)} for e in transactions])
    assert online.equals(compute_features(events, labels))


def test_interleaved_scoring_through_the_stream_equals_batch_features() -> None:
    """Production order with trimming: read, then publish; the builder catches up each time."""
    events = generate(SyntheticConfig(days=12, n_cards=25, n_merchants=8))
    labels = label_events(events, LABELS)
    topics, store, group = _isolated(trim_history=True)

    progress = threading.Condition()
    processed = [0]

    def on_processed() -> None:
        with progress:
            processed[0] += 1
            progress.notify_all()

    stop = threading.Event()
    worker = threading.Thread(
        target=run_feature_builder,
        args=(BOOTSTRAP, group, topics, store),
        kwargs={"stop": stop, "on_processed": on_processed, "commit_every": 50},
        daemon=True,
    )
    worker.start()
    sink = KafkaSink(BOOTSTRAP, topics)
    published = 0
    rows: list[dict[str, object]] = []
    try:
        for item in timeline(events, labels):
            if isinstance(item.payload, TransactionEvent):
                sink.flush()
                with progress:
                    assert progress.wait_for(
                        lambda target=published: processed[0] >= target, timeout=60
                    )
                rows.append({"txn_id": item.payload.txn_id, **store.read_features(item.payload)})
                sink.send_transaction(item.payload)
            else:
                sink.send_label(item.payload)
            published += 1
        sink.flush()
    finally:
        stop.set()
        worker.join(timeout=30)
    assert _online_frame(rows).equals(compute_features(events, labels))

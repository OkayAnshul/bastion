"""Decision records through real Redpanda into the console's SQLite store (Phase 4).

Run with ``make up`` then ``make test-integration``; CI runs it on every push. The test uses its own
topic and consumer group, so runs never see each other's data.
"""

import os
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from bastion.serving.decision_log import KafkaDecisionSink
from bastion.serving.decision_store import DecisionStore
from bastion.streaming.config import TopicConfig
from bastion.streaming.decision_sink import run_decision_sink
from bastion.streaming.topics import ensure_topics
from tests.unit.test_decision_store import _record

pytestmark = pytest.mark.integration

BOOTSTRAP = os.environ.get("BASTION_KAFKA_BOOTSTRAP", "localhost:19092")


def test_published_decisions_are_stored_once_per_transaction(tmp_path: Path) -> None:
    run = uuid.uuid4().hex[:10]
    topic = TopicConfig(name=f"it-{run}-decisions", partitions=3)
    ensure_topics(BOOTSTRAP, [topic])
    start = datetime(2026, 6, 1, tzinfo=UTC)
    records = [
        _record(
            f"t{i}",
            probability=0.1 + i / 100,
            amount=10.0 + i,
            action="review" if i % 3 == 0 else "approve",
            event_ts=start + timedelta(minutes=i),
        )
        for i in range(60)
    ]
    producer = KafkaDecisionSink(BOOTSTRAP, topic.name)
    producer.write(records)
    producer.write(records[:10])  # a redelivered batch
    producer.flush()

    store = DecisionStore(tmp_path / "decisions.db")
    consumed = run_decision_sink(
        BOOTSTRAP,
        f"it-{run}",
        topic.name,
        store,
        batch_size=25,
        max_messages=70,
        idle_timeout_s=60,
    )
    assert consumed == 70
    assert sum(store.action_counts().values()) == 60  # upserts absorbed the redelivery
    assert len(store.review_queue()) == 20

import asyncio
import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from bastion.schemas.decisions import DecisionRecord
from bastion.serving.decision_log import DecisionLogger
from bastion.serving.decision_store import DecisionStore, SqliteDecisionSink

T0 = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def _record(
    txn_id: str,
    *,
    probability: float,
    amount: float,
    action: str | None,
    card_id: str = "c1",
    event_ts: datetime = T0,
) -> DecisionRecord:
    fields: dict[str, Any] = {
        "txn_id": txn_id,
        "event_ts": event_ts,
        "scored_at": event_ts + timedelta(milliseconds=5),
        "card_id": card_id,
        "merchant_id": "m1",
        "device_id": None,
        "amount": amount,
        "currency": "USD",
        "model_version": "test/1",
        "fraud_probability": probability,
        "raw_score": probability,
        "features": {"card_txn_count_1h": 2},
        "timings": {"redis_ms": 1, "features_ms": 1, "model_ms": 1, "total_ms": 3},
    }
    if action is not None:
        fields["decision"] = {
            "action": action,
            "override": None,
            "review_capped": False,
            "review_benefit": 1.0,
            "review_threshold": 0.0,
            "expected_cost": {"approve": probability * amount, "review": 5.0, "block": 1.0},
            "reason_codes": [],
        }
    return DecisionRecord.model_validate(fields)


@pytest.fixture
def store(tmp_path: Path) -> DecisionStore:
    return DecisionStore(tmp_path / "decisions.db")


def test_writes_are_idempotent_upserts(store: DecisionStore) -> None:
    record = _record("t1", probability=0.2, amount=100.0, action="review")
    store.write([record])
    store.write([record])  # at-least-once delivery from the decisions topic
    assert store.action_counts() == {"review": 1}
    assert store.get("t1") == record


def test_review_queue_is_unresolved_reviews_by_expected_fraud_loss(store: DecisionStore) -> None:
    store.write(
        [
            _record("small", probability=0.30, amount=50.0, action="review"),  # 15 USD
            _record("large", probability=0.10, amount=1_000.0, action="review"),  # 100 USD
            _record("resolved", probability=0.90, amount=5_000.0, action="review"),
            _record("blocked", probability=0.95, amount=9_000.0, action="block"),
            _record("phase3", probability=0.50, amount=500.0, action=None),  # no decision yet
        ]
    )
    store.record_verdict("resolved", "fraud", analyst="a1", note="card testing")
    assert [item.txn_id for item in store.review_queue()] == ["large", "small"]
    assert store.action_counts() == {"review": 3, "block": 1, "none": 1}
    verdict = store.verdict("resolved")
    assert verdict is not None
    assert (verdict.verdict, verdict.analyst, verdict.note) == ("fraud", "a1", "card testing")


def test_a_verdict_needs_a_known_transaction(store: DecisionStore) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        store.record_verdict("unknown", "legitimate", analyst="a1")


def test_card_history_is_strictly_earlier_and_newest_first(store: DecisionStore) -> None:
    store.write(
        [
            _record("old", probability=0.1, amount=10.0, action="approve", event_ts=T0),
            _record(
                "recent",
                probability=0.1,
                amount=10.0,
                action="approve",
                event_ts=T0 + timedelta(hours=1),
            ),
            _record(
                "same_time",
                probability=0.1,
                amount=10.0,
                action="approve",
                event_ts=T0 + timedelta(hours=2),
            ),
            _record("other_card", probability=0.1, amount=10.0, action="approve", card_id="c2"),
        ]
    )
    history = store.card_history("c1", before=T0 + timedelta(hours=2))
    assert [record.txn_id for record in history] == ["recent", "old"]


def test_reviews_are_counted_by_event_day(store: DecisionStore) -> None:
    store.write(
        [
            _record("d1", probability=0.2, amount=100.0, action="review", event_ts=T0),
            _record(
                "d2",
                probability=0.2,
                amount=100.0,
                action="review",
                event_ts=T0 + timedelta(days=1),
            ),
        ]
    )
    assert store.reviews_on(date(2026, 6, 1)) == 1
    assert store.reviews_on(date(2026, 6, 3)) == 0


def test_the_decision_logger_can_write_into_the_store(tmp_path: Path) -> None:
    sink = SqliteDecisionSink(tmp_path / "decisions.db")

    async def log_two() -> None:
        logger = DecisionLogger(sink, flush_interval_s=0.01)
        drain = asyncio.create_task(logger.run())
        logger.log(_record("t1", probability=0.1, amount=10.0, action="approve"))
        logger.log(_record("t2", probability=0.9, amount=10.0, action="block"))
        logger.close()
        await drain

    asyncio.run(log_two())
    assert sink.store.action_counts() == {"approve": 1, "block": 1}

"""Streaming feature builder: keeps the online store current from the event stream (ADR-004).

Delivery is at-least-once. Offsets are committed only after the Redis writes succeed, so a crash
reprocesses a few messages. Every write is idempotent (sorted-set members keyed by transaction id,
earliest-time maps written with ``ZADD LT``), so reprocessing leaves the store unchanged.

Ordering: Redpanda keeps each card's events in order within its partition. Across partitions and
across the two topics there is no global order, so a label can be applied slightly after a later
transaction. Reads are as-of the transaction's event time, so late *application* never exposes
future information. It can only mean a very recent label is not yet visible, as in any real system.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from confluent_kafka import Consumer, KafkaException

from bastion.features.online import OnlineFeatureStore
from bastion.schemas.events import LabelEvent, TransactionEvent
from bastion.streaming.config import Topics


def decode_message(
    topic: str | None, value: bytes | None, topics: Topics
) -> TransactionEvent | LabelEvent:
    if value is None:
        raise ValueError(f"empty message on topic {topic!r}")
    if topic == topics.transactions.name:
        return TransactionEvent.model_validate_json(value)
    if topic == topics.labels.name:
        return LabelEvent.model_validate_json(value)
    raise ValueError(f"unexpected topic {topic!r}")


def apply_item(store: OnlineFeatureStore, item: TransactionEvent | LabelEvent) -> None:
    if isinstance(item, TransactionEvent):
        store.write_transaction(item)
    else:
        store.write_label(item)


def run_feature_builder(
    bootstrap: str,
    group_id: str,
    topics: Topics,
    store: OnlineFeatureStore,
    *,
    max_messages: int | None = None,
    idle_timeout_s: float | None = None,
    commit_every: int = 500,
    stop: threading.Event | None = None,
    on_processed: Callable[[], None] | None = None,
) -> int:
    """Consume both topics until stopped, idle past ``idle_timeout_s``, or at ``max_messages``."""
    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([topics.transactions.name, topics.labels.name])
    processed = uncommitted = 0
    last_activity = time.monotonic()
    try:
        while stop is None or not stop.is_set():
            message = consumer.poll(0.5)
            if message is None:
                idle = time.monotonic() - last_activity
                if idle_timeout_s is not None and idle > idle_timeout_s:
                    break
                continue
            error = message.error()
            if error is not None:
                raise KafkaException(error)
            apply_item(store, decode_message(message.topic(), message.value(), topics))
            processed += 1
            uncommitted += 1
            last_activity = time.monotonic()
            if on_processed is not None:
                on_processed()
            if uncommitted >= commit_every:
                consumer.commit(asynchronous=False)
                uncommitted = 0
            if max_messages is not None and processed >= max_messages:
                break
        if uncommitted:
            consumer.commit(asynchronous=False)
    finally:
        consumer.close()
    return processed

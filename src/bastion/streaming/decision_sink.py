"""Decision sink: the decisions topic into the analyst console's SQLite store (Phase 4).

In the compose demo the scoring service publishes every decision record to the decisions topic, and
this consumer writes them to ``bastion.serving.decision_store`` in batches. Delivery is
at-least-once: offsets are committed only after a batch is written, and the store's upserts make a
redelivered record harmless.
"""

from __future__ import annotations

import threading
import time

from confluent_kafka import Consumer, KafkaException

from bastion.schemas.decisions import DecisionRecord
from bastion.serving.decision_store import DecisionStore


def decode_decision(value: bytes | None) -> DecisionRecord:
    if value is None:
        raise ValueError("empty message on the decisions topic")
    return DecisionRecord.model_validate_json(value)


def run_decision_sink(
    bootstrap: str,
    group_id: str,
    topic: str,
    store: DecisionStore,
    *,
    batch_size: int = 200,
    max_messages: int | None = None,
    idle_timeout_s: float | None = None,
    stop: threading.Event | None = None,
) -> int:
    """Consume until stopped, idle past ``idle_timeout_s``, or at ``max_messages``; return count."""
    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([topic])
    batch: list[DecisionRecord] = []
    written = 0
    last_activity = time.monotonic()

    def write_batch() -> None:
        nonlocal written
        if batch:
            store.write(batch)
            consumer.commit(asynchronous=False)  # only after the records are safely stored
            written += len(batch)
            batch.clear()

    try:
        while stop is None or not stop.is_set():
            message = consumer.poll(0.5)
            if message is None:
                write_batch()  # a quiet stream should not leave decisions invisible to analysts
                if idle_timeout_s is not None and time.monotonic() - last_activity > idle_timeout_s:
                    break
                continue
            error = message.error()
            if error is not None:
                raise KafkaException(error)
            batch.append(decode_decision(message.value()))
            last_activity = time.monotonic()
            if len(batch) >= batch_size:
                write_batch()
            if max_messages is not None and written + len(batch) >= max_messages:
                break
        write_batch()
    finally:
        consumer.close()
    return written

"""Event-time replay: the transaction simulator (Phase 2).

Replays a canonical event table, together with its simulated label arrivals, in event-time order. A
transaction is emitted at its ``event_ts`` and a label at its ``label_ts``. Pacing is optional:
``speedup`` event-seconds pass per wall-clock second (3600 means one hour of payments per second),
and ``None`` replays as fast as possible.

Sinks decide where items go. ``KafkaSink`` publishes to Redpanda, keyed by card so each card's
events stay ordered on one partition. ``StoreSink`` writes straight into an online store, for tests
and local runs without a broker.
"""

from __future__ import annotations

import heapq
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

import polars as pl
from confluent_kafka import KafkaException, Producer

from bastion.features.online import OnlineFeatureStore, epoch_ms
from bastion.schemas.events import LabelEvent, TransactionEvent
from bastion.schemas.tables import row_to_event
from bastion.streaming.config import Topics

_LABEL, _TRANSACTION = 0, 1  # at equal times labels go first; reads are strictly-before either way


@dataclass(frozen=True, order=True)
class TimelineItem:
    time_ms: int
    kind: int
    sequence: int
    payload: TransactionEvent | LabelEvent = field(compare=False)


def timeline(events: pl.DataFrame, labels: pl.DataFrame | None = None) -> Iterator[TimelineItem]:
    """Transactions and label arrivals merged lazily in time order.

    ``events`` must be sorted by ``event_ts`` (the offline contract) and ``labels`` by ``label_ts``
    (as ``bastion.data.labels.label_events`` returns them). Nothing is materialised up front.
    """

    def transactions() -> Iterator[TimelineItem]:
        for sequence, row in enumerate(events.iter_rows(named=True)):
            event = row_to_event(row)
            yield TimelineItem(epoch_ms(event.event_ts), _TRANSACTION, sequence, event)

    def arrivals() -> Iterator[TimelineItem]:
        if labels is None:
            return
        for sequence, row in enumerate(labels.iter_rows(named=True)):
            label = LabelEvent.model_validate(row)
            yield TimelineItem(epoch_ms(label.label_ts), _LABEL, sequence, label)

    yield from heapq.merge(transactions(), arrivals())


class EventSink(Protocol):
    def send_transaction(self, event: TransactionEvent) -> None: ...
    def send_label(self, label: LabelEvent) -> None: ...
    def flush(self) -> None: ...


class StoreSink:
    """Writes directly to an online store, with no broker in between."""

    def __init__(self, store: OnlineFeatureStore) -> None:
        self._store = store

    def send_transaction(self, event: TransactionEvent) -> None:
        self._store.write_transaction(event)

    def send_label(self, label: LabelEvent) -> None:
        self._store.write_label(label)

    def flush(self) -> None:
        return None


class KafkaSink:
    """Publishes JSON events to Redpanda, keyed by card id."""

    def __init__(self, bootstrap: str, topics: Topics) -> None:
        self._topics = topics
        self._producer = Producer(
            {
                "bootstrap.servers": bootstrap,
                "enable.idempotence": True,  # retries never duplicate or reorder a partition
                "linger.ms": 5,
                "compression.type": "zstd",
            }
        )
        self._error: Any = None

    def _on_delivery(self, error: Any, _message: Any) -> None:
        if error is not None and self._error is None:
            self._error = error

    def _produce(self, topic: str, key: str, value: str) -> None:
        self._producer.produce(
            topic, key=key.encode(), value=value.encode(), on_delivery=self._on_delivery
        )
        self._producer.poll(0)  # serve delivery callbacks without blocking

    def send_transaction(self, event: TransactionEvent) -> None:
        self._produce(self._topics.transactions.name, event.card_id, event.model_dump_json())

    def send_label(self, label: LabelEvent) -> None:
        self._produce(self._topics.labels.name, label.card_id, label.model_dump_json())

    def flush(self) -> None:
        remaining = self._producer.flush(30)
        if self._error is not None:
            raise KafkaException(self._error)
        if remaining:
            raise RuntimeError(f"{remaining} messages were not delivered within 30 seconds")


@dataclass(frozen=True)
class ReplayStats:
    transactions: int
    labels: int
    event_span_ms: int
    wall_seconds: float


def replay(
    items: Iterable[TimelineItem],
    sink: EventSink,
    *,
    speedup: float | None = None,
    limit: int | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> ReplayStats:
    """Send items to ``sink`` in order, pacing by event time when ``speedup`` is set.

    ``limit`` caps the number of transactions; replay stops before the first transaction past it.
    """
    if speedup is not None and speedup <= 0:
        raise ValueError("speedup must be positive, or None for as fast as possible")
    transactions = labels = 0
    first_ms: int | None = None
    last_ms = 0
    wall_start = clock()
    for item in items:
        is_transaction = isinstance(item.payload, TransactionEvent)
        if is_transaction and limit is not None and transactions >= limit:
            break
        if first_ms is None:
            first_ms = item.time_ms
        if speedup is not None:
            due = wall_start + (item.time_ms - first_ms) / 1000 / speedup
            delay = due - clock()
            if delay > 0:
                sleep(delay)
        if isinstance(item.payload, TransactionEvent):
            sink.send_transaction(item.payload)
            transactions += 1
        else:
            sink.send_label(item.payload)
            labels += 1
        last_ms = item.time_ms
    sink.flush()
    span = last_ms - first_ms if first_ms is not None else 0
    return ReplayStats(transactions, labels, span, clock() - wall_start)

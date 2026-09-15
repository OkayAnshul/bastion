"""Fire-and-forget decision logging: 0 ms of the latency budget (ARCHITECTURE.md §3.3).

The request handler only drops a record on an in-memory queue, which never blocks. A background task
drains the queue in batches and writes them to a sink off the event loop. If the queue is full, the
record is dropped and counted: a logging backlog must never slow a payment decision.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from confluent_kafka import Producer

from bastion.schemas.decisions import DecisionRecord


class DecisionSink(Protocol):
    def write(self, records: Sequence[DecisionRecord]) -> None: ...
    def flush(self) -> None: ...


class JsonlDecisionSink:
    """Appends one JSON record per line. Simple, local, and what the tests and benchmarks use."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path

    def write(self, records: Sequence[DecisionRecord]) -> None:
        with self._path.open("a") as fh:
            fh.writelines(record.model_dump_json() + "\n" for record in records)

    def flush(self) -> None:
        return None


class KafkaDecisionSink:
    """Publishes records to the decisions topic, keyed by card id."""

    def __init__(self, bootstrap: str, topic: str) -> None:
        self._topic = topic
        self._producer = Producer({"bootstrap.servers": bootstrap, "linger.ms": 20})

    def write(self, records: Sequence[DecisionRecord]) -> None:
        for record in records:
            self._producer.produce(
                self._topic, key=record.card_id.encode(), value=record.model_dump_json().encode()
            )
        self._producer.poll(0)

    def flush(self) -> None:
        self._producer.flush(10)


class NullDecisionSink:
    def write(self, records: Sequence[DecisionRecord]) -> None:
        return None

    def flush(self) -> None:
        return None


class DecisionLogger:
    def __init__(
        self,
        sink: DecisionSink,
        *,
        max_queue: int = 10_000,
        batch_size: int = 500,
        flush_interval_s: float = 0.5,
    ) -> None:
        self._sink = sink
        self._queue: asyncio.Queue[DecisionRecord] = asyncio.Queue(maxsize=max_queue)
        self._batch_size = batch_size
        self._flush_interval_s = flush_interval_s
        self._closing = asyncio.Event()
        self.dropped = 0
        self.written = 0

    def log(self, record: DecisionRecord) -> None:
        """Queue a record without waiting. Counts and drops it if the queue is full."""
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            self.dropped += 1

    async def run(self) -> None:
        """Drain the queue until ``close`` is called and nothing is left, then flush the sink."""
        while not (self._closing.is_set() and self._queue.empty()):
            batch = await self._next_batch()
            if batch:
                await asyncio.to_thread(self._sink.write, batch)
                self.written += len(batch)
        await asyncio.to_thread(self._sink.flush)

    async def _next_batch(self) -> list[DecisionRecord]:
        try:
            first = await asyncio.wait_for(self._queue.get(), timeout=self._flush_interval_s)
        except TimeoutError:
            return []
        batch = [first]
        while len(batch) < self._batch_size:
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return batch

    def close(self) -> None:
        self._closing.set()

    def stats(self) -> dict[str, Any]:
        return {"queued": self._queue.qsize(), "written": self.written, "dropped": self.dropped}

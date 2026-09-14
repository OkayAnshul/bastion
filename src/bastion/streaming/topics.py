"""Kafka topics, created idempotently by application code (ARCHITECTURE.md §7).

A one-shot "create topics" container would complicate ``docker compose up --wait``; creating
missing topics at startup costs one admin call and is safe to repeat.
"""

from __future__ import annotations

from collections.abc import Sequence

from confluent_kafka import KafkaError, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic  # type: ignore[attr-defined]

from bastion.streaming.config import TopicConfig


def ensure_topics(
    bootstrap: str,
    topics: Sequence[TopicConfig],
    *,
    replication_factor: int = 1,
    timeout_s: float = 30.0,
) -> list[str]:
    """Create missing topics and return the names created; existing topics are left untouched."""
    admin = AdminClient({"bootstrap.servers": bootstrap})
    futures = admin.create_topics(
        [
            NewTopic(t.name, num_partitions=t.partitions, replication_factor=replication_factor)
            for t in topics
        ],
        request_timeout=timeout_s,
    )
    created = []
    for name, future in futures.items():
        try:
            future.result()
            created.append(name)
        except KafkaException as exc:
            if exc.args[0].code() != KafkaError.TOPIC_ALREADY_EXISTS:
                raise
    return created

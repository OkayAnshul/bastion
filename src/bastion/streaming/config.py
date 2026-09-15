"""Streaming configuration: topics, consumer group, online store (``configs/streaming.yaml``)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from bastion.config import load_config
from bastion.features.online import OnlineStoreConfig


class TopicConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    partitions: int = Field(ge=1)


class Topics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    transactions: TopicConfig
    labels: TopicConfig
    decisions: TopicConfig = TopicConfig(name="decisions.log", partitions=3)


class StreamingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    topics: Topics
    consumer_group: str = Field(min_length=1)
    online_store: OnlineStoreConfig


def load_streaming_config(config_dir: Path | None = None) -> StreamingConfig:
    return load_config("streaming", StreamingConfig, config_dir)

"""Scoring response and decision-log contracts (Phase 3).

Every scored transaction produces one ``DecisionRecord``: what was known, what the model said, which
model said it, and how long each stage took. Monitoring (Phase 6) and the analyst console (Phase 4)
read these records; nothing downstream has to re-derive a decision.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from bastion.schemas.events import AttributeValue, EntityId, UtcDatetime


class StageTimings(BaseModel):
    """Server-side milliseconds per stage, measured inside the request handler."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    redis_ms: float = Field(ge=0)
    features_ms: float = Field(ge=0)
    model_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)


class ScoreResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    txn_id: EntityId
    model_version: str
    fraud_probability: float = Field(ge=0, le=1)  # calibrated (ADR-006)
    raw_score: float
    timings: StageTimings


class DecisionRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    txn_id: EntityId
    event_ts: UtcDatetime
    scored_at: UtcDatetime
    card_id: EntityId
    merchant_id: EntityId
    device_id: EntityId | None
    amount: float = Field(gt=0)
    currency: str
    model_version: str
    fraud_probability: float = Field(ge=0, le=1)
    raw_score: float
    features: dict[str, AttributeValue]  # exactly what the model saw, for skew audits
    timings: StageTimings

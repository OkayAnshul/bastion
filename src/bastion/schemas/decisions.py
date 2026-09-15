"""Scoring response and decision-log contracts (Phase 3).

Every scored transaction produces one ``DecisionRecord``: what was known, what the model said, which
model said it, and how long each stage took. Monitoring (Phase 6) and the analyst console (Phase 4)
read these records; nothing downstream has to re-derive a decision.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from bastion.evaluation.cost import Action
from bastion.schemas.events import AttributeValue, EntityId, UtcDatetime


class StageTimings(BaseModel):
    """Server-side milliseconds per stage, measured inside the request handler."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    redis_ms: float = Field(ge=0)
    features_ms: float = Field(ge=0)
    model_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)
    # Phase 4 stages. Zero in records written before the policy engine existed.
    policy_ms: float = Field(default=0.0, ge=0)  # overrides, expected costs, review capacity
    explain_ms: float = Field(default=0.0, ge=0)  # reason codes, for reviewed and blocked only


class ReasonCodeOut(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    feature: str
    value: float | str | None
    contribution: float  # log-odds this input added to the model's margin (TreeSHAP)
    description: str


class PolicyDecision(BaseModel):
    """What the policy engine did with the calibrated probability (ADR-005)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: Action
    override: str | None  # the hard rule that decided, if one did
    review_capped: bool  # a review was wanted, but the day's budget was already used
    review_benefit: float  # expected saving of a review over the better of approve and block
    review_threshold: float
    expected_cost: dict[str, float]  # per action: approve, review, block
    reason_codes: list[ReasonCodeOut]  # empty for approvals


class ScoreResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    txn_id: EntityId
    model_version: str
    fraud_probability: float = Field(ge=0, le=1)  # calibrated (ADR-006)
    raw_score: float
    decision: PolicyDecision
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
    decision: PolicyDecision | None = None  # None in records written before Phase 4
    timings: StageTimings

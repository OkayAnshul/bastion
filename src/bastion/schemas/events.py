"""Canonical event contracts (ARCHITECTURE.md §3.1).

Every component uses these types. The simulator produces them, Redpanda carries them as JSON, and
the feature builder and scoring service consume them. IEEE-CIS rows are mapped onto them by
``bastion.data.adapter`` (ADR-011).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field, model_validator


def _to_utc(ts: datetime) -> datetime:
    return ts.astimezone(UTC)


# Naive datetimes are rejected: an event time without a zone is ambiguous, and ambiguity in
# event time becomes silent leakage in point-in-time joins.
UtcDatetime = Annotated[AwareDatetime, AfterValidator(_to_utc)]
EntityId = Annotated[str, Field(min_length=1, max_length=128)]
AttributeValue = bool | int | float | str | None
Channel = Literal["ecom", "pos", "atm"]


class TransactionEvent(BaseModel):
    """A payment authorisation as seen by Bastion.

    ``attributes`` carries vendor-provided fields that exist at authorisation time (for IEEE-CIS:
    the anonymised C/D/M/V columns). Missing values are ``None``; NaN is not valid JSON and is
    rejected so it cannot leak into the stream.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    txn_id: EntityId
    event_ts: UtcDatetime
    card_id: EntityId
    device_id: EntityId | None = None
    merchant_id: EntityId
    merchant_category: str = Field(min_length=1, max_length=16)
    ip: str | None = None
    amount: float = Field(gt=0, allow_inf_nan=False)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    channel: Channel
    is_fraud: bool | None = None
    attributes: dict[str, AttributeValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _attributes_are_finite(self) -> Self:
        bad = [
            k for k, v in self.attributes.items() if isinstance(v, float) and not math.isfinite(v)
        ]
        if bad:
            raise ValueError(f"non-finite attribute values (use null for missing): {sorted(bad)}")
        return self


class ScoreRequest(TransactionEvent):
    """Input to ``POST /v1/score``.

    The label cannot exist at decision time: a chargeback arrives days later. A request that
    carries one is a bug in the caller (usually the replay leaking labels), so it is rejected.
    """

    @model_validator(mode="after")
    def _no_label_at_scoring_time(self) -> Self:
        if self.is_fraud is not None:
            raise ValueError("is_fraud must be null at scoring time; labels arrive later")
        return self


class LabelEvent(BaseModel):
    """The moment a transaction's fraud label becomes known to Bastion.

    ``event_ts`` is when the transaction happened; ``label_ts`` is when the label became
    available. Only labels with ``label_ts`` before a decision's time may influence that decision.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    txn_id: EntityId
    card_id: EntityId
    merchant_id: EntityId
    event_ts: UtcDatetime
    label_ts: UtcDatetime
    is_fraud: bool

    @model_validator(mode="after")
    def _label_after_transaction(self) -> Self:
        if self.label_ts < self.event_ts:
            raise ValueError(
                "label_ts precedes event_ts: a label cannot exist before its transaction"
            )
        return self

"""Online executor (ADR-004): the shared feature definitions, evaluated from Redis.

The streaming feature builder writes each event into per-entity Redis structures. The scoring path
reads one transaction's context in a single pipelined round trip, then evaluates the *same*
functions from ``bastion.features.definitions`` that the batch executor uses.

Every structure can be queried as of any time: histories are sorted sets scored by event time, and
reads use exclusive upper bounds. A read for a transaction at time ``q`` therefore sees only events
and labels strictly before ``q``, whether or not later events have been written already.

Keys, under ``{prefix}:``

- ``card:{card}:events``: sorted set, one member per transaction, scored by event time (ms)
- ``device:{device}:events``: sorted set, one member per transaction, scored by event time
- ``first_seen:card`` and ``first_seen:device``: sorted sets, entity to earliest event time
- ``card:{card}:devices`` and ``card:{card}:merchants``: first time this card used each one
- ``card:{card}:fraud_labels``: fraud labels on the card, scored by label arrival time
- ``merchant:{merchant}:labels`` and ``merchant:{merchant}:fraud_labels``: labels by arrival time
- ``labels:all`` and ``labels:fraud``: the same, across all merchants

Earliest times are written with ``ZADD LT``, so out-of-order writes from different partitions keep
the minimum. Labels are set members keyed by transaction id, so replaying a label counts it once.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import numpy.typing as npt
import redis
import redis.asyncio
from pydantic import BaseModel, ConfigDict, Field

from bastion.features.batch import (
    DEFAULT_LABEL_STRENGTH,
    DEVICE_COUNT_WINDOWS,
    DEVICE_DISTINCT_WINDOWS,
    DISTINCT_WINDOWS,
)
from bastion.features.definitions import (
    DEVIATION_WINDOW_MS,
    MS_PER_DAY,
    NEVER_SEEN_MS,
    WINDOWS_MS,
    age_days,
    distinct_in_window_many,
    from_milli_units,
    mean_and_std,
    seen_before,
    smoothed_rate,
    to_cents,
    to_milli_units,
    window_aggregates,
    window_aggregates_many,
)
from bastion.schemas.events import LabelEvent, TransactionEvent

# The longest window that reads each history decides how much of it must be kept.
CARD_HISTORY_MS = DEVIATION_WINDOW_MS
DEVICE_HISTORY_MS = max(WINDOWS_MS[w] for w in (*DEVICE_COUNT_WINDOWS, *DEVICE_DISTINCT_WINDOWS))
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_SECONDS_PER_DAY = 86_400

# A Redis command as (method name, positional args, keyword args), replayable on any client.
Command = tuple[str, tuple[Any, ...], dict[str, Any]]


class OnlineStoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prefix: str = Field(default="bastion", min_length=1)
    trim_history: bool = True
    history_ttl_slack_days: float = Field(default=2, gt=0)
    lifetime_ttl_days: float = Field(default=400, gt=0)


def epoch_ms(ts: datetime) -> int:
    """Exact integer epoch milliseconds (``ts.timestamp() * 1000`` can be off by one)."""
    return (ts - _EPOCH) // timedelta(milliseconds=1)


def _key(prefix: str, *parts: str) -> str:
    return ":".join((prefix, *parts))


@dataclass(frozen=True)
class OnlineSnapshot:
    """Everything one feature read needs, as fetched from Redis."""

    card_events: list[tuple[str, float]]
    card_first_seen: float | None
    card_merchant_first_seen: float | None
    device_events: list[tuple[str, float]] | None  # None when the transaction has no device
    device_first_seen: float | None
    card_device_first_seen: float | None
    card_fraud_labels: int
    merchant_labels: int
    merchant_fraud_labels: int
    all_labels: int
    all_fraud_labels: int


def snapshot_commands(prefix: str, event: TransactionEvent) -> list[Command]:
    """The commands one feature read needs, in the order ``snapshot_from_results`` expects.

    Shared by the synchronous store and the asyncio scoring path, so both issue the same reads.
    """
    q = epoch_ms(event.event_ts)
    before_q = f"({q}"
    before_day = f"({q - q % MS_PER_DAY}"  # merchant rates: labels known by the start of the day
    card, merchant = event.card_id, event.merchant_id
    commands: list[Command] = [
        (
            "zrangebyscore",
            (_key(prefix, "card", card, "events"), q - CARD_HISTORY_MS, before_q),
            {"withscores": True},
        ),
        ("zscore", (_key(prefix, "first_seen", "card"), card), {}),
        ("zscore", (_key(prefix, "card", card, "merchants"), merchant), {}),
        ("zcount", (_key(prefix, "card", card, "fraud_labels"), "-inf", before_q), {}),
        ("zcount", (_key(prefix, "merchant", merchant, "labels"), "-inf", before_day), {}),
        ("zcount", (_key(prefix, "merchant", merchant, "fraud_labels"), "-inf", before_day), {}),
        ("zcount", (_key(prefix, "labels", "all"), "-inf", before_day), {}),
        ("zcount", (_key(prefix, "labels", "fraud"), "-inf", before_day), {}),
    ]
    if event.device_id is not None:
        device = event.device_id
        commands += [
            (
                "zrangebyscore",
                (_key(prefix, "device", device, "events"), q - DEVICE_HISTORY_MS, before_q),
                {"withscores": True},
            ),
            ("zscore", (_key(prefix, "first_seen", "device"), device), {}),
            ("zscore", (_key(prefix, "card", card, "devices"), device), {}),
        ]
    return commands


def snapshot_from_results(event: TransactionEvent, results: Sequence[Any]) -> OnlineSnapshot:
    has_device = event.device_id is not None
    return OnlineSnapshot(
        card_events=list(results[0]),
        card_first_seen=results[1],
        card_merchant_first_seen=results[2],
        card_fraud_labels=int(results[3]),
        merchant_labels=int(results[4]),
        merchant_fraud_labels=int(results[5]),
        all_labels=int(results[6]),
        all_fraud_labels=int(results[7]),
        device_events=list(results[8]) if has_device else None,
        device_first_seen=results[9] if has_device else None,
        card_device_first_seen=results[10] if has_device else None,
    )


async def read_snapshot_async(
    client: redis.asyncio.Redis, prefix: str, event: TransactionEvent
) -> OnlineSnapshot:
    """The same single-round-trip read on an asyncio client, for the scoring service."""
    pipe = client.pipeline(transaction=False)
    for name, args, kwargs in snapshot_commands(prefix, event):
        getattr(pipe, name)(*args, **kwargs)
    return snapshot_from_results(event, await pipe.execute())


class OnlineFeatureStore:
    def __init__(self, client: redis.Redis, config: OnlineStoreConfig) -> None:
        self.client = client
        self.config = config

    def _key(self, *parts: str) -> str:
        return _key(self.config.prefix, *parts)

    # -------------------------------------------------------------------------- write path

    def write_transaction(self, event: TransactionEvent) -> None:
        """Record one transaction. Idempotent: rewriting the same event changes nothing."""
        ts = epoch_ms(event.event_ts)
        slack = self.config.history_ttl_slack_days * _SECONDS_PER_DAY
        history_ttl = int(CARD_HISTORY_MS / 1000 + slack)
        device_ttl = int(DEVICE_HISTORY_MS / 1000 + slack)
        lifetime_ttl = int(self.config.lifetime_ttl_days * _SECONDS_PER_DAY)

        pipe = self.client.pipeline(transaction=False)
        card_events = self._key("card", event.card_id, "events")
        member = json.dumps([event.txn_id, event.amount, event.merchant_id, event.device_id])
        pipe.zadd(card_events, {member: ts})
        if self.config.trim_history:
            pipe.zremrangebyscore(card_events, "-inf", f"({ts - CARD_HISTORY_MS}")
        pipe.expire(card_events, history_ttl)
        pipe.zadd(self._key("first_seen", "card"), {event.card_id: ts}, lt=True)
        merchants = self._key("card", event.card_id, "merchants")
        pipe.zadd(merchants, {event.merchant_id: ts}, lt=True)
        pipe.expire(merchants, lifetime_ttl)

        if event.device_id is not None:
            device_events = self._key("device", event.device_id, "events")
            pipe.zadd(device_events, {json.dumps([event.txn_id, event.card_id]): ts})
            if self.config.trim_history:
                pipe.zremrangebyscore(device_events, "-inf", f"({ts - DEVICE_HISTORY_MS}")
            pipe.expire(device_events, device_ttl)
            pipe.zadd(self._key("first_seen", "device"), {event.device_id: ts}, lt=True)
            devices = self._key("card", event.card_id, "devices")
            pipe.zadd(devices, {event.device_id: ts}, lt=True)
            pipe.expire(devices, lifetime_ttl)
        pipe.execute()

    def write_label(self, label: LabelEvent) -> None:
        """Record a label's arrival. Idempotent: labels are set members keyed by transaction id."""
        ts = epoch_ms(label.label_ts)
        lifetime_ttl = int(self.config.lifetime_ttl_days * _SECONDS_PER_DAY)
        pipe = self.client.pipeline(transaction=False)
        merchant_labels = self._key("merchant", label.merchant_id, "labels")
        pipe.zadd(merchant_labels, {label.txn_id: ts})
        pipe.expire(merchant_labels, lifetime_ttl)
        pipe.zadd(self._key("labels", "all"), {label.txn_id: ts})
        if label.is_fraud:
            card_fraud = self._key("card", label.card_id, "fraud_labels")
            pipe.zadd(card_fraud, {label.txn_id: ts})
            pipe.expire(card_fraud, lifetime_ttl)
            merchant_fraud = self._key("merchant", label.merchant_id, "fraud_labels")
            pipe.zadd(merchant_fraud, {label.txn_id: ts})
            pipe.expire(merchant_fraud, lifetime_ttl)
            pipe.zadd(self._key("labels", "fraud"), {label.txn_id: ts})
        pipe.execute()

    # -------------------------------------------------------------------------- read path

    def snapshot(self, event: TransactionEvent) -> OnlineSnapshot:
        """Fetch a transaction's context in one pipelined round trip, as of its event time."""
        pipe = self.client.pipeline(transaction=False)
        for name, args, kwargs in snapshot_commands(self.config.prefix, event):
            getattr(pipe, name)(*args, **kwargs)
        return snapshot_from_results(event, pipe.execute())

    def read_features(
        self, event: TransactionEvent, *, label_strength: float = DEFAULT_LABEL_STRENGTH
    ) -> dict[str, Any]:
        return features_from_snapshot(event, self.snapshot(event), label_strength=label_strength)


# ------------------------------------------------------------------------------ evaluation


def _decode(member: Any) -> str:
    return member.decode() if isinstance(member, bytes) else str(member)


def _first_seen(score: float | None) -> npt.NDArray[np.int64]:
    return np.array([NEVER_SEEN_MS if score is None else int(score)], dtype=np.int64)


def _scalar(value: float) -> float | None:
    return None if np.isnan(value) else float(value)


def _codes(values: list[str]) -> npt.NDArray[np.int64]:
    """Local integer codes; distinct counts do not depend on which code a value gets."""
    lookup: dict[str, int] = {}
    return np.array([lookup.setdefault(v, len(lookup)) for v in values], dtype=np.int64)


def features_from_snapshot(
    event: TransactionEvent,
    snapshot: OnlineSnapshot,
    *,
    label_strength: float = DEFAULT_LABEL_STRENGTH,
) -> dict[str, Any]:
    """Evaluate every feature for one transaction from its snapshot, with the batch definitions.

    History arrays use entity code 0 and the transaction is the single query; the functions are the
    ones ``bastion.features.batch`` calls on the whole table.
    """
    q = np.array([epoch_ms(event.event_ts)], dtype=np.int64)
    one = np.zeros(1, dtype=np.int64)
    out: dict[str, Any] = {}

    # ---- card history
    rows = [json.loads(_decode(member)) for member, _ in snapshot.card_events]
    ts = np.array([int(score) for _, score in snapshot.card_events], dtype=np.int64)
    entity = np.zeros(ts.size, dtype=np.int64)
    amounts = np.array([row[1] for row in rows], dtype=np.float64)
    amount_mu = to_milli_units(amounts)
    velocity = window_aggregates_many(entity, ts, amount_mu, one, q, list(WINDOWS_MS.values()))
    for name, agg in zip(WINDOWS_MS, velocity, strict=True):
        out[f"card_txn_count_{name}"] = int(agg.count[0])
        out[f"card_amount_sum_{name}"] = float(from_milli_units(agg.total)[0])

    merchant_codes = _codes([row[2] for row in rows])
    device_values = [row[3] for row in rows]
    with_device = np.array([d is not None for d in device_values], dtype=np.bool_)
    device_codes = _codes([d for d in device_values if d is not None])
    windows = [WINDOWS_MS[name] for name in DISTINCT_WINDOWS]
    merchants = distinct_in_window_many(entity, ts, merchant_codes, one, q, windows)
    devices = distinct_in_window_many(
        entity[with_device], ts[with_device], device_codes, one, q, windows
    )
    for name, merchant_count, device_count in zip(
        DISTINCT_WINDOWS, merchants, devices, strict=True
    ):
        out[f"card_distinct_merchants_{name}"] = int(merchant_count[0])
        out[f"card_distinct_devices_{name}"] = int(device_count[0])

    cents = to_cents(amounts)
    first = window_aggregates(entity, ts, cents, one, q, DEVIATION_WINDOW_MS)
    second = window_aggregates(entity, ts, cents * cents, one, q, DEVIATION_WINDOW_MS)
    mean, std = mean_and_std(first.count, first.total, second.total)
    amount = np.array([event.amount], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = amount / mean
        zscore = np.where(std > 0, (amount - mean) / std, np.nan)
    out["card_amount_mean_30d"] = _scalar(mean[0])
    out["card_amount_std_30d"] = _scalar(std[0])
    out["amount_to_card_mean_30d"] = _scalar(ratio[0])
    out["amount_zscore_30d"] = _scalar(zscore[0])

    # ---- device history
    if snapshot.device_events is None:
        for name in DEVICE_COUNT_WINDOWS:
            out[f"device_txn_count_{name}"] = None
        for name in DEVICE_DISTINCT_WINDOWS:
            out[f"device_distinct_cards_{name}"] = None
    else:
        device_ts = np.array([int(score) for _, score in snapshot.device_events], dtype=np.int64)
        device_entity = np.zeros(device_ts.size, dtype=np.int64)
        card_codes = _codes([json.loads(_decode(m))[1] for m, _ in snapshot.device_events])
        ones = np.ones(device_ts.size, dtype=np.int64)
        count_windows = [WINDOWS_MS[name] for name in DEVICE_COUNT_WINDOWS]
        counts = window_aggregates_many(device_entity, device_ts, ones, one, q, count_windows)
        for name, agg in zip(DEVICE_COUNT_WINDOWS, counts, strict=True):
            out[f"device_txn_count_{name}"] = int(agg.count[0])
        card_windows = [WINDOWS_MS[name] for name in DEVICE_DISTINCT_WINDOWS]
        cards = distinct_in_window_many(device_entity, device_ts, card_codes, one, q, card_windows)
        for name, card_count in zip(DEVICE_DISTINCT_WINDOWS, cards, strict=True):
            out[f"device_distinct_cards_{name}"] = int(card_count[0])

    # ---- familiarity
    has_device = snapshot.device_events is not None
    out["card_age_days"] = float(age_days(_first_seen(snapshot.card_first_seen), q)[0])
    out["device_age_days"] = (
        float(age_days(_first_seen(snapshot.device_first_seen), q)[0]) if has_device else None
    )
    out["card_device_seen_before"] = (
        bool(seen_before(_first_seen(snapshot.card_device_first_seen), q)[0])
        if has_device
        else None
    )
    out["card_merchant_seen_before"] = bool(
        seen_before(_first_seen(snapshot.card_merchant_first_seen), q)[0]
    )

    # ---- contextual
    out["hour_of_day"] = event.event_ts.hour
    out["day_of_week"] = event.event_ts.isoweekday()

    # ---- arrived labels
    out["card_prior_fraud_labels"] = snapshot.card_fraud_labels
    out["merchant_known_labels"] = snapshot.merchant_labels
    rate = smoothed_rate(
        np.array([snapshot.merchant_fraud_labels], dtype=np.int64),
        np.array([snapshot.merchant_labels], dtype=np.int64),
        np.array([snapshot.all_fraud_labels], dtype=np.int64),
        np.array([snapshot.all_labels], dtype=np.int64),
        label_strength,
    )
    out["merchant_known_fraud_rate"] = _scalar(rate[0])
    return out

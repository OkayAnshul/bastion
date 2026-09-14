"""Feature definitions shared by the batch (training) and online (serving) executors.

This module is the single source of truth for what a feature means (ADR-004). Each function answers
"what was true strictly before time q?" and works on plain arrays. The batch executor passes the
whole history at once; the online executor passes one entity's recent events fetched from Redis.
Both call exactly this code, so a definition cannot drift between training and serving.

Conventions (ADR-003):

* Time is int64 epoch milliseconds, UTC.
* A window ``w`` at query time ``q`` covers events with ``q - w <= ts < q``. The upper bound
  is exclusive: an event never sees itself, or other events with the same timestamp. IEEE-CIS has
  one-second resolution, so same-timestamp events are common.
* Money is aggregated as int64 milli-units (1/1000 of the currency unit). Integer sums are exact, so
  batch and online agree bit-for-bit instead of "within a float tolerance".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

Int64Array = npt.NDArray[np.int64]

WINDOWS_MS: Final[dict[str, int]] = {
    "1m": 60_000,
    "10m": 10 * 60_000,
    "1h": 60 * 60_000,
    "24h": 24 * 60 * 60_000,
    "7d": 7 * 24 * 60 * 60_000,
}

MILLI_UNITS_PER_UNIT: Final = 1000

# Entity and time are packed into one sortable int64: (entity << 42) | ts. 42 bits of epoch
# milliseconds covers dates until the year 2109; the remaining 21 bits allow 2 million entities.
_TS_BITS: Final = 42
MAX_TIMESTAMP_MS: Final = (1 << _TS_BITS) - 1
MAX_ENTITY_CODE: Final = (1 << (63 - _TS_BITS)) - 1

# First-seen value for a key that has never appeared: compares as "not before" any query time.
NEVER_SEEN_MS: Final = int(np.iinfo(np.int64).max)


def to_milli_units(amount: npt.ArrayLike) -> Int64Array:
    return np.rint(np.asarray(amount, dtype=np.float64) * MILLI_UNITS_PER_UNIT).astype(np.int64)


def from_milli_units(amount_mu: Int64Array) -> npt.NDArray[np.float64]:
    return amount_mu.astype(np.float64) / MILLI_UNITS_PER_UNIT


def _entity_time_key(entity: Int64Array, ts: Int64Array) -> Int64Array:
    if entity.size and (entity.min() < 0 or entity.max() > MAX_ENTITY_CODE):
        raise ValueError(f"entity codes must lie in [0, {MAX_ENTITY_CODE}]")
    if ts.size and (ts.min() < 0 or ts.max() > MAX_TIMESTAMP_MS):
        raise ValueError(f"timestamps must lie in [0, {MAX_TIMESTAMP_MS}] epoch ms")
    return (entity.astype(np.int64) << _TS_BITS) | ts.astype(np.int64)


@dataclass(frozen=True)
class WindowAggregate:
    count: Int64Array
    total: Int64Array  # sum of the history values inside the window


def window_aggregates(
    history_entity: Int64Array,
    history_ts: Int64Array,
    history_values: Int64Array,
    query_entity: Int64Array,
    query_ts: Int64Array,
    window_ms: int,
) -> WindowAggregate:
    """Count and sum of ``history_values`` for each query entity's events in ``[q - window_ms, q)``.

    ``history_*`` must be sorted by (entity, ts). In batch, history and queries are the same rows.
    Online, history is one entity's recent events and the query is the incoming transaction, with
    entity code 0 for both.
    """
    if window_ms <= 0:
        raise ValueError("window_ms must be positive")
    key = _entity_time_key(history_entity, history_ts)
    if np.any(key[1:] < key[:-1]):
        raise ValueError("history must be sorted by (entity, ts)")
    if history_values.size and float(np.abs(history_values.astype(np.float64)).sum()) >= 2.0**62:
        raise OverflowError("cumulative sums of history_values would overflow int64")

    upper_key = _entity_time_key(query_entity, query_ts)
    # Clamping at 0 keeps the lower bound inside the query's own entity block.
    lower_ts = np.maximum(query_ts.astype(np.int64) - window_ms, 0).astype(np.int64)
    lower_key = _entity_time_key(query_entity, lower_ts)

    upper = np.searchsorted(key, upper_key, side="left")  # first event at or after q: excluded
    lower = np.searchsorted(key, lower_key, side="left")  # first event at or after q - w: included
    cumulative = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(history_values, dtype=np.int64))
    )
    return WindowAggregate(
        count=(upper - lower).astype(np.int64),
        total=(cumulative[upper] - cumulative[lower]).astype(np.int64),
    )


def seen_before(first_seen_ms: Int64Array, query_ts: Int64Array) -> npt.NDArray[np.bool_]:
    """Whether a key (for example a card-device pair) appeared strictly before each query time.

    ``first_seen_ms`` is the key's earliest timestamp over *all* data, which looks like a leaky
    whole-dataset aggregate but is not: events at or after ``q`` can't pull the minimum below ``q``,
    so ``first_seen < q`` gives the same answer with or without future events. That property also
    lets the online store keep one first-seen timestamp per key and still answer as of any time.
    """
    return np.asarray(first_seen_ms < query_ts, dtype=np.bool_)


# ------------------------------------------------------------------------------ Phase 1 groups

MS_PER_DAY: Final = 24 * 60 * 60_000
DEVIATION_WINDOW_MS: Final = 30 * MS_PER_DAY
LIFETIME_WINDOW_MS: Final = MAX_TIMESTAMP_MS
CENTS_PER_UNIT: Final = 100


def to_cents(amount: npt.ArrayLike) -> Int64Array:
    """Integer cents, used for second moments: squared milli-units could overflow int64 sums."""
    return np.rint(np.asarray(amount, dtype=np.float64) * CENTS_PER_UNIT).astype(np.int64)


def distinct_in_window(
    history_entity: Int64Array,
    history_ts: Int64Array,
    history_value: Int64Array,
    query_entity: Int64Array,
    query_ts: Int64Array,
    window_ms: int,
) -> Int64Array:
    """Number of distinct ``history_value`` codes per query entity in ``[q - window_ms, q)``.

    Each occurrence of a value *covers* the query times at which it is that value's latest earlier
    occurrence and still inside the window: ``q`` in ``(ts, min(next_ts, ts + window_ms)]``. The
    distinct count at ``q`` is (covers started before ``q``) minus (covers ended before ``q``): two
    binary searches per query instead of a per-row set. History need not be sorted.
    """
    if window_ms <= 0:
        raise ValueError("window_ms must be positive")
    order = np.lexsort((history_ts, history_value, history_entity))
    entity = history_entity[order].astype(np.int64)
    value = history_value[order].astype(np.int64)
    ts = history_ts[order].astype(np.int64)
    if ts.size:
        # Repeats of a value at the same timestamp add nothing.
        fresh = np.ones(ts.size, dtype=np.bool_)
        fresh[1:] = (entity[1:] != entity[:-1]) | (value[1:] != value[:-1]) | (ts[1:] != ts[:-1])
        entity, value, ts = entity[fresh], value[fresh], ts[fresh]

    next_ts = np.full(ts.size, MAX_TIMESTAMP_MS, dtype=np.int64)
    if ts.size > 1:
        same_key = (entity[1:] == entity[:-1]) & (value[1:] == value[:-1])
        next_ts[:-1] = np.where(same_key, ts[1:], MAX_TIMESTAMP_MS)
    cover_end = np.minimum(next_ts, np.minimum(ts + window_ms, MAX_TIMESTAMP_MS))

    starts = np.sort(_entity_time_key(entity, ts))
    ends = np.sort(_entity_time_key(entity, cover_end))
    at_query = _entity_time_key(query_entity, query_ts)
    block_start = _entity_time_key(query_entity, np.zeros(query_ts.size, dtype=np.int64))
    started = np.searchsorted(starts, at_query, side="left") - np.searchsorted(
        starts, block_start, side="left"
    )
    ended = np.searchsorted(ends, at_query, side="left") - np.searchsorted(
        ends, block_start, side="left"
    )
    return (started - ended).astype(np.int64)


def mean_and_std(
    count: Int64Array, sum_cents: Int64Array, sum_squared_cents: Int64Array
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Mean and population standard deviation, in currency units, from windowed integer moments.

    NaN where undefined: a mean needs one earlier event, a standard deviation two. The integer
    inputs are identical in both executors, so only this final, deterministic float step remains.
    """
    n = count.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_cents = sum_cents / n
        variance = np.maximum(sum_squared_cents / n - mean_cents**2, 0.0)
    mean = np.where(count >= 1, mean_cents / CENTS_PER_UNIT, np.nan)
    std = np.where(count >= 2, np.sqrt(variance) / CENTS_PER_UNIT, np.nan)
    return mean.astype(np.float64), std.astype(np.float64)


def age_days(first_seen_ms: Int64Array, query_ts: Int64Array) -> npt.NDArray[np.float64]:
    """Days since a key was first seen strictly before ``q``; 0 on its first appearance.

    Point-in-time safe for the same reason as ``seen_before``.
    """
    with np.errstate(invalid="ignore"):
        age = (query_ts - first_seen_ms) / MS_PER_DAY
    return np.where(first_seen_ms < query_ts, age, 0.0).astype(np.float64)


def smoothed_rate(
    positives: Int64Array,
    total: Int64Array,
    prior_positives: Int64Array,
    prior_total: Int64Array,
    strength: float,
) -> npt.NDArray[np.float64]:
    """Shrink a rate toward a prior: ``(positives + strength * prior) / (total + strength)``.

    ``prior = prior_positives / prior_total``. An entity with few known labels stays near the prior
    instead of swinging to 0 or 1. NaN while no prior exists yet.
    """
    if strength <= 0:
        raise ValueError("strength must be positive")
    with np.errstate(divide="ignore", invalid="ignore"):
        prior = np.where(prior_total > 0, prior_positives / prior_total, np.nan)
    return ((positives + strength * prior) / (total + strength)).astype(np.float64)

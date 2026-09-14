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
    amount_mu: Int64Array


def window_aggregates(
    history_entity: Int64Array,
    history_ts: Int64Array,
    history_amount_mu: Int64Array,
    query_entity: Int64Array,
    query_ts: Int64Array,
    window_ms: int,
) -> WindowAggregate:
    """Count and total amount of each query entity's events in ``[q - window_ms, q)``.

    ``history_*`` must be sorted by (entity, ts). In batch, history and queries are the same rows.
    Online, history is one entity's recent events and the query is the incoming transaction, with
    entity code 0 for both.
    """
    if window_ms <= 0:
        raise ValueError("window_ms must be positive")
    key = _entity_time_key(history_entity, history_ts)
    if np.any(key[1:] < key[:-1]):
        raise ValueError("history must be sorted by (entity, ts)")

    upper_key = _entity_time_key(query_entity, query_ts)
    # Clamping at 0 keeps the lower bound inside the query's own entity block.
    lower_ts = np.maximum(query_ts.astype(np.int64) - window_ms, 0).astype(np.int64)
    lower_key = _entity_time_key(query_entity, lower_ts)

    upper = np.searchsorted(key, upper_key, side="left")  # first event at or after q: excluded
    lower = np.searchsorted(key, lower_key, side="left")  # first event at or after q - w: included
    cumulative = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(history_amount_mu, dtype=np.int64))
    )
    return WindowAggregate(
        count=(upper - lower).astype(np.int64),
        amount_mu=(cumulative[upper] - cumulative[lower]).astype(np.int64),
    )


def seen_before(first_seen_ms: Int64Array, query_ts: Int64Array) -> npt.NDArray[np.bool_]:
    """Whether a key (for example a card-device pair) appeared strictly before each query time.

    ``first_seen_ms`` is the key's earliest timestamp over *all* data, which looks like a leaky
    whole-dataset aggregate but is not: events at or after ``q`` can't pull the minimum below ``q``,
    so ``first_seen < q`` gives the same answer with or without future events. That property also
    lets the online store keep one first-seen timestamp per key and still answer as of any time.
    """
    return np.asarray(first_seen_ms < query_ts, dtype=np.bool_)

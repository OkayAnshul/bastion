"""Batch executor: evaluates the shared feature definitions over the whole offline event table.

Every row's features use only events strictly before that row's ``event_ts`` (ADR-003). The
no-future-information property test in ``tests/unit/test_features.py`` checks this directly.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import polars as pl

from bastion.features.definitions import (
    WINDOWS_MS,
    from_milli_units,
    seen_before,
    to_milli_units,
    window_aggregates,
)

_TS = "_ts_ms"
_ENTITY = "_entity"


def compute_features(events: pl.DataFrame) -> pl.DataFrame:
    """One row per event, in input order: ``txn_id`` followed by every feature column."""
    return (
        events.select("txn_id")
        .join(card_velocity(events), on="txn_id", how="left", maintain_order="left")
        .join(card_device_seen_before(events), on="txn_id", how="left", maintain_order="left")
    )


def card_velocity(events: pl.DataFrame, windows: Mapping[str, int] = WINDOWS_MS) -> pl.DataFrame:
    """Per-window ``card_txn_count_<w>`` and ``card_amount_sum_<w>``, strictly before each event."""
    codes = events.select(pl.col("card_id").unique(maintain_order=True)).with_row_index(_ENTITY)
    frame = (
        events.select("txn_id", "card_id", "amount", pl.col("event_ts").dt.epoch("ms").alias(_TS))
        .join(codes, on="card_id", how="left")
        .sort(_ENTITY, _TS)
    )
    entity = frame[_ENTITY].to_numpy().astype(np.int64)
    ts = frame[_TS].to_numpy().astype(np.int64)
    amount_mu = to_milli_units(frame["amount"].to_numpy())

    columns = [frame["txn_id"]]
    for name, window_ms in windows.items():
        agg = window_aggregates(entity, ts, amount_mu, entity, ts, window_ms)
        columns.append(pl.Series(f"card_txn_count_{name}", agg.count))
        columns.append(pl.Series(f"card_amount_sum_{name}", from_milli_units(agg.amount_mu)))
    return pl.DataFrame(columns)


def card_device_seen_before(events: pl.DataFrame) -> pl.DataFrame:
    """``card_device_seen_before``: had this card used this device before? Null without a device."""
    ts = pl.col("event_ts").dt.epoch("ms")
    frame = events.select(
        "txn_id",
        "device_id",
        ts.alias(_TS),
        # Global minimum per pair: point-in-time safe given how it is compared (see seen_before).
        ts.min().over("card_id", "device_id").alias("_first_seen"),
    )
    seen = seen_before(
        frame["_first_seen"].to_numpy().astype(np.int64), frame[_TS].to_numpy().astype(np.int64)
    )
    return frame.with_columns(pl.Series("_seen", seen)).select(
        "txn_id",
        pl.when(pl.col("device_id").is_not_null())
        .then(pl.col("_seen"))
        .alias("card_device_seen_before"),
    )

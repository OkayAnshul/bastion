"""Batch executor: evaluates the shared feature definitions over the whole offline event table.

Every row's features use only events strictly before that row's ``event_ts``, and only labels that
had arrived by then (ADR-003). ``tests/unit/test_features.py`` checks this directly: deleting every
event after a cutoff must change no feature at or before it.

Feature groups follow ARCHITECTURE.md §3.2: velocity, deviation, entity risk, contextual.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from bastion.features.definitions import (
    DEVIATION_WINDOW_MS,
    LIFETIME_WINDOW_MS,
    WINDOWS_MS,
    Int64Array,
    WindowAggregate,
    age_days,
    distinct_in_window,
    from_milli_units,
    mean_and_std,
    seen_before,
    smoothed_rate,
    to_cents,
    to_milli_units,
    window_aggregates,
)

DISTINCT_WINDOWS = ("1h", "24h", "7d")
DEVICE_COUNT_WINDOWS = ("1h", "24h")
DEVICE_DISTINCT_WINDOWS = ("24h", "7d")
# Pseudo-count pulling a merchant's known fraud rate toward the global known rate.
DEFAULT_LABEL_STRENGTH = 100.0

_CODE = "_code"
_COUNT_PREFIXES = ("card_txn_count_", "card_distinct_", "device_txn_count_", "device_distinct_")
_INTEGER_FEATURES = ("card_prior_fraud_labels", "merchant_known_labels")


def feature_names(*, with_labels: bool) -> list[str]:
    """Feature columns in canonical order. Label features need the arrived-label table."""
    names: list[str] = []
    for window in WINDOWS_MS:
        names += [f"card_txn_count_{window}", f"card_amount_sum_{window}"]
    names += [f"card_distinct_merchants_{w}" for w in DISTINCT_WINDOWS]
    names += [f"card_distinct_devices_{w}" for w in DISTINCT_WINDOWS]
    names += [
        "card_amount_mean_30d",
        "card_amount_std_30d",
        "amount_to_card_mean_30d",
        "amount_zscore_30d",
    ]
    names += [f"device_txn_count_{w}" for w in DEVICE_COUNT_WINDOWS]
    names += [f"device_distinct_cards_{w}" for w in DEVICE_DISTINCT_WINDOWS]
    names += [
        "card_age_days",
        "device_age_days",
        "card_device_seen_before",
        "card_merchant_seen_before",
        "hour_of_day",
        "day_of_week",
    ]
    if with_labels:
        names += ["card_prior_fraud_labels", "merchant_known_labels", "merchant_known_fraud_rate"]
    return names


def feature_schema(*, with_labels: bool) -> dict[str, pl.DataType]:
    """Column dtypes shared by every executor, so their outputs compare exactly."""
    schema: dict[str, pl.DataType] = {}
    for name in feature_names(with_labels=with_labels):
        if name.startswith(_COUNT_PREFIXES) or name in _INTEGER_FEATURES:
            schema[name] = pl.Int64()
        elif name.endswith("_seen_before"):
            schema[name] = pl.Boolean()
        elif name in ("hour_of_day", "day_of_week"):
            schema[name] = pl.Int8()
        else:
            schema[name] = pl.Float64()
    return schema


def compute_features(
    events: pl.DataFrame,
    labels: pl.DataFrame | None = None,
    *,
    label_strength: float = DEFAULT_LABEL_STRENGTH,
) -> pl.DataFrame:
    """One row per event, in input order: ``txn_id`` then every feature.

    ``labels`` is the arrived-label table from ``bastion.data.labels.label_events``. Without it, the
    label-based entity-risk features are omitted.
    """
    columns = _columns(events)
    features: dict[str, pl.Series] = {}
    features |= _card_velocity(columns)
    features |= _card_deviation(columns)
    features |= _device_velocity(columns)
    features |= _familiarity(events, columns)
    features |= _contextual(events)
    if labels is not None:
        features |= _label_features(labels, columns, label_strength)
    schema = feature_schema(with_labels=labels is not None)
    frame = pl.DataFrame([events["txn_id"], *(features[name].alias(name) for name in schema)])
    return frame.cast(schema)  # type: ignore[arg-type]


# ------------------------------------------------------------------------------ plumbing


@dataclass(frozen=True)
class _Columns:
    """Event columns as arrays in input row order. Entity codes are -1 where the id is null."""

    ts: Int64Array
    day_start: Int64Array
    amount: npt.NDArray[np.float64]
    card: Int64Array
    device: Int64Array
    merchant: Int64Array
    card_codes: pl.DataFrame
    merchant_codes: pl.DataFrame


def _code_table(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    return frame.select(pl.col(column).drop_nulls().unique(maintain_order=True)).with_row_index(
        _CODE
    )


def _lookup(frame: pl.DataFrame, column: str, table: pl.DataFrame) -> Int64Array:
    codes = frame.select(column).join(table, on=column, how="left", maintain_order="left")[_CODE]
    return codes.cast(pl.Int64).fill_null(-1).to_numpy().astype(np.int64)


def _epoch_ms(frame: pl.DataFrame, column: str, *, day: bool = False) -> Int64Array:
    expr = pl.col(column).dt.truncate("1d") if day else pl.col(column)
    return frame.select(expr.dt.epoch("ms"))[column].to_numpy().astype(np.int64)


def _columns(events: pl.DataFrame) -> _Columns:
    card_codes = _code_table(events, "card_id")
    merchant_codes = _code_table(events, "merchant_id")
    return _Columns(
        ts=_epoch_ms(events, "event_ts"),
        day_start=_epoch_ms(events, "event_ts", day=True),
        amount=events["amount"].to_numpy().astype(np.float64),
        card=_lookup(events, "card_id", card_codes),
        device=_lookup(events, "device_id", _code_table(events, "device_id")),
        merchant=_lookup(events, "merchant_id", merchant_codes),
        card_codes=card_codes,
        merchant_codes=merchant_codes,
    )


def _windowed(
    entity: Int64Array,
    ts: Int64Array,
    values: Int64Array,
    window_ms: int,
    query_entity: Int64Array,
    query_ts: Int64Array,
) -> WindowAggregate:
    """``window_aggregates`` over history rows that have an entity (code >= 0).

    Queries without an entity are evaluated as entity 0; the caller must mask them.
    """
    valid = entity >= 0
    order = np.lexsort((ts[valid], entity[valid]))
    return window_aggregates(
        entity[valid][order],
        ts[valid][order],
        values[valid][order],
        np.maximum(query_entity, 0),
        query_ts,
        window_ms,
    )


def _masked(values: npt.ArrayLike, valid: npt.NDArray[np.bool_]) -> pl.Series:
    frame = pl.DataFrame({"value": values, "valid": valid})
    series = frame.select(pl.when(pl.col("valid")).then(pl.col("value")))["value"]
    return series.fill_nan(None) if series.dtype.is_float() else series


def _float(values: npt.NDArray[np.float64]) -> pl.Series:
    return pl.Series(values).fill_nan(None)


# ------------------------------------------------------------------------------ feature groups


def _card_velocity(c: _Columns) -> dict[str, pl.Series]:
    out: dict[str, pl.Series] = {}
    amount_mu = to_milli_units(c.amount)
    for name, window_ms in WINDOWS_MS.items():
        agg = _windowed(c.card, c.ts, amount_mu, window_ms, c.card, c.ts)
        out[f"card_txn_count_{name}"] = pl.Series(agg.count)
        out[f"card_amount_sum_{name}"] = pl.Series(from_milli_units(agg.total))
    has_device = c.device >= 0
    for name in DISTINCT_WINDOWS:
        window_ms = WINDOWS_MS[name]
        out[f"card_distinct_merchants_{name}"] = pl.Series(
            distinct_in_window(c.card, c.ts, c.merchant, c.card, c.ts, window_ms)
        )
        out[f"card_distinct_devices_{name}"] = pl.Series(
            distinct_in_window(
                c.card[has_device], c.ts[has_device], c.device[has_device], c.card, c.ts, window_ms
            )
        )
    return out


def _card_deviation(c: _Columns) -> dict[str, pl.Series]:
    cents = to_cents(c.amount)
    first = _windowed(c.card, c.ts, cents, DEVIATION_WINDOW_MS, c.card, c.ts)
    second = _windowed(c.card, c.ts, cents * cents, DEVIATION_WINDOW_MS, c.card, c.ts)
    mean, std = mean_and_std(first.count, first.total, second.total)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = c.amount / mean
        zscore = np.where(std > 0, (c.amount - mean) / std, np.nan)
    return {
        "card_amount_mean_30d": _float(mean),
        "card_amount_std_30d": _float(std),
        "amount_to_card_mean_30d": _float(ratio),
        "amount_zscore_30d": _float(zscore),
    }


def _device_velocity(c: _Columns) -> dict[str, pl.Series]:
    has_device = c.device >= 0
    ones = np.ones(c.ts.size, dtype=np.int64)
    out: dict[str, pl.Series] = {}
    for name in DEVICE_COUNT_WINDOWS:
        agg = _windowed(c.device, c.ts, ones, WINDOWS_MS[name], c.device, c.ts)
        out[f"device_txn_count_{name}"] = _masked(agg.count, has_device)
    for name in DEVICE_DISTINCT_WINDOWS:
        counts = distinct_in_window(
            c.device[has_device],
            c.ts[has_device],
            c.card[has_device],
            np.maximum(c.device, 0),
            c.ts,
            WINDOWS_MS[name],
        )
        out[f"device_distinct_cards_{name}"] = _masked(counts, has_device)
    return out


def _familiarity(events: pl.DataFrame, c: _Columns) -> dict[str, pl.Series]:
    ms = pl.col("event_ts").dt.epoch("ms")
    # Global minima: point-in-time safe only because of how they are compared (seen_before).
    first = events.select(
        ms.min().over("card_id").alias("card"),
        ms.min().over("device_id").alias("device"),
        ms.min().over("card_id", "device_id").alias("card_device"),
        ms.min().over("card_id", "merchant_id").alias("card_merchant"),
    )

    def first_seen(name: str) -> Int64Array:
        return first[name].to_numpy().astype(np.int64)

    has_device = c.device >= 0
    return {
        "card_age_days": pl.Series(age_days(first_seen("card"), c.ts)),
        "device_age_days": _masked(age_days(first_seen("device"), c.ts), has_device),
        "card_device_seen_before": _masked(
            seen_before(first_seen("card_device"), c.ts), has_device
        ),
        "card_merchant_seen_before": pl.Series(seen_before(first_seen("card_merchant"), c.ts)),
    }


def _contextual(events: pl.DataFrame) -> dict[str, pl.Series]:
    ts = events["event_ts"]
    return {"hour_of_day": ts.dt.hour(), "day_of_week": ts.dt.weekday()}


def _label_features(labels: pl.DataFrame, c: _Columns, strength: float) -> dict[str, pl.Series]:
    """Entity risk from labels that had arrived before the decision.

    ``card_prior_fraud_labels`` counts chargebacks on the card with ``label_ts`` strictly before the
    transaction. The merchant rate refreshes daily, like a nightly batch job: only labels that
    arrived before the start of the transaction's day count.
    """
    fraud_labels = labels.filter(pl.col("is_fraud"))
    card_prior = _windowed(
        _lookup(fraud_labels, "card_id", c.card_codes),
        _epoch_ms(fraud_labels, "label_ts"),
        np.ones(fraud_labels.height, dtype=np.int64),
        LIFETIME_WINDOW_MS,
        c.card,
        c.ts,
    )
    label_day = _epoch_ms(labels, "label_ts", day=True)
    is_fraud = labels["is_fraud"].cast(pl.Int64).to_numpy().astype(np.int64)
    merchant = _windowed(
        _lookup(labels, "merchant_id", c.merchant_codes),
        label_day,
        is_fraud,
        LIFETIME_WINDOW_MS,
        c.merchant,
        c.day_start,
    )
    everyone = _windowed(
        np.zeros(labels.height, dtype=np.int64),
        label_day,
        is_fraud,
        LIFETIME_WINDOW_MS,
        np.zeros(c.ts.size, dtype=np.int64),
        c.day_start,
    )
    rate = smoothed_rate(merchant.total, merchant.count, everyone.total, everyone.count, strength)
    return {
        "card_prior_fraud_labels": pl.Series(card_prior.count),
        "merchant_known_labels": pl.Series(merchant.count),
        "merchant_known_fraud_rate": _float(rate),
    }

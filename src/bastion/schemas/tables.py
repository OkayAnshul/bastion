"""Contract for the offline canonical event table (the offline store's source of truth).

Both producers of historical events, the IEEE-CIS adapter and the synthetic generator, write
this layout: canonical columns from §3.1 plus flat ``attr_*`` vendor-attribute columns. Unlike
the per-event JSON contract, the offline table *does* carry ``is_fraud``, because it is the
labelled history used for training and evaluation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import polars as pl

from bastion.schemas.events import TransactionEvent

ATTRIBUTE_PREFIX = "attr_"

CANONICAL_SCHEMA: dict[str, pl.DataType] = {
    "txn_id": pl.String(),
    "event_ts": pl.Datetime("ms", "UTC"),
    "card_id": pl.String(),
    "device_id": pl.String(),
    "merchant_id": pl.String(),
    "merchant_category": pl.String(),
    "ip": pl.String(),
    "amount": pl.Float64(),
    "currency": pl.String(),
    "channel": pl.String(),
    "is_fraud": pl.Boolean(),
}

NON_NULL_COLUMNS = (
    "txn_id",
    "event_ts",
    "card_id",
    "merchant_id",
    "merchant_category",
    "amount",
    "currency",
    "channel",
)


class EventTableError(ValueError):
    """The event table violates the offline contract. The message lists every violation."""


def validate_event_table(df: pl.DataFrame) -> None:
    """Raise ``EventTableError`` listing all contract violations, or return silently.

    Sorting by ``event_ts`` is part of the contract: replay and batch feature computation both
    rely on it.
    """
    problems: list[str] = []

    missing = [c for c in CANONICAL_SCHEMA if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")

    wrong_types = {
        c: f"{df.schema[c]} != {expected}"
        for c, expected in CANONICAL_SCHEMA.items()
        if c in df.columns and df.schema[c] != expected
    }
    if wrong_types:
        problems.append(f"wrong dtypes: {wrong_types}")

    unexpected = [
        c for c in df.columns if c not in CANONICAL_SCHEMA and not c.startswith(ATTRIBUTE_PREFIX)
    ]
    if unexpected:
        problems.append(
            f"unexpected columns (vendor fields need the '{ATTRIBUTE_PREFIX}' prefix): {unexpected}"
        )

    if problems:  # the value checks below assume the columns exist with the right types
        raise EventTableError("; ".join(problems))

    nulls = {c: n for c in NON_NULL_COLUMNS if (n := df[c].null_count())}
    if nulls:
        problems.append(f"nulls in required columns: {nulls}")
    if df["txn_id"].n_unique() != df.height:
        problems.append("txn_id is not unique")
    if df.filter(pl.col("amount") <= 0).height:
        problems.append("non-positive amounts")
    if not df["event_ts"].is_sorted():
        problems.append("rows are not sorted by event_ts")

    if problems:
        raise EventTableError("; ".join(problems))


def row_to_event(row: Mapping[str, object], *, include_label: bool = False) -> TransactionEvent:
    """Build a ``TransactionEvent`` from one row of the canonical table.

    Labels are stripped unless ``include_label`` is set, so a replayed event looks exactly like a
    live authorisation. NaN attributes become ``None``.
    """
    fields = {c: row[c] for c in CANONICAL_SCHEMA if c != "is_fraud"}
    if include_label:
        fields["is_fraud"] = row["is_fraud"]
    fields["attributes"] = {
        key.removeprefix(ATTRIBUTE_PREFIX): (
            None if isinstance(value, float) and math.isnan(value) else value
        )
        for key, value in row.items()
        if key.startswith(ATTRIBUTE_PREFIX)
    }
    return TransactionEvent.model_validate(fields)

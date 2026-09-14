"""Map IEEE-CIS rows onto Bastion's canonical event table (ADR-011).

IEEE-CIS is anonymised: it has no explicit card, device, merchant or IP identifiers. This module
builds documented proxies for them at the dataset boundary, so the rest of the platform (streaming,
Redis keys, graph) is written against the canonical schema only. Caveats are listed in
``docs/data_dictionary.md``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from bastion.data.ieee_cis import load_raw
from bastion.log import get_logger
from bastion.schemas.tables import ATTRIBUTE_PREFIX, validate_event_table

log = get_logger(__name__)

# TransactionDT is seconds from an undisclosed reference; the first value is one day. The anchor
# is an assumption: only relative time is meaningful (ADR-011).
ANCHOR = datetime(2017, 11, 30, tzinfo=UTC)
SECONDS_PER_DAY = 86_400

# D1n = the account's first-seen day. It separates customers who share card and address fields.
CARD_KEY_COLUMNS = ("card1", "card2", "card3", "card4", "card5", "card6", "addr1", "D1n")
DEVICE_KEY_COLUMNS = ("DeviceInfo", "id_30", "id_31", "id_33")
MERCHANT_KEY_COLUMNS = ("ProductCD", "R_emaildomain")

CONSUMED_COLUMNS = ("TransactionID", "isFraud", "TransactionDT", "TransactionAmt")
EVENTS_FILE = "events_ieee.parquet"
_NULL_TOKEN = "<null>"


def stable_id(prefix: str, key: str) -> str:
    """Deterministic short id: the same key maps to the same id on every machine and version.

    ``polars.Expr.hash`` is documented as unstable across Polars versions, so it must not be used
    for ids that get persisted to Parquet, Redis and MLflow artifacts.
    """
    return f"{prefix}_{hashlib.blake2b(key.encode(), digest_size=8).hexdigest()}"


def _composite_key(columns: Sequence[str]) -> pl.Expr:
    # Nulls get an explicit token: concat_str would otherwise null the whole key.
    return pl.concat_str(
        [pl.col(c).cast(pl.String).fill_null(_NULL_TOKEN) for c in columns], separator="|"
    )


def _stable_ids(keys: pl.Series, prefix: str) -> pl.Series:
    mapping = {key: stable_id(prefix, key) for key in keys.drop_nulls().unique().to_list()}
    return keys.replace_strict(mapping, default=None, return_dtype=pl.String)


def to_canonical(transactions: pl.DataFrame, identity: pl.DataFrame) -> pl.DataFrame:
    """Join the raw tables and emit the canonical event table, validated and sorted by time."""
    raw_columns = [c for c in transactions.columns if c not in CONSUMED_COLUMNS] + [
        c for c in identity.columns if c != "TransactionID"
    ]

    df = transactions.join(identity, on="TransactionID", how="left", validate="1:1").with_columns(
        D1n=((pl.col("TransactionDT") // SECONDS_PER_DAY) - pl.col("D1")).cast(pl.Int64)
    )
    has_fingerprint = pl.any_horizontal(pl.col(c).is_not_null() for c in DEVICE_KEY_COLUMNS)
    keys = df.select(
        card=_composite_key(CARD_KEY_COLUMNS),
        device=pl.when(has_fingerprint).then(_composite_key(DEVICE_KEY_COLUMNS)),
        merchant=_composite_key(MERCHANT_KEY_COLUMNS),
    )

    # Positional, aliased expressions keep the canonical columns first: Polars places keyword
    # expressions *after* positional ones, which would bury them behind ~430 attributes.
    canonical = df.select(
        (pl.lit("ieee-") + pl.col("TransactionID").cast(pl.String)).alias("txn_id"),
        (pl.lit(ANCHOR) + pl.duration(seconds=pl.col("TransactionDT")))
        .cast(pl.Datetime("ms", "UTC"))
        .alias("event_ts"),
        _stable_ids(keys["card"], "c").alias("card_id"),
        _stable_ids(keys["device"], "d").alias("device_id"),
        _stable_ids(keys["merchant"], "m").alias("merchant_id"),
        pl.col("ProductCD").alias("merchant_category"),
        pl.lit(None, dtype=pl.String).alias("ip"),
        pl.col("TransactionAmt").cast(pl.Float64).alias("amount"),
        pl.lit("USD").alias("currency"),
        pl.lit("ecom").alias("channel"),
        (pl.col("isFraud") == 1).alias("is_fraud"),
        *[pl.col(c).alias(f"{ATTRIBUTE_PREFIX}{c}") for c in raw_columns],
    ).sort("event_ts", "txn_id")

    validate_event_table(canonical)
    return canonical


def prepare(raw_dir: Path, processed_dir: Path) -> Path:
    """Build ``processed_dir/events_ieee.parquet`` from the raw IEEE-CIS train files."""
    transactions, identity = load_raw(raw_dir)
    canonical = to_canonical(transactions, identity)
    processed_dir.mkdir(parents=True, exist_ok=True)
    out = processed_dir / EVENTS_FILE
    canonical.write_parquet(out, compression="zstd")
    log.info(
        "ieee_cis.prepared",
        path=str(out),
        rows=canonical.height,
        columns=canonical.width,
        cards=canonical["card_id"].n_unique(),
    )
    return out

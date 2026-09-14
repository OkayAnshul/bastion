"""Deliberately leaky features, kept only for the Phase 1 leakage experiment (ADR-003).

This emits exactly the columns of ``bastion.features.batch.compute_features``, computed the way a
hurried notebook would: whole-dataset group-bys. Each block names its leak. The leakage experiment
trains on both pipelines and publishes the gap. ``tests/unit/test_naive_features.py`` shows these
features fail the no-future-information test that the real pipeline passes, and checks that nothing
outside the experiment imports this module.
"""

from __future__ import annotations

import polars as pl

from bastion.features.batch import (
    DEVICE_COUNT_WINDOWS,
    DEVICE_DISTINCT_WINDOWS,
    DISTINCT_WINDOWS,
    feature_schema,
)
from bastion.features.definitions import MS_PER_DAY, WINDOWS_MS

# Calendar buckets standing in for sliding windows: the classic notebook shortcut.
_BUCKETS = {"1m": "1m", "10m": "10m", "1h": "1h", "24h": "1d", "7d": "1w"}


def compute_naive_features(events: pl.DataFrame, *, with_labels: bool = False) -> pl.DataFrame:
    ts = pl.col("event_ts")
    ms = ts.dt.epoch("ms")
    has_device = pl.col("device_id").is_not_null()

    def bucket(window: str) -> pl.Expr:
        return ts.dt.truncate(_BUCKETS[window])

    exprs: list[pl.Expr] = []

    # LEAK (self + future): a calendar bucket contains the transaction itself and every later
    # transaction in the same bucket.
    for w in WINDOWS_MS:
        exprs.append(pl.len().over("card_id", bucket(w)).alias(f"card_txn_count_{w}"))
        exprs.append(
            pl.col("amount").sum().over("card_id", bucket(w)).alias(f"card_amount_sum_{w}")
        )
    for w in DISTINCT_WINDOWS:
        exprs.append(
            pl.col("merchant_id")
            .n_unique()
            .over("card_id", bucket(w))
            .alias(f"card_distinct_merchants_{w}")
        )
        exprs.append(
            pl.col("device_id")
            .drop_nulls()
            .n_unique()
            .over("card_id", bucket(w))
            .alias(f"card_distinct_devices_{w}")
        )

    # LEAK (future): statistics over the card's entire history, including later transactions.
    mean = pl.col("amount").mean().over("card_id")
    std = pl.col("amount").std(ddof=0).over("card_id")
    exprs += [
        mean.alias("card_amount_mean_30d"),
        std.alias("card_amount_std_30d"),
        (pl.col("amount") / mean).alias("amount_to_card_mean_30d"),
        pl.when(std > 0).then((pl.col("amount") - mean) / std).alias("amount_zscore_30d"),
    ]

    # LEAK (self + future), as for cards.
    for w in DEVICE_COUNT_WINDOWS:
        exprs.append(
            pl.when(has_device)
            .then(pl.len().over("device_id", bucket(w)))
            .alias(f"device_txn_count_{w}")
        )
    for w in DEVICE_DISTINCT_WINDOWS:
        exprs.append(
            pl.when(has_device)
            .then(pl.col("card_id").n_unique().over("device_id", bucket(w)))
            .alias(f"device_distinct_cards_{w}")
        )

    # NOT a leak: "time since first seen" compares against the earliest time, which later events
    # cannot lower. The naive version happens to be correct here, which is itself worth knowing.
    exprs += [
        ((ms - ms.min().over("card_id")) / MS_PER_DAY).alias("card_age_days"),
        pl.when(has_device)
        .then((ms - ms.min().over("device_id")) / MS_PER_DAY)
        .alias("device_age_days"),
    ]

    # LEAK (future): "seen before" as "this pair occurs more than once anywhere", which is true even
    # on the pair's very first transaction.
    exprs += [
        pl.when(has_device)
        .then(pl.len().over("card_id", "device_id") > 1)
        .alias("card_device_seen_before"),
        (pl.len().over("card_id", "merchant_id") > 1).alias("card_merchant_seen_before"),
        ts.dt.hour().alias("hour_of_day"),
        ts.dt.weekday().alias("day_of_week"),
    ]

    if with_labels:
        # LEAK (target): labels counted regardless of when they arrived, including this row's own
        # label. With IEEE-CIS's forward label propagation this is close to reading the answer.
        fraud = pl.col("is_fraud").cast(pl.Int64)
        exprs += [
            fraud.sum().over("card_id").alias("card_prior_fraud_labels"),
            pl.len().over("merchant_id").alias("merchant_known_labels"),
            fraud.mean().over("merchant_id").alias("merchant_known_fraud_rate"),
        ]

    schema = feature_schema(with_labels=with_labels)
    return events.select("txn_id", *exprs).select("txn_id", *schema).cast(schema)  # type: ignore[arg-type]

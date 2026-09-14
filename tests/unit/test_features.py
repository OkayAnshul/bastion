import math
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bastion.data.labels import LabelDelayConfig, label_events
from bastion.features.batch import compute_features, feature_schema
from bastion.features.definitions import (
    WINDOWS_MS,
    distinct_in_window,
    from_milli_units,
    to_milli_units,
    window_aggregates,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)  # a Thursday
DAY = 86_400
Row = tuple[str, int, float, str | None]  # (card_id, seconds after T0, amount, device_id)
LABELS = LabelDelayConfig(seed=11, fraud_median_days=5.0, fraud_sigma=0.6, maturity_days=14.0)


def _events(
    rows: list[Row], merchants: list[str] | None = None, fraud: list[bool] | None = None
) -> pl.DataFrame:
    n = len(rows)
    return pl.DataFrame(
        {
            "txn_id": [f"t{i}" for i in range(n)],
            "event_ts": pl.Series(
                [T0 + timedelta(seconds=s) for _, s, _, _ in rows], dtype=pl.Datetime("ms", "UTC")
            ),
            "card_id": [card for card, _, _, _ in rows],
            "device_id": pl.Series([device for _, _, _, device in rows], dtype=pl.String),
            "merchant_id": merchants if merchants is not None else ["m1"] * n,
            "merchant_category": ["W"] * n,
            "amount": [amount for _, _, amount, _ in rows],
            "is_fraud": fraud if fraud is not None else [False] * n,
        }
    )


def _feature(rows: list[Row], column: str, merchants: list[str] | None = None) -> list[object]:
    return compute_features(_events(rows, merchants))[column].to_list()


def _labels(rows: list[tuple[str, str, int, int, bool]]) -> pl.DataFrame:
    """rows: (card_id, merchant_id, event seconds, label seconds, is_fraud)."""
    ts = pl.Datetime("ms", "UTC")
    return pl.DataFrame(
        {
            "txn_id": [f"l{i}" for i in range(len(rows))],
            "card_id": [r[0] for r in rows],
            "merchant_id": [r[1] for r in rows],
            "event_ts": pl.Series([T0 + timedelta(seconds=r[2]) for r in rows], dtype=ts),
            "label_ts": pl.Series([T0 + timedelta(seconds=r[3]) for r in rows], dtype=ts),
            "is_fraud": [r[4] for r in rows],
        }
    )


# ------------------------------------------------------------------------------ schema


def test_feature_columns_follow_the_declared_schema() -> None:
    events = _events([("c", 0, 5.0, "d1"), ("c", 10, 7.0, None)])
    assert compute_features(events).schema == {
        "txn_id": pl.String,
        **feature_schema(with_labels=False),
    }
    with_labels = compute_features(events, label_events(events, LABELS))
    assert with_labels.schema == {"txn_id": pl.String, **feature_schema(with_labels=True)}


# ------------------------------------------------------------------------------ velocity


def test_window_includes_its_start_and_excludes_the_query_time() -> None:
    rows: list[Row] = [
        ("c", 0, 1.0, None),
        ("c", 30, 1.0, None),
        ("c", 60, 1.0, None),
        ("c", 120, 1.0, None),
    ]
    # at 60s the 1m window is [0s, 60s): events at 0s and 30s. At 120s it is [60s, 120s): just 60s.
    assert _feature(rows, "card_txn_count_1m") == [0, 1, 2, 1]


def test_events_sharing_a_timestamp_do_not_see_each_other() -> None:
    rows: list[Row] = [("c", 10, 5.0, None), ("c", 10, 7.0, None), ("c", 11, 1.0, None)]
    assert _feature(rows, "card_txn_count_1h") == [0, 0, 2]
    assert _feature(rows, "card_amount_sum_1h") == [0.0, 0.0, 12.0]


def test_amount_sums_are_exact() -> None:
    rows: list[Row] = [("c", 0, 0.1, None), ("c", 1, 0.2, None), ("c", 2, 9.99, None)]
    # float addition would give 0.30000000000000004; milli-unit integers give exactly 0.3
    assert _feature(rows, "card_amount_sum_1h")[2] == 0.3


def test_other_cards_never_leak_into_a_window() -> None:
    rows: list[Row] = [("a", 0, 1.0, None), ("a", 50, 1.0, None), ("b", 55, 1.0, None)]
    assert _feature(rows, "card_txn_count_1m") == [0, 1, 0]


def test_online_style_call_matches_batch_for_one_entity() -> None:
    rng = np.random.default_rng(3)
    rows: list[Row] = [
        (
            str(rng.choice(["a", "b", "c"])),
            int(rng.integers(0, 3 * DAY)),
            float(rng.uniform(1, 500)),
            None,
        )
        for _ in range(300)
    ]
    events = _events(rows)
    batch = compute_features(events)

    query = events.row(123, named=True)
    history = events.filter(
        (pl.col("card_id") == query["card_id"]) & (pl.col("event_ts") < query["event_ts"])
    ).sort("event_ts")
    ts = history["event_ts"].dt.epoch("ms").to_numpy().astype(np.int64)
    amount_mu = to_milli_units(history["amount"].to_numpy())
    zeros = np.zeros(ts.size, dtype=np.int64)
    q_ts = np.array([int(query["event_ts"].timestamp() * 1000)], dtype=np.int64)

    for name, window_ms in WINDOWS_MS.items():
        online = window_aggregates(
            zeros, ts, amount_mu, np.zeros(1, dtype=np.int64), q_ts, window_ms
        )
        assert online.count[0] == batch[f"card_txn_count_{name}"][123]
        assert from_milli_units(online.total)[0] == batch[f"card_amount_sum_{name}"][123]


def test_unsorted_history_is_rejected() -> None:
    ts = np.array([5, 1], dtype=np.int64)
    zeros = np.zeros(2, dtype=np.int64)
    with pytest.raises(ValueError, match="sorted"):
        window_aggregates(zeros, ts, zeros, zeros[:1], ts[:1], 60_000)


history_rows = st.lists(
    st.tuples(st.integers(0, 2), st.integers(0, 3), st.integers(0, 100)), max_size=40
)
query_rows = st.lists(st.tuples(st.integers(0, 2), st.integers(0, 120)), min_size=1, max_size=20)


@settings(max_examples=200, deadline=None)
@given(history=history_rows, queries=query_rows, window=st.integers(1, 60))
def test_distinct_in_window_matches_brute_force(
    history: list[tuple[int, int, int]], queries: list[tuple[int, int]], window: int
) -> None:
    h = np.array(history, dtype=np.int64).reshape(-1, 3)
    q = np.array(queries, dtype=np.int64).reshape(-1, 2)
    got = distinct_in_window(h[:, 0], h[:, 2], h[:, 1], q[:, 0], q[:, 1], window)
    expected = [
        len({v for e, v, t in history if e == qe and qt - window <= t < qt}) for qe, qt in queries
    ]
    assert got.tolist() == expected


def test_distinct_merchants_and_devices_per_card() -> None:
    rows: list[Row] = [
        ("c", 0, 1.0, "dA"),
        ("c", 10, 1.0, "dB"),
        ("c", 20, 1.0, None),
        ("c", 30, 1.0, "dA"),
    ]
    merchants = ["m1", "m2", "m1", "m3"]
    assert _feature(rows, "card_distinct_merchants_1h", merchants) == [0, 1, 2, 2]
    assert _feature(rows, "card_distinct_devices_1h", merchants) == [
        0,
        1,
        2,
        2,
    ]  # null is not a device


# ------------------------------------------------------------------------------ deviation


def test_deviation_against_the_cards_recent_history() -> None:
    rows: list[Row] = [
        ("c", 0, 10.0, None),
        ("c", 1, 20.0, None),
        ("c", 2, 30.0, None),
        ("c", 3, 40.0, None),
    ]
    features = compute_features(_events(rows))
    assert features["card_amount_mean_30d"].to_list() == [None, 10.0, 15.0, 20.0]
    assert features["card_amount_std_30d"][1] is None  # one earlier event: no spread yet
    std = math.sqrt(((10 - 20) ** 2 + 0 + (30 - 20) ** 2) / 3)
    assert features["card_amount_std_30d"][3] == pytest.approx(std)
    assert features["amount_to_card_mean_30d"][3] == pytest.approx(2.0)
    assert features["amount_zscore_30d"][3] == pytest.approx((40 - 20) / std)


def test_zscore_is_null_when_history_has_no_spread() -> None:
    rows: list[Row] = [("c", 0, 10.0, None), ("c", 1, 10.0, None), ("c", 2, 50.0, None)]
    assert _feature(rows, "amount_zscore_30d")[2] is None


def test_deviation_forgets_events_older_than_30_days() -> None:
    rows: list[Row] = [
        ("c", 0, 1000.0, None),
        ("c", 31 * DAY, 10.0, None),
        ("c", 31 * DAY + 1, 10.0, None),
    ]
    assert _feature(rows, "card_amount_mean_30d")[2] == 10.0


# ------------------------------------------------------------------------------ entity risk


def test_device_features_are_null_without_a_device() -> None:
    rows: list[Row] = [
        ("c1", 0, 1.0, "dX"),
        ("c2", 10, 1.0, "dX"),
        ("c3", 20, 1.0, "dX"),
        ("c1", 30, 1.0, None),
    ]
    assert _feature(rows, "device_distinct_cards_24h") == [0, 1, 2, None]
    assert _feature(rows, "device_txn_count_1h") == [0, 1, 2, None]
    assert _feature(rows, "device_age_days")[3] is None


def test_ages_count_days_since_first_seen() -> None:
    rows: list[Row] = [("c", 0, 1.0, "dA"), ("c", 2 * DAY, 1.0, None), ("k", 3 * DAY, 1.0, "dA")]
    assert _feature(rows, "card_age_days") == [0.0, 2.0, 0.0]
    assert _feature(rows, "device_age_days") == [0.0, None, 3.0]


def test_device_seen_before_is_per_card_and_strictly_earlier() -> None:
    rows: list[Row] = [
        ("c", 0, 1.0, "d1"),  # first use of d1 on c
        ("c", 5, 1.0, "d1"),  # seen at 0s
        ("c", 5, 1.0, "d2"),  # first use of d2
        ("c", 5, 1.0, "d2"),  # same second as the first use: not strictly earlier
        ("c", 9, 1.0, None),  # no device: unknown, not "new"
        ("k", 10, 1.0, "d1"),  # d1 is new for card k even though c used it
    ]
    assert _feature(rows, "card_device_seen_before") == [False, True, False, False, None, False]


def test_merchant_seen_before_is_per_card() -> None:
    rows: list[Row] = [
        ("c", 0, 1.0, None),
        ("c", 5, 1.0, None),
        ("c", 6, 1.0, None),
        ("k", 7, 1.0, None),
    ]
    merchants = ["m1", "m1", "m2", "m1"]
    assert _feature(rows, "card_merchant_seen_before", merchants) == [False, True, False, False]


def test_contextual_hour_and_weekday() -> None:
    features = compute_features(_events([("c", 3 * 3600, 1.0, None)]))
    assert features["hour_of_day"][0] == 3
    assert features["day_of_week"][0] == 4  # Thursday, Monday = 1


# ------------------------------------------------------------------------------ arrived labels


def test_card_prior_fraud_counts_only_chargebacks_that_had_arrived() -> None:
    events = _events([("c", 0, 1.0, None), ("c", 100, 1.0, None), ("c", 200, 1.0, None)])
    labels = _labels(
        [("c", "m1", 0, 150, True)]
    )  # the first transaction's chargeback lands at 150s
    assert compute_features(events, labels)["card_prior_fraud_labels"].to_list() == [0, 0, 1]


def test_merchant_rate_uses_labels_known_by_the_start_of_the_day() -> None:
    events = _events([("c", 13 * 3600, 1.0, None), ("c", DAY, 1.0, None)])
    labels = _labels(
        [
            ("x", "m1", 0, 12 * 3600, False),  # m1 label arrives mid-day 0
            ("y", "m2", 0, 6 * 3600, True),  # another merchant's fraud, also day 0
        ]
    )
    features = compute_features(events, labels, label_strength=100.0)
    # Day 0 events see nothing; day 1 events see both labels (global prior 1/2).
    assert features["merchant_known_labels"].to_list() == [0, 1]
    assert features["merchant_known_fraud_rate"][0] is None
    assert features["merchant_known_fraud_rate"][1] == pytest.approx((0 + 100 * 0.5) / (1 + 100))


# ------------------------------------------------------------------------------ ADR-003 invariant


event_rows = st.lists(
    st.tuples(
        st.sampled_from(["c1", "c2", "c3"]),
        st.integers(min_value=0, max_value=3 * DAY),
        st.floats(min_value=0.01, max_value=5_000, allow_nan=False),
        st.sampled_from([None, "d1", "d2"]),
    ),
    min_size=1,
    max_size=60,
)


@settings(max_examples=75, deadline=None)
@given(rows=event_rows, cutoff=st.integers(min_value=0, max_value=3 * DAY))
def test_features_never_depend_on_later_events(rows: list[Row], cutoff: int) -> None:
    """ADR-003 invariant: dropping events after a cutoff changes no feature at or before it."""
    events = _events(rows)
    keep = pl.Series([seconds <= cutoff for _, seconds, _, _ in rows])
    expected = compute_features(events).filter(keep)
    truncated = compute_features(events.filter(keep))
    assert truncated.equals(expected)


labelled_rows = st.lists(
    st.tuples(
        st.sampled_from(["c1", "c2"]),
        st.integers(min_value=0, max_value=20 * DAY),
        st.floats(min_value=0.01, max_value=5_000, allow_nan=False),
        st.sampled_from([None, "d1"]),
        st.booleans(),
        st.sampled_from(["m1", "m2"]),
    ),
    min_size=1,
    max_size=50,
)


@settings(max_examples=60, deadline=None)
@given(rows=labelled_rows, cutoff=st.integers(min_value=0, max_value=20 * DAY))
def test_label_features_never_depend_on_later_events_or_labels(
    rows: list[tuple[str, int, float, str | None, bool, str]], cutoff: int
) -> None:
    events = _events(
        [(r[0], r[1], r[2], r[3]) for r in rows],
        merchants=[r[5] for r in rows],
        fraud=[r[4] for r in rows],
    )
    keep = pl.Series([r[1] <= cutoff for r in rows])
    expected = compute_features(events, label_events(events, LABELS)).filter(keep)
    past = events.filter(keep)
    assert compute_features(past, label_events(past, LABELS)).equals(expected)

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bastion.features.batch import compute_features
from bastion.features.definitions import (
    WINDOWS_MS,
    from_milli_units,
    to_milli_units,
    window_aggregates,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)
Row = tuple[str, int, float, str | None]  # (card_id, seconds after T0, amount, device_id)


def _events(rows: list[Row]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "txn_id": [f"t{i}" for i in range(len(rows))],
            "event_ts": pl.Series(
                [T0 + timedelta(seconds=s) for _, s, _, _ in rows], dtype=pl.Datetime("ms", "UTC")
            ),
            "card_id": [card for card, _, _, _ in rows],
            "amount": [amount for _, _, amount, _ in rows],
            "device_id": pl.Series([device for _, _, _, device in rows], dtype=pl.String),
        }
    )


def _feature(rows: list[Row], column: str) -> list[object]:
    return compute_features(_events(rows))[column].to_list()


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
            int(rng.integers(0, 3 * 86_400)),
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
        assert from_milli_units(online.amount_mu)[0] == batch[f"card_amount_sum_{name}"][123]


def test_unsorted_history_is_rejected() -> None:
    ts = np.array([5, 1], dtype=np.int64)
    zeros = np.zeros(2, dtype=np.int64)
    with pytest.raises(ValueError, match="sorted"):
        window_aggregates(zeros, ts, zeros, zeros[:1], ts[:1], 60_000)


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


event_rows = st.lists(
    st.tuples(
        st.sampled_from(["c1", "c2", "c3"]),
        st.integers(min_value=0, max_value=3 * 86_400),
        st.floats(min_value=0.01, max_value=5_000, allow_nan=False),
        st.sampled_from([None, "d1", "d2"]),
    ),
    min_size=1,
    max_size=60,
)


@settings(max_examples=75, deadline=None)
@given(rows=event_rows, cutoff=st.integers(min_value=0, max_value=3 * 86_400))
def test_features_never_depend_on_later_events(rows: list[Row], cutoff: int) -> None:
    """ADR-003 invariant: dropping events after a cutoff changes no feature at or before it."""
    events = _events(rows)
    keep = pl.Series([seconds <= cutoff for _, seconds, _, _ in rows])
    expected = compute_features(events).filter(keep)
    truncated = compute_features(events.filter(keep))
    assert truncated.equals(expected)

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from bastion.data.splits import (
    SplitConfig,
    SplitWindow,
    assign_splits,
    load_split_config,
    select_split,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG = SplitConfig(
    windows=(
        SplitWindow(name="train", start_day=0, end_day=2),
        SplitWindow(name="gap", start_day=2, end_day=3),
        SplitWindow(name="test", start_day=3, end_day=None),
    )
)


def _events(timestamps: list[datetime]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "txn_id": [f"t{i}" for i in range(len(timestamps))],
            "event_ts": pl.Series(timestamps, dtype=pl.Datetime("ms", "UTC")),
        }
    )


def test_windows_are_calendar_days_from_the_first_day() -> None:
    events = _events(
        [
            datetime(2026, 1, 1, 15, 0, tzinfo=UTC),  # first event is mid-afternoon: still day 0
            datetime(2026, 1, 2, 23, 59, 59, tzinfo=UTC),  # last second of day 1
            datetime(2026, 1, 3, 0, 0, 0, tzinfo=UTC),  # first instant of day 2
            datetime(2026, 1, 4, 0, 0, 0, tzinfo=UTC),
            datetime(2026, 3, 1, tzinfo=UTC),  # open-ended last window
        ]
    )
    assert assign_splits(events, CONFIG)["split"].to_list() == [
        "train",
        "train",
        "gap",
        "test",
        "test",
    ]


def test_row_order_is_preserved_never_shuffled() -> None:
    events = _events(
        [
            datetime(2026, 1, 5, tzinfo=UTC),
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 3, tzinfo=UTC),
        ]
    )
    result = assign_splits(events, CONFIG)
    assert result["txn_id"].to_list() == ["t0", "t1", "t2"]
    assert result["split"].to_list() == ["test", "train", "gap"]


def test_bounded_last_window_leaves_later_rows_unassigned() -> None:
    bounded = SplitConfig(windows=(SplitWindow(name="only", start_day=0, end_day=1),))
    events = _events([datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC)])
    assert assign_splits(events, bounded)["split"].to_list() == ["only", None]


def test_select_split_returns_only_that_window() -> None:
    events = _events([datetime(2026, 1, d, tzinfo=UTC) for d in (1, 2, 3, 4)])
    assert select_split(events, CONFIG, "train")["txn_id"].to_list() == ["t0", "t1"]
    with pytest.raises(KeyError):
        select_split(events, CONFIG, "validation")


@pytest.mark.parametrize(
    "windows",
    [
        [("a", 0, 2), ("b", 3, None)],  # gap between windows
        [("a", 0, 2), ("b", 1, None)],  # overlap
        [("a", 0, None), ("b", 2, None)],  # open-ended window that is not last
        [("a", 0, 2), ("a", 2, None)],  # duplicate names
        [("a", 1, 2)],  # does not start at day 0
        [("a", 0, 0)],  # empty window
    ],
)
def test_invalid_window_layouts_are_rejected(windows: list[tuple[str, int, int | None]]) -> None:
    with pytest.raises(ValidationError):
        SplitConfig(
            windows=tuple(SplitWindow(name=n, start_day=s, end_day=e) for n, s, e in windows)
        )


def test_repository_split_config_is_valid() -> None:
    config = load_split_config(REPO_ROOT / "configs")
    assert [w.name for w in config.windows] == [
        "train",
        "maturity_gap",
        "early_stopping",
        "calibration",
        "test",
    ]

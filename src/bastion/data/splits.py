"""Temporal split windows (ADR-007).

Rows are assigned to windows purely by event time; nothing is ever shuffled. The maturity gap
between the training and evaluation windows stands in for chargeback delay: at the training cutoff,
recent transactions' fraud labels would not have arrived yet.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Self

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from bastion.config import load_config

SPLIT_COLUMN = "split"


class SplitWindow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    start_day: int = Field(ge=0)
    end_day: int | None = None  # exclusive; None means "through the last event"


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    windows: tuple[SplitWindow, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _contiguous_and_ordered(self) -> Self:
        names = [w.name for w in self.windows]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate window names: {names}")
        if self.windows[0].start_day != 0:
            raise ValueError("the first window must start at day 0")
        for window in self.windows:
            if window.end_day is not None and window.end_day <= window.start_day:
                raise ValueError(f"window {window.name!r} is empty or reversed")
        for current, following in itertools.pairwise(self.windows):
            if current.end_day is None:
                raise ValueError(f"only the last window may be open-ended, not {current.name!r}")
            if current.end_day != following.start_day:
                raise ValueError(
                    f"windows must be contiguous: {current.name!r} ends at day "
                    f"{current.end_day} but {following.name!r} starts at day {following.start_day}"
                )
        return self

    def window(self, name: str) -> SplitWindow:
        for window in self.windows:
            if window.name == name:
                return window
        raise KeyError(f"unknown split {name!r}; configured: {[w.name for w in self.windows]}")


def load_split_config(config_dir: Path | None = None) -> SplitConfig:
    return load_config("splits", SplitConfig, config_dir)


def assign_splits(events: pl.DataFrame, config: SplitConfig) -> pl.DataFrame:
    """Return ``events`` with a ``split`` column, preserving row order.

    Day 0 is the first UTC calendar day present in ``events``. Rows outside every window (possible
    only when the last window is bounded) get a null split.
    """
    if events.is_empty():
        return events.with_columns(pl.lit(None, dtype=pl.String).alias(SPLIT_COLUMN))

    origin = events["event_ts"].dt.truncate("1d").min()
    day = (pl.col("event_ts").dt.truncate("1d") - pl.lit(origin)).dt.total_days()

    label: pl.Expr = pl.lit(None, dtype=pl.String)
    for window in reversed(config.windows):
        if window.end_day is None:
            in_window = day >= window.start_day
        else:
            in_window = day.is_between(window.start_day, window.end_day, closed="left")
        label = pl.when(in_window).then(pl.lit(window.name)).otherwise(label)
    return events.with_columns(label.alias(SPLIT_COLUMN))


def select_split(events: pl.DataFrame, config: SplitConfig, name: str) -> pl.DataFrame:
    """Rows of one split, in their original order."""
    config.window(name)  # unknown names fail loudly instead of returning an empty frame
    return assign_splits(events, config).filter(pl.col(SPLIT_COLUMN) == name).drop(SPLIT_COLUMN)

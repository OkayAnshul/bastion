"""Shared figure styling: the reference data-viz palette on a light surface.

Series colours are categorical slots in a fixed order, checked with the dataviz palette validator.
The first three stay distinguishable when all appear together; aqua is below 3:1 contrast, so every
figure also has a table. Text uses ink tokens, never a series colour. Figures use matplotlib's
object API, so there is no pyplot global state.
"""

from __future__ import annotations

from typing import Literal

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # categorical slots 1-3: blue, orange, aqua
DEEMPHASIS = BASELINE  # the grey for "context" marks when one series is the story


def new_figure(width: float, height: float) -> Figure:
    return Figure(figsize=(width, height), dpi=150, facecolor=SURFACE, layout="constrained")


def style_axes(ax: Axes, title: str, *, grid: Literal["x", "y", "both"] = "y") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis=grid, color=GRIDLINE, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    ax.set_title(title, loc="left", color=INK_PRIMARY, fontsize=11, fontweight="bold")


def label_axes(ax: Axes, x: str, y: str) -> None:
    ax.set_xlabel(x, color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel(y, color=INK_SECONDARY, fontsize=9)


def percent_axis(ax: Axes, digits: int = 0) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.{digits}f}%"))


def count_axis(ax: Axes) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

"""Exploratory data analysis of a canonical event table (Phase 0).

Answers the Phase 0 exit questions from measured data: fraud rate, amount distribution by class,
temporal density, missingness and entity cardinality. ``bastion eda`` regenerates the markdown
report, a JSON copy of every number, and the figures. Nothing in the report is typed by hand.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.ticker import FuncFormatter, NullFormatter

from bastion.data.splits import SPLIT_COLUMN, SplitConfig, assign_splits
from bastion.plotting import (
    BASELINE,
    INK_MUTED,
    INK_SECONDARY,
    SERIES,
    SURFACE,
    count_axis,
    new_figure,
    percent_axis,
    style_axes,
)
from bastion.provenance import git_revision
from bastion.schemas.tables import ATTRIBUTE_PREFIX

LEGIT_COLOR = SERIES[0]
FRAUD_COLOR = SERIES[1]

FIGURES = ("eda_amount_by_class.png", "eda_daily.png", "eda_hourly.png")
QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


def _f(value: object) -> float:
    """Polars scalar aggregates are typed loosely; missing values become NaN."""
    return float(value) if isinstance(value, int | float) else float("nan")


# ------------------------------------------------------------------------------ statistics


def compute_eda_stats(events: pl.DataFrame, splits: SplitConfig) -> dict[str, Any]:
    """Every number the report shows, as JSON-serialisable values."""
    labelled = events.filter(pl.col("is_fraud").is_not_null())
    return {
        "overview": _overview(events),
        "class_balance": _class_balance(labelled, unlabelled=events.height - labelled.height),
        "amounts": _amounts(labelled),
        "windows": _windows(labelled, splits),
        "missingness": _missingness(events),
        "cardinality": _cardinality(events),
        "categories": _categories(labelled),
    }


def _overview(events: pl.DataFrame) -> dict[str, Any]:
    first = cast(datetime, events["event_ts"].min())
    last = cast(datetime, events["event_ts"].max())
    return {
        "transactions": events.height,
        "first_event": first.isoformat(),
        "last_event": last.isoformat(),
        "span_days": (last - first).total_seconds() / 86_400,
        "currencies": sorted(events["currency"].unique().to_list()),
        "attribute_columns": sum(c.startswith(ATTRIBUTE_PREFIX) for c in events.columns),
    }


def _class_balance(labelled: pl.DataFrame, *, unlabelled: int) -> dict[str, Any]:
    n_fraud = int(labelled["is_fraud"].sum())
    total_amount = _f(labelled["amount"].sum())
    fraud_amount = _f(labelled.filter(pl.col("is_fraud"))["amount"].sum())
    return {
        "labelled_transactions": labelled.height,
        "unlabelled_transactions": unlabelled,
        "fraud_transactions": n_fraud,
        "fraud_rate": n_fraud / labelled.height if labelled.height else float("nan"),
        "total_amount": total_amount,
        "fraud_amount": fraud_amount,
        "fraud_amount_share": fraud_amount / total_amount if total_amount else float("nan"),
    }


def _amounts(labelled: pl.DataFrame) -> dict[str, dict[str, float]]:
    result = {}
    for name, is_fraud in (("legitimate", False), ("fraud", True)):
        amount = labelled.filter(pl.col("is_fraud") == is_fraud)["amount"]
        stats = {"count": float(amount.len()), "mean": _f(amount.mean())}
        stats |= {f"p{round(q * 100):02d}": _f(amount.quantile(q, "linear")) for q in QUANTILES}
        stats["max"] = _f(amount.max())
        result[name] = stats
    return result


def _windows(labelled: pl.DataFrame, splits: SplitConfig) -> list[dict[str, Any]]:
    per_window = (
        assign_splits(labelled.select("event_ts", "amount", "is_fraud"), splits)
        .group_by(SPLIT_COLUMN)
        .agg(
            pl.len().alias("transactions"),
            pl.col("is_fraud").sum().alias("frauds"),
            pl.col("amount").filter(pl.col("is_fraud")).sum().alias("fraud_amount"),
            pl.col("event_ts").min().alias("first"),
            pl.col("event_ts").max().alias("last"),
        )
    )
    rows = {row[SPLIT_COLUMN]: row for row in per_window.iter_rows(named=True)}
    result = []
    for window in splits.windows:
        row = rows.get(window.name)
        n = int(row["transactions"]) if row else 0
        result.append(
            {
                "window": window.name,
                "start_day": window.start_day,
                "end_day": window.end_day,
                "transactions": n,
                "frauds": int(row["frauds"]) if row else 0,
                "fraud_rate": row["frauds"] / n if row and n else float("nan"),
                "fraud_amount": _f(row["fraud_amount"]) if row else 0.0,
                "first_event": row["first"].isoformat() if row else None,
                "last_event": row["last"].isoformat() if row else None,
            }
        )
    return result


def _missingness(events: pl.DataFrame) -> dict[str, Any]:
    nulls = events.null_count().row(0, named=True)
    share = {column: nulls[column] / events.height for column in events.columns}
    attributes = {
        c.removeprefix(ATTRIBUTE_PREFIX): s
        for c, s in share.items()
        if c.startswith(ATTRIBUTE_PREFIX)
    }
    buckets = {
        "complete (0%)": sum(s == 0 for s in attributes.values()),
        "up to 25%": sum(0 < s <= 0.25 for s in attributes.values()),
        "25-50%": sum(0.25 < s <= 0.5 for s in attributes.values()),
        "50-75%": sum(0.5 < s <= 0.75 for s in attributes.values()),
        "over 75%": sum(s > 0.75 for s in attributes.values()),
    }
    most_missing = sorted(attributes.items(), key=lambda item: (-item[1], item[0]))[:15]
    return {
        "canonical": {c: share[c] for c in ("device_id", "ip", "is_fraud")},
        "attribute_buckets": buckets,
        "most_missing_attributes": [{"column": c, "missing_share": s} for c, s in most_missing],
    }


def _distribution(values: pl.Series) -> dict[str, float]:
    return {
        "p50": _f(values.quantile(0.5, "nearest")),
        "p90": _f(values.quantile(0.9, "nearest")),
        "p99": _f(values.quantile(0.99, "nearest")),
        "max": _f(values.max()),
    }


def _cardinality(events: pl.DataFrame) -> dict[str, Any]:
    per_card = events.group_by("card_id").len()["len"]
    per_merchant = events.group_by("merchant_id").len()["len"]
    cards_per_device = (
        events.drop_nulls("device_id").group_by("device_id").agg(pl.col("card_id").n_unique())
    )["card_id"]
    return {
        "cards": events["card_id"].n_unique(),
        "devices": events["device_id"].drop_nulls().n_unique(),
        "merchants": events["merchant_id"].n_unique(),
        "device_coverage": 1 - events["device_id"].null_count() / events.height,
        "transactions_per_card": _distribution(per_card),
        "single_transaction_card_share": _f((per_card == 1).mean()),
        "largest_merchant_transaction_share": _f(per_merchant.max()) / events.height,
        "cards_per_device": _distribution(cards_per_device) if cards_per_device.len() else {},
        "devices_shared_by_multiple_cards_share": (
            _f((cards_per_device > 1).mean()) if cards_per_device.len() else float("nan")
        ),
    }


def _categories(labelled: pl.DataFrame) -> list[dict[str, Any]]:
    table = (
        labelled.group_by("merchant_category")
        .agg(pl.len().alias("transactions"), pl.col("is_fraud").sum().alias("frauds"))
        .with_columns(
            (pl.col("transactions") / labelled.height).alias("share"),
            (pl.col("frauds") / pl.col("transactions")).alias("fraud_rate"),
        )
        .sort("transactions", "merchant_category", descending=[True, False])
        .head(15)
    )
    return table.to_dicts()


# ------------------------------------------------------------------------------ figures


def plot_amount_by_class(labelled: pl.DataFrame, currency: str, path: Path) -> None:
    amount = labelled["amount"].to_numpy()
    fraud = labelled["is_fraud"].to_numpy().astype(bool)
    edges = np.logspace(np.log10(amount.min()), np.log10(amount.max()), 60)
    centres = np.sqrt(edges[:-1] * edges[1:])

    fig = new_figure(8, 4)
    ax = fig.add_subplot()
    for label, mask, color in (("Legitimate", ~fraud, LEGIT_COLOR), ("Fraud", fraud, FRAUD_COLOR)):
        counts, _ = np.histogram(amount[mask], bins=edges)
        share = 100 * counts / max(counts.sum(), 1)  # each class normalised to its own size
        ax.plot(centres, share, color=color, linewidth=2, solid_capstyle="round", label=label)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    percent_axis(ax)
    ax.set_xlabel(f"Transaction amount ({currency}, log scale)", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Share of the class's transactions", color=INK_SECONDARY, fontsize=9)
    ax.legend(frameon=False, loc="upper right", labelcolor=INK_SECONDARY, fontsize=9)
    style_axes(ax, "Amount distribution by class")
    fig.savefig(path)


def _daily(labelled: pl.DataFrame) -> pl.DataFrame:
    return (
        labelled.group_by(pl.col("event_ts").dt.truncate("1d").alias("day"))
        .agg(pl.len().alias("transactions"), pl.col("is_fraud").sum().alias("frauds"))
        .with_columns((pl.col("frauds") / pl.col("transactions")).alias("fraud_rate"))
        .sort("day")
    )


def plot_daily(labelled: pl.DataFrame, windows: list[dict[str, Any]], path: Path) -> None:
    daily = _daily(labelled)
    days = daily["day"].to_list()
    fig = new_figure(9, 5.5)
    volume, rate = fig.subplots(2, 1, sharex=True)
    volume.plot(days, daily["transactions"].to_numpy(), color=LEGIT_COLOR, linewidth=2)
    rate.plot(days, 100 * daily["fraud_rate"].to_numpy(), color=FRAUD_COLOR, linewidth=2)
    style_axes(volume, "Transactions per day")
    style_axes(rate, "Fraud rate per day")
    percent_axis(rate, digits=1)
    count_axis(volume)
    # Headroom above the data keeps the window names below clear of the line.
    volume.set_ylim(0, 1.3 * _f(daily["transactions"].max()))
    rate.set_ylim(bottom=0)

    # Split windows: a hairline at each start, the window name just inside it.
    for i, window in enumerate(w for w in windows if w["first_event"] is not None):
        start = datetime.fromisoformat(window["first_event"])
        for ax in (volume, rate):
            ax.axvline(start, color=BASELINE, linewidth=1, zorder=0)
        volume.text(
            start,
            0.97 - 0.1 * (i % 2),  # alternate heights so narrow windows' names stay legible
            f" {window['window']}",
            transform=volume.get_xaxis_transform(),
            color=INK_MUTED,
            fontsize=8,
            va="top",
        )
    locator = AutoDateLocator()  # type: ignore[no-untyped-call]
    rate.xaxis.set_major_locator(locator)
    rate.xaxis.set_major_formatter(ConciseDateFormatter(locator))  # type: ignore[no-untyped-call]
    fig.savefig(path)


def plot_hourly(labelled: pl.DataFrame, path: Path) -> None:
    hourly = (
        labelled.group_by(pl.col("event_ts").dt.hour().alias("hour"))
        .agg(pl.len().alias("transactions"), pl.col("is_fraud").mean().alias("fraud_rate"))
        .sort("hour")
    )
    hours = hourly["hour"].to_numpy()
    fig = new_figure(9, 5)
    volume, rate = fig.subplots(2, 1, sharex=True)
    volume.bar(hours, hourly["transactions"].to_numpy(), width=0.5, color=LEGIT_COLOR)
    rate.plot(
        hours,
        100 * hourly["fraud_rate"].to_numpy(),
        color=FRAUD_COLOR,
        linewidth=2,
        marker="o",
        markersize=6,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
    )
    style_axes(volume, "Transactions by hour of day (UTC as mapped)")
    count_axis(volume)
    style_axes(rate, "Fraud rate by hour of day")
    percent_axis(rate, digits=1)
    rate.set_ylim(bottom=0)
    rate.set_xticks(range(0, 24, 3))
    rate.set_xlabel("Hour", color=INK_SECONDARY, fontsize=9)
    fig.savefig(path)


# ------------------------------------------------------------------------------ report


def _clean(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _pct(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{100 * value:.{digits}f}%"


def _amount(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:,.2f}"


def write_eda_report(
    events: pl.DataFrame, splits: SplitConfig, out_dir: Path, *, dataset: str, source: str
) -> Path:
    """Write ``eda.md``, ``eda.json`` and ``figures/*.png`` to ``out_dir``; return the md path."""
    stats = compute_eda_stats(events, splits)
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    labelled = events.filter(pl.col("is_fraud").is_not_null())
    currency = "/".join(stats["overview"]["currencies"])
    plot_amount_by_class(labelled, currency, figures / FIGURES[0])
    plot_daily(labelled, stats["windows"], figures / FIGURES[1])
    plot_hourly(labelled, figures / FIGURES[2])

    revision = git_revision()
    payload = {"dataset": dataset, "source": source, "git_revision": revision, **stats}
    (out_dir / "eda.json").write_text(json.dumps(_clean(payload), indent=2, sort_keys=True) + "\n")

    md_path = out_dir / "eda.md"
    md_path.write_text(_markdown(stats, dataset=dataset, source=source, revision=revision))
    return md_path


def _markdown(stats: dict[str, Any], *, dataset: str, source: str, revision: str) -> str:
    ov, cb, am, card = (
        stats["overview"],
        stats["class_balance"],
        stats["amounts"],
        stats["cardinality"],
    )
    cur = "/".join(ov["currencies"])
    legit, fraud = am["legitimate"], am["fraud"]
    lines = [
        f"# Exploratory data analysis: {dataset}",
        "",
        f"Generated by `bastion eda` from `{source}` (git revision `{revision}`). Every number is",
        "computed from the data; `eda.json` holds all of them.",
        "",
        "## Phase 0 answers",
        "",
        "| Question | Answer |",
        "|---|---|",
        f"| Transactions | {ov['transactions']:,} |",
        f"| Time span | {ov['first_event'][:10]} to {ov['last_event'][:10]}"
        f" ({ov['span_days']:.1f} days) |",
        f"| Fraud rate | {_pct(cb['fraud_rate'])} ({cb['fraud_transactions']:,} of"
        f" {cb['labelled_transactions']:,} labelled) |",
        f"| Fraud share of transaction value | {_pct(cb['fraud_amount_share'])} |",
        f"| Median amount: legitimate vs fraud | {_amount(legit['p50'])} vs"
        f" {_amount(fraud['p50'])} {cur} |",
        f"| Mean amount: legitimate vs fraud | {_amount(legit['mean'])} vs"
        f" {_amount(fraud['mean'])} {cur} |",
        f"| Distinct cards / devices / merchants | {card['cards']:,} / {card['devices']:,} /"
        f" {card['merchants']:,} |",
        "",
        "## Amount distribution by class",
        "",
        "Each curve is normalised to its own class, so the shapes compare despite the imbalance.",
        "Look for amount ranges where the curves separate; an amount threshold only works there.",
        "",
        "![Amount distribution by class](figures/eda_amount_by_class.png)",
        "",
        f"| Statistic ({cur}) | Legitimate | Fraud |",
        "|---|---|---|",
        f"| Count | {legit['count']:,.0f} | {fraud['count']:,.0f} |",
        f"| Mean | {_amount(legit['mean'])} | {_amount(fraud['mean'])} |",
    ]
    for q in QUANTILES:
        key = f"p{round(q * 100):02d}"
        lines.append(f"| {key} | {_amount(legit[key])} | {_amount(fraud[key])} |")
    lines += [
        f"| Max | {_amount(legit['max'])} | {_amount(fraud['max'])} |",
        "",
        "## Temporal density and the split windows",
        "",
        "Vertical lines mark where each split window starts (`configs/splits.yaml`).",
        "Look for shifts in volume or fraud rate between windows: a model tuned on one window",
        "meets the next one unchanged.",
        "",
        "![Transactions and fraud rate per day](figures/eda_daily.png)",
        "",
        "| Window | Days | Transactions | Fraud rate | Fraud value |",
        "|---|---|---|---|---|",
    ]
    for w in stats["windows"]:
        end = "end" if w["end_day"] is None else w["end_day"]
        lines.append(
            f"| `{w['window']}` | {w['start_day']} to {end} | {w['transactions']:,}"
            f" | {_pct(w['fraud_rate'])} | {_amount(w['fraud_amount'])} {cur} |"
        )
    lines += [
        "",
        "## Hour of day",
        "",
        "For IEEE-CIS, `TransactionDT` maps to clock time with an unknown offset (ADR-011),",
        "so the shape of the daily cycle is meaningful but its hour labels are not.",
        "",
        "![Volume and fraud rate by hour](figures/eda_hourly.png)",
        "",
        "## Missingness",
        "",
        "| Canonical column | Missing |",
        "|---|---|",
    ]
    for column, share in stats["missingness"]["canonical"].items():
        lines.append(f"| `{column}` | {_pct(share)} |")
    lines += [
        "",
        f"Vendor attribute columns ({ov['attribute_columns']}) by missing share:",
        "",
        "| Missing share | Columns |",
        "|---|---|",
    ]
    for bucket, count in stats["missingness"]["attribute_buckets"].items():
        lines.append(f"| {bucket} | {count} |")
    if stats["missingness"]["most_missing_attributes"]:
        lines += ["", "Most-missing attributes:", "", "| Attribute | Missing |", "|---|---|"]
        for item in stats["missingness"]["most_missing_attributes"]:
            lines.append(f"| `{item['column']}` | {_pct(item['missing_share'])} |")
    tpc, cpd = card["transactions_per_card"], card["cards_per_device"]
    lines += [
        "",
        "## Entity cardinality",
        "",
        "For IEEE-CIS these describe ADR-011 proxy ids, not real cards, devices or merchants.",
        "A hub (one merchant or device with a large share of traffic) weakens velocity and graph",
        "features built on that entity.",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Transactions per card (p50 / p90 / p99 / max) | {tpc['p50']:,.0f} / {tpc['p90']:,.0f} /"
        f" {tpc['p99']:,.0f} / {tpc['max']:,.0f} |",
        f"| Cards with a single transaction | {_pct(card['single_transaction_card_share'])} |",
        f"| Transactions carrying a device id | {_pct(card['device_coverage'])} |",
        f"| Largest merchant's share of transactions | "
        f"{_pct(card['largest_merchant_transaction_share'])} |",
    ]
    if cpd:
        lines += [
            f"| Cards per device (p50 / p99 / max) | {cpd['p50']:,.0f} / {cpd['p99']:,.0f} /"
            f" {cpd['max']:,.0f} |",
            f"| Devices used by more than one card |"
            f" {_pct(card['devices_shared_by_multiple_cards_share'])} |",
        ]
    lines += [
        "",
        "## Merchant categories",
        "",
        "| Category | Share of transactions | Fraud rate |",
        "|---|---|---|",
    ]
    for row in stats["categories"]:
        lines.append(
            f"| `{row['merchant_category']}` | {_pct(row['share'])} | {_pct(row['fraud_rate'])} |"
        )
    return "\n".join(lines) + "\n"

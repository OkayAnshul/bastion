"""Run the rules baseline across the temporal split and write its report."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import polars as pl

from bastion.data.splits import SPLIT_COLUMN, SplitConfig, assign_splits
from bastion.evaluation.cost import Action, CostModel
from bastion.evaluation.metrics import evaluate_decisions, ranking_metrics
from bastion.features.batch import compute_features
from bastion.provenance import git_revision
from bastion.rules.baseline import (
    RULE_NAMES,
    RuleConfig,
    RuleThresholds,
    apply_rules,
    fit_thresholds,
)

EVENT_COLUMNS = ("txn_id", "event_ts", "card_id", "device_id", "amount", "is_fraud")
REPORT_STEM = "rules_baseline"


@dataclass(frozen=True)
class RulesBaselineResult:
    fit_window: str
    headline_window: str
    thresholds: RuleThresholds
    windows: dict[str, dict[str, float]]  # every non-empty split window, all rules combined
    rules: dict[str, dict[str, float]]  # each rule on its own, headline window


def score_rules(
    events: pl.DataFrame, splits: SplitConfig, config: RuleConfig, fit_window: str = "train"
) -> tuple[pl.DataFrame, RuleThresholds]:
    """Features, split labels, rule flags and the block/approve action for every event."""
    frame = events.select(EVENT_COLUMNS).join(
        compute_features(events), on="txn_id", how="left", maintain_order="left"
    )
    frame = assign_splits(frame, splits)
    train = frame.filter(pl.col(SPLIT_COLUMN) == fit_window)
    if train.is_empty():
        raise ValueError(f"fit window {fit_window!r} has no rows")
    thresholds = fit_thresholds(train, config)
    scored = apply_rules(frame, thresholds).with_columns(
        pl.when(pl.col("rules_fired") > 0)
        .then(pl.lit(Action.BLOCK.value))
        .otherwise(pl.lit(Action.APPROVE.value))
        .alias("action")
    )
    return scored, thresholds


def _decision_metrics(part: pl.DataFrame, actions: pl.Series, costs: CostModel) -> dict[str, float]:
    if part["is_fraud"].null_count():
        raise ValueError("evaluation rows must all be labelled")
    report = evaluate_decisions(
        actions.to_numpy(), part["is_fraud"].to_numpy(), part["amount"].to_numpy(), costs
    )
    return report.to_dict()


def run_rules_baseline(
    events: pl.DataFrame,
    splits: SplitConfig,
    costs: CostModel,
    config: RuleConfig,
    *,
    fit_window: str = "train",
    headline_window: str = "test",
) -> RulesBaselineResult:
    scored, thresholds = score_rules(events, splits, config, fit_window)

    windows: dict[str, dict[str, float]] = {}
    for window in splits.windows:
        part = scored.filter(pl.col(SPLIT_COLUMN) == window.name)
        if part.is_empty():
            continue
        metrics = _decision_metrics(part, part["action"], costs)
        metrics |= ranking_metrics(part["is_fraud"].to_numpy(), part["rules_fired"].to_numpy())
        windows[window.name] = metrics

    headline = scored.filter(pl.col(SPLIT_COLUMN) == headline_window)
    if headline.is_empty():
        raise ValueError(f"headline window {headline_window!r} has no rows")
    rules = {}
    for name in RULE_NAMES:
        actions = headline.select(
            pl.when(pl.col(f"rule_{name}"))
            .then(pl.lit(Action.BLOCK.value))
            .otherwise(pl.lit(Action.APPROVE.value))
        ).to_series()
        rules[name] = _decision_metrics(headline, actions, costs)

    return RulesBaselineResult(fit_window, headline_window, thresholds, windows, rules)


# ------------------------------------------------------------------------------ report


def _clean(value: Any) -> Any:
    """JSON has no NaN: undefined metrics become null."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    return value


def _pct(value: float | None) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{100 * value:.2f}%"


def _money(value: float, currency: str) -> str:
    return f"{value:,.0f} {currency}"


def _num(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value:.{digits}f}"


def write_report(
    result: RulesBaselineResult, costs: CostModel, out_dir: Path, *, dataset: str, source: str
) -> Path:
    """Write ``rules_baseline.json`` (all values) and ``rules_baseline.md``; return the md path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    revision = git_revision()
    payload = {
        "dataset": dataset,
        "source": source,
        "git_revision": revision,
        "cost_model": costs.model_dump(),
        **asdict(result),
    }
    json_path = out_dir / f"{REPORT_STEM}.json"
    json_path.write_text(json.dumps(_clean(payload), indent=2, sort_keys=True) + "\n")

    cur = costs.currency
    t = result.thresholds
    head = result.windows[result.headline_window]
    lines = [
        f"# Rules baseline: {dataset}",
        "",
        "Five hand-written rules; a transaction is **blocked if any rule fires**.",
        "This is the number every model must beat (ROADMAP Phase 0).",
        "",
        f"- Source: `{source}` · git revision `{revision}` · generated by `bastion baseline rules`",
        f"- Thresholds were fit on legitimate `{result.fit_window}` transactions only.",
        f"- Money is in {cur}, using the **assumed** costs in `configs/costs.yaml`: a false decline"
        f" costs {costs.false_decline_margin_rate:.0%} of the amount"
        f" + {costs.false_decline_fixed_cost:g}"
        f" {cur}. Rules never send a transaction to review.",
        "",
        "## Thresholds",
        "",
        "| Rule | Fires when | Threshold |",
        "|---|---|---|",
        f"| `high_amount` | amount > T | {t.high_amount:,.2f} {cur} |",
        f"| `velocity_1h` | card transactions in the previous hour > T | {t.velocity_1h:g} |",
        f"| `new_device_high_amount` | device new for this card and amount > T"
        f" | {t.new_device_amount:,.2f} {cur} |",
        f"| `amount_spike` | ≥ {t.spike_min_history} card transactions in the previous 7 days and"
        f" amount > {t.spike_multiplier:g}x their mean | n/a |",
        f"| `no_history_high_amount` | no card transactions in the previous 7 days and amount > T"
        f" | {t.no_history_amount:,.2f} {cur} |",
        "",
        f"## Headline: `{result.headline_window}` window",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Transactions | {head['n_transactions']:,.0f} |",
        f"| Fraud rate | {_pct(head['n_fraud'] / head['n_transactions'])} |",
        f"| Block rate | {_pct(head['block_rate'])} |",
        f"| Precision (blocked that were fraud) | {_pct(head['intervention_precision'])} |",
        f"| Recall (frauds blocked) | {_pct(head['fraud_recall'])} |",
        f"| Fraud value caught | {_money(head['fraud_value_caught'], cur)} of"
        f" {_money(head['fraud_value_total'], cur)} ({_pct(head['fraud_value_caught_rate'])}) |",
        f"| False-decline rate | {_pct(head['false_decline_rate'])} |",
        f"| PR-AUC (score = number of rules fired) | {_num(head['pr_auc'])} |",
        f"| Loss: missed fraud | {_money(head['loss_missed_fraud'], cur)} |",
        f"| Loss: false declines | {_money(head['loss_false_declines'], cur)} |",
        f"| **Loss: total** | **{_money(head['loss_total'], cur)}** |",
        "",
        "## All windows",
        "",
        "| Window | Transactions | Fraud rate | Block rate | Precision | Recall | Value caught"
        " | False-decline rate | PR-AUC | Total loss |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in result.windows.items():
        lines.append(
            f"| `{name}` | {m['n_transactions']:,.0f} | {_pct(m['n_fraud'] / m['n_transactions'])}"
            f" | {_pct(m['block_rate'])} | {_pct(m['intervention_precision'])}"
            f" | {_pct(m['fraud_recall'])} | {_pct(m['fraud_value_caught_rate'])}"
            f" | {_pct(m['false_decline_rate'])} | {_num(m['pr_auc'])}"
            f" | {_money(m['loss_total'], cur)} |"
        )
    lines += [
        "",
        f"## Each rule alone, `{result.headline_window}` window",
        "",
        "| Rule | Fire rate | Precision | Recall | Value caught | False-decline rate"
        " | Total loss |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in result.rules.items():
        lines.append(
            f"| `{name}` | {_pct(m['block_rate'])} | {_pct(m['intervention_precision'])}"
            f" | {_pct(m['fraud_recall'])} | {_pct(m['fraud_value_caught_rate'])}"
            f" | {_pct(m['false_decline_rate'])} | {_money(m['loss_total'], cur)} |"
        )
    md_path = out_dir / f"{REPORT_STEM}.md"
    md_path.write_text("\n".join(lines) + "\n")
    return md_path

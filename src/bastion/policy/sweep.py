"""Offline policy evaluation: the review-budget sweep and cost sensitivity (ROADMAP Phase 4).

For every daily review budget, each policy's thresholds are tuned on the calibration window without
labels, then evaluated once on the test window with its labels:

- the expected-loss policy (``bastion.policy.decide``);
- the probability-threshold baseline, tuned on the same objective and budget;
- the rules baseline: block when any Phase 0 rule fires, never review;
- approving everything: the loss with no screening at all.

Every policy gets the same overrides first. The report also states the cost assumptions, how well
the model's probabilities were calibrated on the test window, and how the operating point moves
when each cost assumption changes.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import numpy.typing as npt
import polars as pl

from bastion.data.labels import LabelDelayConfig, label_events
from bastion.data.splits import SplitConfig
from bastion.evaluation.calibration import brier_score, expected_calibration_error
from bastion.evaluation.cost import Action, CostModel
from bastion.evaluation.metrics import evaluate_decisions, ranking_metrics
from bastion.features.batch import DEFAULT_LABEL_STRENGTH, compute_features
from bastion.plotting import (
    INK_MUTED,
    SERIES,
    count_axis,
    label_axes,
    new_figure,
    percent_axis,
    style_axes,
)
from bastion.policy.config import OverrideConfig, PolicyConfig
from bastion.policy.decide import (
    NO_OVERRIDE,
    BoolArray,
    decide_by_probability,
    decide_expected_loss,
    tune_probability_thresholds,
    tune_review_threshold,
)
from bastion.policy.expected_loss import ExpectedCosts, FloatArray, StrArray, expected_costs
from bastion.policy.overrides import overrides_for_rows
from bastion.provenance import git_revision
from bastion.rules.baseline import RuleConfig
from bastion.rules.evaluate import score_rules
from bastion.training.bundle import ModelBundle
from bastion.training.dataset import build_model_frame
from bastion.training.pipeline import labels_of, split_rows
from bastion.training.tracking import configure_tracking, log_metrics, tracked_run

EXPERIMENT = "bastion-phase4-policy"
REPORT_STEM = "policy"
FIGURE = "policy_budget_curve.png"
TUNING_WINDOW = "calibration"
EVALUATION_WINDOW = "test"
EXPECTED_LOSS = "expected_loss"
PROBABILITY = "probability_thresholds"
RULES = "rules"
APPROVE_ALL = "approve_all"
POLICY_LABELS = {
    EXPECTED_LOSS: "Expected-loss policy",
    PROBABILITY: "Probability thresholds (baseline)",
    RULES: "Rules baseline (no reviews)",
    APPROVE_ALL: "Approve everything",
}
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class PolicyWindow:
    """One split window in event order, with everything a decision needs."""

    name: str
    txn_ids: list[str]
    probability: FloatArray
    amount: FloatArray
    is_fraud: BoolArray
    day: IntArray  # days since 1970-01-01 UTC: the key of the scoring service's daily budget
    forced: StrArray  # override action per row, "" where none applies
    override_rules: StrArray

    @property
    def days(self) -> int:
        return int(np.unique(self.day).size)


def policy_windows(
    events: pl.DataFrame,
    bundle: ModelBundle,
    *,
    splits: SplitConfig,
    labels: LabelDelayConfig,
    overrides: OverrideConfig,
) -> dict[str, PolicyWindow]:
    """Calibrated probabilities, amounts, labels, days and overrides for both policy windows."""
    strength = float(bundle.metadata.get("label_strength", DEFAULT_LABEL_STRENGTH))
    features = compute_features(events, label_events(events, labels), label_strength=strength)
    ids = events.select("txn_id", "card_id", "device_id", "merchant_id")
    frame = build_model_frame(events, features, splits).join(
        ids, on="txn_id", how="left", maintain_order="left"
    )
    windows: dict[str, PolicyWindow] = {}
    for name in (TUNING_WINDOW, EVALUATION_WINDOW):
        rows = split_rows(frame, name).sort("event_ts", maintain_order=True)
        forced, rules = overrides_for_rows(
            rows["card_id"].to_list(),
            rows["device_id"].to_list(),
            rows["merchant_id"].to_list(),
            rows["card_txn_count_1h"].to_list(),
            overrides,
        )
        windows[name] = PolicyWindow(
            name=name,
            txn_ids=rows["txn_id"].to_list(),
            probability=bundle.predict_proba(rows),
            amount=rows["amount"].to_numpy().astype(np.float64),
            is_fraud=labels_of(rows),
            day=rows["event_ts"].dt.epoch("d").to_numpy().astype(np.int64),
            forced=forced,
            override_rules=rules,
        )
    return windows


@dataclass(frozen=True)
class PolicyOutcome:
    policy: str
    reviews_per_day: int | None  # the budget; None for policies that never review
    parameters: dict[str, float]
    tuning_expected_loss: float | None  # the tuning objective on the calibration window
    test_expected_loss: float  # what the model expected the test decisions to cost
    test: dict[str, float]  # evaluate_decisions on the test window, with its labels
    reviews_per_day_mean: float
    reviews_per_day_max: int
    capped: int  # wanted a review after the day's budget was used
    overridden: int


@dataclass(frozen=True)
class SensitivityRow:
    variant: str
    costs: dict[str, Any]
    review_threshold: float
    test: dict[str, float]
    reviews_per_day_mean: float


@dataclass(frozen=True)
class SweepResult:
    budgets: tuple[int, ...]
    operating_budget: int
    outcomes: list[PolicyOutcome]
    sensitivity: list[SensitivityRow]
    windows: dict[str, dict[str, float]]
    test_calibration: dict[str, float]
    override_counts: dict[str, int]

    def outcome(self, policy: str, budget: int | None = None) -> PolicyOutcome:
        for outcome in self.outcomes:
            if outcome.policy == policy and outcome.reviews_per_day == budget:
                return outcome
        raise KeyError(f"no outcome for policy {policy!r} at budget {budget}")


def cost_variants(costs: CostModel) -> list[tuple[str, CostModel]]:
    """The assumed costs, then each assumption halved, doubled or moved to a plausible extreme."""

    def changed(**update: float) -> CostModel:
        return CostModel.model_validate({**costs.model_dump(), **update})

    review, margin = costs.review_cost_per_case, costs.false_decline_margin_rate
    return [
        ("Assumed costs", costs),
        ("Review cost halved", changed(review_cost_per_case=review / 2)),
        ("Review cost doubled", changed(review_cost_per_case=review * 2)),
        ("False-decline margin halved", changed(false_decline_margin_rate=margin / 2)),
        ("False-decline margin doubled", changed(false_decline_margin_rate=min(1.0, margin * 2))),
        ("Analysts stop 70% of reviewed fraud", changed(review_catch_rate=0.7)),
        ("Analysts stop all reviewed fraud", changed(review_catch_rate=1.0)),
    ]


def _expected_cost_of(actions: StrArray, costs: ExpectedCosts) -> float:
    chosen = np.select(
        [actions == Action.APPROVE.value, actions == Action.REVIEW.value],
        [costs.approve, costs.review],
        costs.block,
    )
    return float(chosen.sum())


def _outcome(
    policy: str,
    budget: int | None,
    parameters: dict[str, float],
    tuning_loss: float | None,
    actions: StrArray,
    capped: int,
    window: PolicyWindow,
    window_costs: ExpectedCosts,
    costs: CostModel,
) -> PolicyOutcome:
    reviewed_days = window.day[actions == Action.REVIEW.value]
    per_day = np.unique(reviewed_days, return_counts=True)[1]
    return PolicyOutcome(
        policy=policy,
        reviews_per_day=budget,
        parameters=parameters,
        tuning_expected_loss=tuning_loss,
        test_expected_loss=_expected_cost_of(actions, window_costs),
        test=evaluate_decisions(actions, window.is_fraud, window.amount, costs).to_dict(),
        reviews_per_day_mean=reviewed_days.size / window.days,
        reviews_per_day_max=int(per_day.max()) if per_day.size else 0,
        capped=capped,
        overridden=int((window.forced != NO_OVERRIDE).sum()),
    )


def _with_overrides(actions: npt.ArrayLike, window: PolicyWindow) -> StrArray:
    return np.where(window.forced != NO_OVERRIDE, window.forced, np.asarray(actions, dtype=np.str_))


def _window_summary(window: PolicyWindow) -> dict[str, float]:
    return {
        "rows": float(window.amount.size),
        "days": float(window.days),
        "fraud_rate": float(window.is_fraud.mean()),
        "fraud_value": float(window.amount[window.is_fraud].sum()),
        "overridden": float((window.forced != NO_OVERRIDE).sum()),
    }


def run_policy_sweep(
    events: pl.DataFrame,
    bundle: ModelBundle,
    *,
    splits: SplitConfig,
    labels: LabelDelayConfig,
    costs: CostModel,
    policy: PolicyConfig,
    rules: RuleConfig,
) -> SweepResult:
    windows = policy_windows(
        events, bundle, splits=splits, labels=labels, overrides=policy.overrides
    )
    tune, test = windows[TUNING_WINDOW], windows[EVALUATION_WINDOW]
    tune_costs = expected_costs(tune.probability, tune.amount, costs)
    test_costs = expected_costs(test.probability, test.amount, costs)
    budgets = tuple(sorted({*policy.review_budget_sweep, policy.reviews_per_day}))

    outcomes: list[PolicyOutcome] = []
    for budget in budgets:
        tuned = tune_review_threshold(
            tune_costs,
            tune.day,
            reviews_per_day=budget,
            candidates=policy.threshold_candidates,
            forced=tune.forced,
        )
        chosen = decide_expected_loss(
            test_costs,
            test.day,
            threshold=tuned.review_threshold,
            reviews_per_day=budget,
            forced=test.forced,
        )
        outcomes.append(
            _outcome(
                EXPECTED_LOSS,
                budget,
                {"review_threshold": tuned.review_threshold},
                tuned.expected_loss,
                chosen.actions,
                int(chosen.capped.sum()),
                test,
                test_costs,
                costs,
            )
        )
        base = tune_probability_thresholds(
            tune.probability, tune_costs, tune.day, reviews_per_day=budget, forced=tune.forced
        )
        chosen = decide_by_probability(
            test.probability,
            test_costs,
            test.day,
            review_at=base.review_at,
            block_at=base.block_at,
            reviews_per_day=budget,
            forced=test.forced,
        )
        outcomes.append(
            _outcome(
                PROBABILITY,
                budget,
                {"review_at": base.review_at, "block_at": base.block_at},
                base.expected_loss,
                chosen.actions,
                int(chosen.capped.sum()),
                test,
                test_costs,
                costs,
            )
        )

    scored, _ = score_rules(events, splits, rules)
    fired = (
        pl.DataFrame({"txn_id": test.txn_ids})
        .join(
            scored.select("txn_id", "rules_fired"), on="txn_id", how="left", maintain_order="left"
        )["rules_fired"]
        .fill_null(0)
        .to_numpy()
        > 0
    )
    rule_actions = _with_overrides(np.where(fired, Action.BLOCK.value, Action.APPROVE.value), test)
    outcomes.append(_outcome(RULES, None, {}, None, rule_actions, 0, test, test_costs, costs))
    approvals = _with_overrides(np.full(len(test.txn_ids), Action.APPROVE.value), test)
    outcomes.append(_outcome(APPROVE_ALL, None, {}, None, approvals, 0, test, test_costs, costs))

    sensitivity = []
    for variant, variant_costs in cost_variants(costs):
        tuned = tune_review_threshold(
            expected_costs(tune.probability, tune.amount, variant_costs),
            tune.day,
            reviews_per_day=policy.reviews_per_day,
            candidates=policy.threshold_candidates,
            forced=tune.forced,
        )
        chosen = decide_expected_loss(
            expected_costs(test.probability, test.amount, variant_costs),
            test.day,
            threshold=tuned.review_threshold,
            reviews_per_day=policy.reviews_per_day,
            forced=test.forced,
        )
        reviewed = int((chosen.actions == Action.REVIEW.value).sum())
        report = evaluate_decisions(chosen.actions, test.is_fraud, test.amount, variant_costs)
        sensitivity.append(
            SensitivityRow(
                variant,
                variant_costs.model_dump(),
                tuned.review_threshold,
                report.to_dict(),
                reviewed / test.days,
            )
        )

    fired_rules = test.override_rules[test.override_rules != NO_OVERRIDE]
    names, counts = np.unique(fired_rules, return_counts=True)
    return SweepResult(
        budgets=budgets,
        operating_budget=policy.reviews_per_day,
        outcomes=outcomes,
        sensitivity=sensitivity,
        windows={window.name: _window_summary(window) for window in windows.values()},
        test_calibration={
            "brier": brier_score(test.is_fraud, test.probability),
            "ece": expected_calibration_error(test.is_fraud, test.probability),
            "mean_probability": float(test.probability.mean()),
            **ranking_metrics(test.is_fraud, test.probability),
        },
        override_counts={str(name): int(count) for name, count in zip(names, counts, strict=True)},
    )


# ------------------------------------------------------------------------------ report


def _clean(value: Any) -> Any:
    """Strict JSON: NaN and infinities become null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    return value


def _pct(value: float | None, digits: int = 2) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{100 * value:.{digits}f}%"


def _money(value: float, currency: str) -> str:
    return f"{value:,.0f} {currency}"


def _num(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value:.{digits}f}"


def _probability(value: float) -> str:
    return "never" if math.isinf(value) else f"{value:.4f}"


def _thresholds(outcome: PolicyOutcome, currency: str) -> str:
    no_reviews = outcome.reviews_per_day == 0  # a tuned review threshold means nothing here
    if outcome.policy == EXPECTED_LOSS:
        if no_reviews:
            return "none: no review budget"
        return f"review if it saves > {outcome.parameters['review_threshold']:,.2f} {currency}"
    if outcome.policy == PROBABILITY:
        review_at, block_at = outcome.parameters["review_at"], outcome.parameters["block_at"]
        block = f"block at p ≥ {_probability(block_at)}"
        return block if no_reviews else f"review at p ≥ {_probability(review_at)}, {block}"
    return "n/a"


def _plot_budget_curve(result: SweepResult, currency: str, path: Path) -> None:
    budgets = list(result.budgets)
    positions = np.arange(len(budgets))
    fig = new_figure(7, 8.5)
    axes = fig.subplots(3, 1, sharex=True)
    panels = (
        ("fraud_value_caught_rate", "Fraud value caught (test window)", True),
        ("false_decline_rate", "False-decline rate (test window)", True),
        ("loss_total", f"Total loss, {currency} (test window, assumed costs)", False),
    )
    rules = result.outcome(RULES)
    off_scale: list[str] = []
    for ax, (metric, title, is_rate) in zip(axes, panels, strict=True):
        scale = 100.0 if is_rate else 1.0
        plotted: list[float] = []
        for policy, color in ((EXPECTED_LOSS, SERIES[0]), (PROBABILITY, SERIES[1])):
            values = [result.outcome(policy, b).test[metric] * scale for b in budgets]
            plotted += [v for v in values if math.isfinite(v)]
            ax.plot(positions, values, color=color, linewidth=2, marker="o", markersize=5,
                    label=POLICY_LABELS[policy])  # fmt: skip
        style_axes(ax, title)
        low, high = (min(plotted), max(plotted)) if plotted else (0.0, 1.0)
        span = max(high - low, 0.1 * abs(high), 1e-6)
        reference = rules.test[metric] * scale
        if low - 2 * span <= reference <= high + 2 * span:
            ax.axhline(reference, color=SERIES[2], linewidth=2, label=POLICY_LABELS[RULES])
        else:
            # Drawn to scale, a distant baseline would flatten both policy curves into one line,
            # so its value goes under the chart instead, where no data can collide with it.
            ax.plot([], [], color=SERIES[2], linewidth=2, label=POLICY_LABELS[RULES])
            shown = f"{reference:.2f}%" if is_rate else f"{reference:,.0f} {currency}"
            off_scale.append(f"{title.split(' (')[0].split(',')[0].lower()} {shown}")
        if is_rate:
            percent_axis(ax, digits=2 if high - low < 1 else 1)
        else:
            count_axis(ax)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="center right")
    axes[-1].set_xticks(positions, [str(b) for b in budgets])
    note = f"\nRules baseline, off scale above: {'; '.join(off_scale)}" if off_scale else ""
    label_axes(axes[-1], f"Review budget (human reviews per day){note}", "")
    fig.savefig(path)


def write_policy_report(
    result: SweepResult,
    costs: CostModel,
    out_dir: Path,
    *,
    dataset: str,
    source: str,
    model_version: str,
) -> Path:
    """Write ``policy.md``, ``policy.json`` and the budget curve; return the markdown path."""
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _plot_budget_curve(result, costs.currency, figures / FIGURE)
    revision = git_revision()
    payload = {
        "dataset": dataset,
        "source": source,
        "git_revision": revision,
        "model_version": model_version,
        "cost_model": costs.model_dump(),
        **asdict(result),
    }
    json_path = out_dir / f"{REPORT_STEM}.json"
    json_path.write_text(json.dumps(_clean(payload), indent=2, sort_keys=True) + "\n")

    cur = costs.currency
    budget = result.operating_budget
    main = result.outcome(EXPECTED_LOSS, budget)
    rules, approve_all = result.outcome(RULES), result.outcome(APPROVE_ALL)
    tuning, evaluation = result.windows[TUNING_WINDOW], result.windows[EVALUATION_WINDOW]
    calibration = result.test_calibration

    lines = [f"# Decision policy: {dataset}", ""]
    if "synthetic" in dataset.lower():
        lines += [
            "> **Synthetic data.** The generator's fraud patterns were written by the author and"
            " are easy to separate, so these numbers show that the policy machinery works end to"
            " end. They are not evidence of how much real fraud it would catch. IEEE-CIS results:"
            " unmeasured.",
            "",
        ]
    lines += [
        f"Generated by `bastion policy sweep` at git revision `{revision}` · model"
        f" `{model_version}` · source `{source}`.",
        "",
        "## Headline",
        "",
        f"At **{budget} reviews per day**, the expected-loss policy caught"
        f" **{_pct(main.test['fraud_value_caught_rate'])}** of fraud value in the test window at a"
        f" **{_pct(main.test['false_decline_rate'])}** false-decline rate. It used"
        f" {main.reviews_per_day_mean:,.1f} reviews per day on average (at most"
        f" {main.reviews_per_day_max}). Total loss was {_money(main.test['loss_total'], cur)},"
        f" against {_money(rules.test['loss_total'], cur)} for the rules baseline and"
        f" {_money(approve_all.test['loss_total'], cur)} when approving everything.",
        "",
        f"![Fraud value caught, false-decline rate and loss by review budget](figures/{FIGURE})",
        "",
        "## Setup",
        "",
        f"- **Tuning:** the `{TUNING_WINDOW}` window, {tuning['rows']:,.0f} transactions over"
        f" {tuning['days']:.0f} days. Thresholds minimise the model's expected loss, so tuning uses"
        " no labels.",
        f"- **Evaluation:** the `{EVALUATION_WINDOW}` window, {evaluation['rows']:,.0f}"
        f" transactions over {evaluation['days']:.0f} days, with fraud rate"
        f" {_pct(evaluation['fraud_rate'])} and fraud value"
        f" {_money(evaluation['fraud_value'], cur)}. Evaluated once, with labels.",
        "- **Budget:** at most N reviews per UTC day, granted in event order, as the scoring"
        " service grants them. A transaction that wants a review after the day's budget is used"
        " gets its cheaper other action.",
        f"- **Assumed costs** (`configs/costs.yaml`, not measurements): a missed fraud costs its"
        f" amount; a false decline costs {costs.false_decline_margin_rate:.0%} of the amount plus"
        f" {costs.false_decline_fixed_cost:g} {cur}; a review costs"
        f" {costs.review_cost_per_case:g} {cur}, and analysts stop"
        f" {costs.review_catch_rate:.0%} of the fraud they review.",
        "- **Overrides** (`configs/policy.yaml`) run before every policy, including the baselines.",
        "",
        "## Model calibration on the test window",
        "",
        "The policy multiplies probabilities by amounts, so miscalibration becomes mispriced"
        " decisions (ADR-006).",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| PR-AUC | {_num(calibration['pr_auc'])} |",
        f"| Brier score | {_num(calibration['brier'], 5)} |",
        f"| Expected calibration error | {_num(calibration['ece'], 5)} |",
        f"| Mean predicted probability | {_pct(calibration['mean_probability'], 3)} |",
        f"| Observed fraud rate | {_pct(calibration['positive_rate'], 3)} |",
        "",
        "## By review budget",
        "",
        "Loss columns are realized on the test window with its labels. *Model-expected loss* is"
        " what the model's probabilities predicted the same decisions would cost.",
        "",
        "| Budget / day | Policy | Thresholds | Reviews/day: mean, max | Fraud value caught"
        " | False-decline rate | Missed fraud | False declines | Review cost | Total loss"
        " | Model-expected loss |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for o in result.outcomes:
        t = o.test
        lines.append(
            f"| {'n/a' if o.reviews_per_day is None else o.reviews_per_day}"
            f" | {POLICY_LABELS[o.policy]} | {_thresholds(o, cur)}"
            f" | {o.reviews_per_day_mean:,.1f}, {o.reviews_per_day_max}"
            f" | {_pct(t['fraud_value_caught_rate'])} | {_pct(t['false_decline_rate'])}"
            f" | {_money(t['loss_missed_fraud'], cur)} | {_money(t['loss_false_declines'], cur)}"
            f" | {_money(t['loss_review_cost'], cur)} | **{_money(t['loss_total'], cur)}**"
            f" | {_money(o.test_expected_loss, cur)} |"
        )
    lines += ["", "## Overrides on the test window", ""]
    if result.override_counts:
        lines += ["| Rule | Transactions |", "|---|---|"]
        lines += [f"| `{rule}` | {count:,} |" for rule, count in result.override_counts.items()]
    else:
        lines.append("No override fired.")
    lines += [
        "",
        "## Sensitivity to the cost assumptions",
        "",
        f"At {budget} reviews per day, with the threshold re-tuned under each set of costs. Loss is"
        " measured under each row's own costs, so compare rows by value caught and false-decline"
        " rate rather than by loss.",
        "",
        "| Costs | Review threshold | Reviews/day | Fraud value caught | False-decline rate"
        " | Total loss |",
        "|---|---|---|---|---|---|",
    ]
    for row in result.sensitivity:
        lines.append(
            f"| {row.variant} | {row.review_threshold:,.2f} {cur}"
            f" | {row.reviews_per_day_mean:,.1f} | {_pct(row.test['fraud_value_caught_rate'])}"
            f" | {_pct(row.test['false_decline_rate'])} | {_money(row.test['loss_total'], cur)} |"
        )
    lines += [
        "",
        "## Limits",
        "",
        "- The costs are assumptions: IEEE-CIS carries no merchant economics.",
        "- Analysts are assumed to clear every legitimate transaction they review and to stop a"
        " fixed share of the fraud they review.",
        "- One tuning window and one test window, so these numbers carry no confidence interval.",
        "- A day's budget is spent in event order. A real team can also defer a case to the next"
        " day, which is not modelled.",
    ]
    path = out_dir / f"{REPORT_STEM}.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def log_policy_sweep(
    result: SweepResult,
    report_path: Path,
    *,
    tracking_uri: str,
    artifacts_dir: Path,
    tags: Mapping[str, str],
) -> str:
    """Log every outcome's test metrics and the report to MLflow; return the run id."""
    configure_tracking(tracking_uri)
    run_name = f"policy-sweep-{tags.get('dataset', 'unknown')}"
    with tracked_run(EXPERIMENT, run_name, tags, artifact_root=artifacts_dir / "mlflow") as run_id:
        mlflow.log_params(
            {
                "operating_budget": result.operating_budget,
                "budgets": ",".join(str(b) for b in result.budgets),
            }
        )
        for o in result.outcomes:
            prefix = o.policy if o.reviews_per_day is None else f"{o.policy}_n{o.reviews_per_day}"
            values = {
                **o.test,
                **o.parameters,
                "reviews_per_day_mean": o.reviews_per_day_mean,
                "test_expected_loss": o.test_expected_loss,
            }
            log_metrics(values, prefix=f"{prefix}_")
        log_metrics(result.test_calibration, prefix="test_calibration_")
        figure = report_path.parent / "figures" / FIGURE
        for path in (report_path, report_path.with_suffix(".json"), figure):
            mlflow.log_artifact(str(path), "report")
    return run_id

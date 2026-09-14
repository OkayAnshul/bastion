"""Phase 1 training run: point-in-time features, baselines, LightGBM, calibration, MLflow.

A run leaves behind, for the test window: metrics for the rules baseline, logistic regression and
LightGBM (raw, isotonic, Platt); precision-recall and reliability figures; a model bundle; a
markdown/JSON report; and an MLflow run holding all of it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.metrics import precision_recall_curve

from bastion.data.labels import LabelDelayConfig, label_events
from bastion.data.splits import SPLIT_COLUMN, SplitConfig
from bastion.evaluation.calibration import (
    brier_score,
    expected_calibration_error,
    fit_calibrator,
    log_loss,
    reliability_bins,
)
from bastion.evaluation.metrics import ranking_metrics
from bastion.features.batch import compute_features, feature_names
from bastion.plotting import (
    BASELINE,
    INK_MUTED,
    SERIES,
    SURFACE,
    label_axes,
    new_figure,
    style_axes,
)
from bastion.provenance import git_revision
from bastion.rules.baseline import RuleConfig
from bastion.rules.evaluate import score_rules
from bastion.training.bundle import ModelBundle
from bastion.training.dataset import ModelConfig, build_model_frame, check_label_maturity, to_matrix
from bastion.training.models import LogisticBaseline
from bastion.training.pipeline import (
    FittedModel,
    bootstrap_pr_auc,
    fit_lightgbm_on_frame,
    labels_of,
    split_rows,
)
from bastion.training.tracking import configure_tracking, data_fingerprint, log_metrics, tracked_run

EXPERIMENT = "bastion-phase1-model"
REPORT_STEM = "model"
FIGURES = ("model_precision_recall.png", "model_reliability.png")
SCORER_LABELS = {
    "rules": "Rules baseline (rules fired)",
    "logistic": "Logistic regression",
    "lightgbm_raw": "LightGBM, uncalibrated",
    "lightgbm_isotonic": "LightGBM + isotonic",
    "lightgbm_platt": "LightGBM + Platt",
}
PROBABILITY_SCORERS = ("logistic", "lightgbm_raw", "lightgbm_isotonic", "lightgbm_platt")


@dataclass(frozen=True)
class TrainingInputs:
    events: pl.DataFrame
    splits: SplitConfig
    labels: LabelDelayConfig
    model: ModelConfig
    rules: RuleConfig
    dataset: str
    source: str


@dataclass(frozen=True)
class TrainingResult:
    run_id: str
    report_path: Path
    bundle_dir: Path
    metrics: dict[str, dict[str, float]]


def train_model(
    inputs: TrainingInputs, *, results_dir: Path, artifacts_dir: Path, tracking_uri: str
) -> TrainingResult:
    cfg = inputs.model
    labels = label_events(inputs.events, inputs.labels)
    features = compute_features(inputs.events, labels, label_strength=cfg.label_strength)
    frame = build_model_frame(inputs.events, features, inputs.splits)
    check_label_maturity(frame, labels, train_window="train", next_window="early_stopping")

    fitted = fit_lightgbm_on_frame(frame, feature_names(with_labels=True), cfg)
    train, calibration, test = (split_rows(frame, w) for w in ("train", "calibration", "test"))
    y_test = labels_of(test)
    raw_calibration, raw_test = fitted.scores(calibration), fitted.scores(test)
    calibrators = {
        method: fit_calibrator(method, raw_calibration, labels_of(calibration))
        for method in ("isotonic", "platt")
    }
    logistic = LogisticBaseline(len(fitted.spec.numeric), cfg.seed).fit(
        to_matrix(train, fitted.spec), labels_of(train)
    )
    scored_rules, _ = score_rules(inputs.events, inputs.splits, inputs.rules)
    rules_fired = test.select("txn_id").join(
        scored_rules.select("txn_id", "rules_fired"), on="txn_id", how="left", maintain_order="left"
    )["rules_fired"]

    scores = {
        "rules": rules_fired.to_numpy().astype(np.float64),
        "logistic": logistic.predict(to_matrix(test, fitted.spec)),
        "lightgbm_raw": raw_test,
        "lightgbm_isotonic": calibrators["isotonic"].predict(raw_test),
        "lightgbm_platt": calibrators["platt"].predict(raw_test),
    }
    metrics = _metrics(y_test, scores, cfg)

    revision = git_revision()
    tags = {
        "dataset": inputs.dataset,
        "git_revision": revision,
        "data_fingerprint": data_fingerprint(inputs.events),
        "feature_mode": "point_in_time",
        "spec_fingerprint": fitted.spec.fingerprint(),
    }
    configure_tracking(tracking_uri)
    with tracked_run(
        EXPERIMENT, f"lightgbm-{inputs.dataset}", tags, artifact_root=artifacts_dir / "mlflow"
    ) as run_id:
        mlflow.log_params(_params(inputs, fitted))
        for name, values in metrics.items():
            log_metrics(values, prefix=f"test_{name}_")

        bundle = ModelBundle.from_booster(
            fitted.booster,
            calibrators[cfg.calibration],
            fitted.spec,
            {
                **tags,
                "mlflow_run_id": run_id,
                "calibration": cfg.calibration,
                "best_iteration": fitted.booster.best_iteration,
                "label_strength": cfg.label_strength,
                "with_label_features": True,
            },
        )
        bundle_dir = bundle.save(artifacts_dir / "models" / run_id)

        figures = results_dir / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        _plot_precision_recall(y_test, scores, figures / FIGURES[0])
        _plot_reliability(y_test, scores, figures / FIGURES[1])
        report = _write_report(
            inputs, frame, fitted, metrics, results_dir, run_id=run_id, revision=revision
        )

        mlflow.log_artifacts(str(bundle_dir), "bundle")
        for path in (report, report.with_suffix(".json"), *(figures / f for f in FIGURES)):
            mlflow.log_artifact(str(path), "report")
    return TrainingResult(run_id, report, bundle_dir, metrics)


def _metrics(
    y: npt.NDArray[np.bool_], scores: dict[str, npt.NDArray[np.float64]], cfg: ModelConfig
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name, s in scores.items():
        values = ranking_metrics(y, s)
        low, high = bootstrap_pr_auc(y, s, resamples=cfg.bootstrap_resamples, seed=cfg.seed)
        values |= {"pr_auc_ci_low": low, "pr_auc_ci_high": high}
        if name in PROBABILITY_SCORERS:  # the rules score is a count, not a probability
            values |= {
                "brier": brier_score(y, s),
                "log_loss": log_loss(y, s),
                "ece": expected_calibration_error(y, s),
            }
        result[name] = values
    return result


def _params(inputs: TrainingInputs, fitted: FittedModel) -> dict[str, Any]:
    cfg = inputs.model
    return {
        **{f"lgbm_{k}": v for k, v in cfg.lightgbm.items()},
        "seed": cfg.seed,
        "best_iteration": fitted.booster.best_iteration,
        "num_boost_round": cfg.num_boost_round,
        "early_stopping_rounds": cfg.early_stopping_rounds,
        "calibration": cfg.calibration,
        "label_strength": cfg.label_strength,
        "label_maturity_days": inputs.labels.maturity_days,
        "fraud_label_median_days": inputs.labels.fraud_median_days,
        "numeric_inputs": len(fitted.spec.numeric),
        "categorical_inputs": len(fitted.spec.categorical),
    }


# ------------------------------------------------------------------------------ figures


def _plot_precision_recall(
    y: npt.NDArray[np.bool_], scores: dict[str, npt.NDArray[np.float64]], path: Path
) -> None:
    fig = new_figure(7, 5)
    ax = fig.add_subplot()
    for name, color in (
        ("lightgbm_isotonic", SERIES[0]),
        ("logistic", SERIES[2]),
        ("rules", SERIES[1]),
    ):
        precision, recall, _ = precision_recall_curve(y, scores[name])
        ax.plot(recall, precision, color=color, linewidth=2, label=SCORER_LABELS[name])
    positive_rate = float(y.mean())
    ax.axhline(positive_rate, color=BASELINE, linewidth=1)
    ax.text(0.99, positive_rate, "no skill ", color=INK_MUTED, fontsize=8, ha="right", va="bottom")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    label_axes(ax, "Recall", "Precision")
    ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=INK_MUTED)
    style_axes(ax, "Precision-recall, test window", grid="both")
    fig.savefig(path)


def _plot_reliability(
    y: npt.NDArray[np.bool_], scores: dict[str, npt.NDArray[np.float64]], path: Path
) -> None:
    floor = 1e-4  # log axes: bins with a zero observed rate are drawn on the floor
    fig = new_figure(6.5, 5.5)
    ax = fig.add_subplot()
    ax.plot([floor, 1], [floor, 1], color=BASELINE, linewidth=1)
    series = (
        ("lightgbm_isotonic", SERIES[0]),
        ("lightgbm_raw", SERIES[1]),
        ("lightgbm_platt", SERIES[2]),
    )
    for name, color in series:
        bins = reliability_bins(y, scores[name], n_bins=15)
        ax.plot(
            [max(b.mean_predicted, floor) for b in bins],
            [max(b.observed_rate, floor) for b in bins],
            color=color,
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            label=SCORER_LABELS[name],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(floor, 1)
    ax.set_ylim(floor, 1)
    label_axes(ax, "Mean predicted probability", "Observed fraud rate")
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK_MUTED)
    style_axes(ax, "Reliability, test window (equal-mass bins)", grid="both")
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


def _num(value: float, digits: int = 4) -> str:
    return "n/a" if math.isnan(value) else f"{value:.{digits}f}"


def _write_report(
    inputs: TrainingInputs,
    frame: pl.DataFrame,
    fitted: FittedModel,
    metrics: dict[str, dict[str, float]],
    results_dir: Path,
    *,
    run_id: str,
    revision: str,
) -> Path:
    windows = (
        frame.group_by(SPLIT_COLUMN, maintain_order=True)
        .agg(pl.len().alias("rows"), pl.col("is_fraud").mean().alias("fraud_rate"))
        .to_dicts()
    )
    gain = fitted.booster.feature_importance(
        importance_type="gain", iteration=fitted.booster.best_iteration
    )
    total_gain = float(gain.sum()) or 1.0
    importance = sorted(
        zip(fitted.booster.feature_name(), (float(g) / total_gain for g in gain), strict=True),
        key=lambda item: -item[1],
    )[:20]

    payload = {
        "dataset": inputs.dataset,
        "source": inputs.source,
        "git_revision": revision,
        "mlflow_run_id": run_id,
        "spec_fingerprint": fitted.spec.fingerprint(),
        "best_iteration": fitted.booster.best_iteration,
        "windows": windows,
        "metrics": metrics,
        "top_features_by_gain": [{"feature": f, "gain_share": g} for f, g in importance],
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"{REPORT_STEM}.json"
    json_path.write_text(json.dumps(_clean(payload), indent=2, default=str) + "\n")

    lines = [
        f"# Model: {inputs.dataset}",
        "",
        f"Generated by `bastion train` from `{inputs.source}` · git revision `{revision}` ·",
        f"MLflow run `{run_id}` · feature spec `{fitted.spec.fingerprint()}`.",
        "",
        "## Setup",
        "",
        f"- LightGBM on point-in-time features (ADR-003), early-stopped on `early_stopping`: best"
        f" iteration {fitted.booster.best_iteration} of at most {inputs.model.num_boost_round}.",
        f"- {len(fitted.spec.numeric)} numeric and {len(fitted.spec.categorical)} categorical "
        "inputs;"
        " vocabularies fit on `train` only.",
        "- Labels arrive with a simulated delay (median fraud delay "
        f"{inputs.labels.fraud_median_days:g}"
        f" days, maturity {inputs.labels.maturity_days:g} days). Checked: every training label "
        "arrived"
        " before `early_stopping` began.",
        "- Calibrators fit on `calibration`; the policy engine uses "
        f"**{inputs.model.calibration}**,"
        " chosen before looking at test results.",
        "",
        "| Window | Rows | Fraud rate |",
        "|---|---|---|",
    ]
    for w in windows:
        lines.append(f"| `{w[SPLIT_COLUMN]}` | {w['rows']:,} | {100 * w['fraud_rate']:.2f}% |")
    lines += [
        "",
        "## Test-window results",
        "",
        "No-skill PR-AUC equals the positive rate: "
        f"{_num(metrics['lightgbm_raw']['positive_rate'])}.",
        "",
        "| Scorer | PR-AUC [95% CI] | ROC-AUC | Brier | Log loss | ECE |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in metrics.items():
        ci = f"[{_num(m['pr_auc_ci_low'], 3)}, {_num(m['pr_auc_ci_high'], 3)}]"
        calibration_cells = (
            f"{_num(m['brier'])} | {_num(m['log_loss'])} | {_num(m['ece'])}"
            if name in PROBABILITY_SCORERS
            else "n/a | n/a | n/a"
        )
        lines.append(
            f"| {SCORER_LABELS[name]} | {_num(m['pr_auc'])} {ci} | {_num(m['roc_auc'])} |"
            f" {calibration_cells} |"
        )
    lines += [
        "",
        f"![Precision-recall curves](figures/{FIGURES[0]})",
        "",
        f"![Reliability curves](figures/{FIGURES[1]})",
        "",
        "Ranking metrics say how well fraud is ordered; calibration metrics say whether the"
        " probabilities can be multiplied by amounts. Monetary results come with the policy engine"
        " (Phase 4).",
        "",
        "## Top features by gain",
        "",
        "| Feature | Share of total gain |",
        "|---|---|",
    ]
    lines += [f"| `{feature}` | {100 * share:.1f}% |" for feature, share in importance]
    md_path = results_dir / f"{REPORT_STEM}.md"
    md_path.write_text("\n".join(lines) + "\n")
    return md_path

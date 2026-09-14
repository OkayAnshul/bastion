"""Leakage experiment (ADR-003, ADR-007): how much of an offline score is an artefact?

One LightGBM configuration, five ways of preparing the data. Only ``pit_temporal`` is an honest
estimate of performance on future transactions. The others show what the two classic mistakes do to
the reported number: features computed with future information, and rows shuffled across time. They
also show what a leaky model actually scores once it meets features computed the way production
would compute them.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import mlflow
import numpy as np
import polars as pl

from bastion.data.labels import LabelDelayConfig, label_events
from bastion.data.splits import SPLIT_COLUMN, SplitConfig
from bastion.evaluation.metrics import ranking_metrics
from bastion.features.batch import compute_features, feature_names
from bastion.features.naive import compute_naive_features
from bastion.plotting import DEEMPHASIS, INK_MUTED, INK_SECONDARY, SERIES, new_figure, style_axes
from bastion.provenance import git_revision
from bastion.training.dataset import ModelConfig, build_model_frame
from bastion.training.pipeline import (
    FittedModel,
    bootstrap_pr_auc,
    fit_lightgbm_on_frame,
    labels_of,
    split_rows,
)
from bastion.training.tracking import configure_tracking, data_fingerprint, log_metrics, tracked_run

EXPERIMENT = "bastion-phase1-leakage"
REPORT_STEM = "leakage"
FIGURE = "leakage_pr_auc.png"
FeatureMode = Literal["naive", "point_in_time"]
SplitMode = Literal["random", "temporal"]


@dataclass(frozen=True)
class Variant:
    name: str
    train_features: FeatureMode
    split: SplitMode
    test_features: FeatureMode
    meaning: str


VARIANTS = (
    Variant("naive_random", "naive", "random", "naive", "leaky features and a shuffled split"),
    Variant("naive_temporal", "naive", "temporal", "naive", "leaky features, temporal split"),
    Variant(
        "naive_in_production",
        "naive",
        "temporal",
        "point_in_time",
        "the leaky model scored on features computed as production would",
    ),
    Variant(
        "pit_random", "point_in_time", "random", "point_in_time", "honest features, shuffled split"
    ),
    Variant(
        "pit_temporal",
        "point_in_time",
        "temporal",
        "point_in_time",
        "honest features, temporal split: the number to believe",
    ),
)
HONEST = "pit_temporal"


@dataclass(frozen=True)
class VariantResult:
    name: str
    train_features: str
    split: str
    test_features: str
    meaning: str
    test_rows: int
    positive_rate: float
    pr_auc: float
    pr_auc_ci_low: float
    pr_auc_ci_high: float
    roc_auc: float
    best_iteration: int


def shuffled_split(frame: pl.DataFrame, seed: int) -> pl.DataFrame:
    """DELIBERATE ANTI-PATTERN (ADR-007), used only to measure its effect.

    Reassigns the temporal split labels to random rows while keeping every window's size. Training
    rows then sit beside test rows in time, and one card's history spans both sides.
    """
    labels = frame[SPLIT_COLUMN].to_numpy()
    permutation = np.random.default_rng(seed).permutation(labels.size)
    return frame.with_columns(pl.Series(SPLIT_COLUMN, labels[permutation]))


def run_leakage_experiment(
    events: pl.DataFrame,
    splits: SplitConfig,
    labels_config: LabelDelayConfig,
    model_config: ModelConfig,
    *,
    dataset: str,
    source: str,
    results_dir: Path,
    artifacts_dir: Path,
    tracking_uri: str,
) -> list[VariantResult]:
    labels = label_events(events, labels_config)
    pit = compute_features(events, labels, label_strength=model_config.label_strength)
    temporal: dict[FeatureMode, pl.DataFrame] = {
        "point_in_time": build_model_frame(events, pit, splits),
        "naive": build_model_frame(
            events, compute_naive_features(events, with_labels=True), splits
        ),
    }
    # One shuffled assignment for both feature modes, so random variants differ only in features.
    shuffled = shuffled_split(temporal["point_in_time"].select(SPLIT_COLUMN), model_config.seed)
    frames: dict[tuple[SplitMode, FeatureMode], pl.DataFrame] = {}
    for mode, frame in temporal.items():
        frames[("temporal", mode)] = frame
        frames[("random", mode)] = frame.with_columns(shuffled[SPLIT_COLUMN])

    revision = git_revision()
    tags = {
        "dataset": dataset,
        "git_revision": revision,
        "data_fingerprint": data_fingerprint(events),
    }
    configure_tracking(tracking_uri)
    fitted: dict[tuple[SplitMode, FeatureMode], FittedModel] = {}
    results: list[VariantResult] = []
    with tracked_run(
        EXPERIMENT, f"leakage-{dataset}", tags, artifact_root=artifacts_dir / "mlflow"
    ):
        for variant in VARIANTS:
            key = (variant.split, variant.train_features)
            if key not in fitted:  # naive_in_production reuses the naive_temporal model
                fitted[key] = fit_lightgbm_on_frame(
                    frames[key], feature_names(with_labels=True), model_config
                )
            model = fitted[key]
            test = split_rows(frames[(variant.split, variant.test_features)], "test")
            y = labels_of(test)
            scores = model.scores(test)
            ranking = ranking_metrics(y, scores)
            low, high = bootstrap_pr_auc(
                y, scores, resamples=model_config.bootstrap_resamples, seed=model_config.seed
            )
            result = VariantResult(
                name=variant.name,
                train_features=variant.train_features,
                split=variant.split,
                test_features=variant.test_features,
                meaning=variant.meaning,
                test_rows=test.height,
                positive_rate=ranking["positive_rate"],
                pr_auc=ranking["pr_auc"],
                pr_auc_ci_low=low,
                pr_auc_ci_high=high,
                roc_auc=ranking["roc_auc"],
                best_iteration=int(model.booster.best_iteration),
            )
            results.append(result)
            with tracked_run(
                EXPERIMENT, variant.name, {**tags, "variant": variant.name}, nested=True
            ):
                mlflow.log_params(
                    {
                        "train_features": variant.train_features,
                        "split": variant.split,
                        "test_features": variant.test_features,
                        "best_iteration": result.best_iteration,
                    }
                )
                log_metrics(
                    {
                        "test_pr_auc": result.pr_auc,
                        "test_pr_auc_ci_low": low,
                        "test_pr_auc_ci_high": high,
                        "test_roc_auc": result.roc_auc,
                        "test_positive_rate": result.positive_rate,
                    }
                )
        report = write_leakage_report(
            results, results_dir, dataset=dataset, source=source, revision=revision
        )
        for path in (report, report.with_suffix(".json"), results_dir / "figures" / FIGURE):
            mlflow.log_artifact(str(path), "report")
    return results


# ------------------------------------------------------------------------------ report


def _num(value: float, digits: int = 4) -> str:
    return "n/a" if math.isnan(value) else f"{value:.{digits}f}"


def _plot(results: list[VariantResult], path: Path) -> None:
    ordered = list(reversed(results))  # first variant at the top
    fig = new_figure(8, 3.8)
    ax = fig.add_subplot()
    for y, r in enumerate(ordered):
        ax.barh(y, r.pr_auc, height=0.45, color=SERIES[0] if r.name == HONEST else DEEMPHASIS)
        tip = r.pr_auc
        if not math.isnan(r.pr_auc_ci_low):
            ax.plot([r.pr_auc_ci_low, r.pr_auc_ci_high], [y, y], color=INK_SECONDARY, linewidth=1)
            tip = max(tip, r.pr_auc_ci_high)
        ax.text(tip, y, f"  {r.pr_auc:.3f}", va="center", color=INK_SECONDARY, fontsize=9)
    honest = next(r for r in results if r.name == HONEST)
    ax.axvline(honest.positive_rate, color=INK_MUTED, linewidth=1)
    ax.set_yticks(range(len(ordered)), [r.name for r in ordered])
    finite = [max(r.pr_auc, r.pr_auc_ci_high) for r in results if not math.isnan(r.pr_auc)]
    ax.set_xlim(0, min(1.0, 1.15 * max(finite, default=1.0)) or 1.0)
    ax.set_xlabel(
        "Test PR-AUC (bars) with 95% bootstrap interval; vertical line = no skill",
        color=INK_MUTED,
        fontsize=8,
    )
    style_axes(ax, "Same model, five data preparations", grid="x")
    fig.savefig(path)


def write_leakage_report(
    results: list[VariantResult], results_dir: Path, *, dataset: str, source: str, revision: str
) -> Path:
    by_name = {r.name: r for r in results}
    figures = results_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _plot(results, figures / FIGURE)

    differences = {
        "leaky_features": by_name["naive_temporal"].pr_auc - by_name[HONEST].pr_auc,
        "shuffled_split": by_name["pit_random"].pr_auc - by_name[HONEST].pr_auc,
        "both_mistakes": by_name["naive_random"].pr_auc - by_name[HONEST].pr_auc,
        "offline_claim_vs_production": by_name["naive_temporal"].pr_auc
        - by_name["naive_in_production"].pr_auc,
    }
    payload = {
        "dataset": dataset,
        "source": source,
        "git_revision": revision,
        "variants": [asdict(r) for r in results],
        "pr_auc_differences": differences,
    }
    (results_dir / f"{REPORT_STEM}.json").write_text(
        json.dumps(payload, indent=2, default=lambda v: None) + "\n"
    )

    lines = [
        f"# Leakage experiment: {dataset}",
        "",
        f"Generated by `bastion experiment leakage` from `{source}` · git revision `{revision}`.",
        "One LightGBM configuration (`configs/model.yaml`), five ways of preparing the data. Only",
        f"`{HONEST}` estimates performance on future transactions.",
        "",
        "| Variant | Train features | Split | Test features | Test PR-AUC [95% CI] | ROC-AUC | "
        "Test rows |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        name = f"**`{r.name}`**" if r.name == HONEST else f"`{r.name}`"
        lines.append(
            f"| {name} | {r.train_features} | {r.split} | {r.test_features} |"
            f" {_num(r.pr_auc)} [{_num(r.pr_auc_ci_low, 3)}, {_num(r.pr_auc_ci_high, 3)}] |"
            f" {_num(r.roc_auc)} | {r.test_rows:,} |"
        )
    lines += [
        "",
        f"![Test PR-AUC by variant](figures/{FIGURE})",
        "",
        "## Measured differences in test PR-AUC",
        "",
        "Positive means the mistaken setup reports a higher number than the honest one.",
        "",
        "| Comparison | Difference |",
        "|---|---|",
        f"| Leaky features (`naive_temporal` - `{HONEST}`) | "
        f"{_num(differences['leaky_features'])} |",
        f"| Shuffled split (`pit_random` - `{HONEST}`) | {_num(differences['shuffled_split'])} |",
        f"| Both mistakes (`naive_random` - `{HONEST}`) | {_num(differences['both_mistakes'])} |",
        "| Leaky model: offline claim - production reality (`naive_temporal` - "
        "`naive_in_production`) |"
        f" {_num(differences['offline_claim_vs_production'])} |",
        "",
        "## What each variant means",
        "",
    ]
    lines += [f"- `{r.name}`: {r.meaning}." for r in results]
    path = results_dir / f"{REPORT_STEM}.md"
    path.write_text("\n".join(lines) + "\n")
    return path

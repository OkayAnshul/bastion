import math
from pathlib import Path

import polars as pl
import pytest

from bastion.data.splits import SPLIT_COLUMN
from bastion.training.experiments.leakage import (
    FIGURE,
    HONEST,
    VARIANTS,
    VariantResult,
    run_leakage_experiment,
    shuffled_split,
)
from tests.unit.test_training import LABELS, fast_model_config, small_splits, synthetic_events


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> tuple[list[VariantResult], Path]:
    root = tmp_path_factory.mktemp("leakage")
    variants = run_leakage_experiment(
        synthetic_events(),
        small_splits(),
        LABELS,
        fast_model_config(),
        dataset="synthetic",
        source="unit test",
        results_dir=root / "results",
        artifacts_dir=root / "artifacts",
        tracking_uri=f"sqlite:///{root}/mlflow.db",
    )
    return variants, root / "results"


def test_every_variant_is_measured_and_reported(results: tuple[list[VariantResult], Path]) -> None:
    variants, results_dir = results
    assert [v.name for v in variants] == [v.name for v in VARIANTS]
    assert all(math.isfinite(v.pr_auc) for v in variants)
    report = (results_dir / "leakage.md").read_text()
    assert f"**`{HONEST}`**" in report
    assert (results_dir / "figures" / FIGURE).stat().st_size > 0


def test_production_variant_reuses_the_leaky_model(
    results: tuple[list[VariantResult], Path],
) -> None:
    by_name = {v.name: v for v in results[0]}
    assert by_name["naive_in_production"].best_iteration == by_name["naive_temporal"].best_iteration
    assert by_name["naive_in_production"].test_rows == by_name["naive_temporal"].test_rows


def test_shuffled_split_keeps_window_sizes_but_moves_rows() -> None:
    frame = pl.DataFrame({SPLIT_COLUMN: ["train"] * 60 + ["test"] * 40})
    shuffled = shuffled_split(frame, seed=3)
    assert (
        shuffled[SPLIT_COLUMN]
        .value_counts()
        .sort(SPLIT_COLUMN)
        .equals(frame[SPLIT_COLUMN].value_counts().sort(SPLIT_COLUMN))
    )
    assert not shuffled.equals(frame)

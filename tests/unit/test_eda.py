import json
from pathlib import Path

import polars as pl
import pytest

from bastion.data.splits import SplitConfig, SplitWindow
from bastion.eda import FIGURES, compute_eda_stats, write_eda_report
from bastion.streaming.synthetic import SyntheticConfig, generate

SPLITS = SplitConfig(
    windows=(
        SplitWindow(name="train", start_day=0, end_day=14),
        SplitWindow(name="test", start_day=14, end_day=None),
    )
)


@pytest.fixture(scope="module")
def events() -> pl.DataFrame:
    return generate(SyntheticConfig(days=21, n_cards=200, n_merchants=40))


def test_statistics_agree_with_direct_computation(events: pl.DataFrame) -> None:
    stats = compute_eda_stats(events, SPLITS)
    balance = stats["class_balance"]
    assert balance["fraud_transactions"] == events["is_fraud"].sum()
    assert balance["fraud_rate"] == pytest.approx(events["is_fraud"].mean())
    assert sum(w["transactions"] for w in stats["windows"]) == events.height
    assert stats["amounts"]["fraud"]["p50"] == pytest.approx(
        events.filter(pl.col("is_fraud"))["amount"].median()
    )
    assert stats["cardinality"]["cards"] == events["card_id"].n_unique()


def test_report_writes_markdown_json_and_figures(events: pl.DataFrame, tmp_path: Path) -> None:
    md = write_eda_report(events, SPLITS, tmp_path, dataset="synthetic", source="unit test")
    text = md.read_text()
    for section in ("Phase 0 answers", "Amount distribution", "Missingness", "Entity cardinality"):
        assert section in text
    for figure in FIGURES:
        assert (tmp_path / "figures" / figure).stat().st_size > 0
        assert f"figures/{figure}" in text
    payload = json.loads((tmp_path / "eda.json").read_text())
    assert payload["dataset"] == "synthetic"
    assert payload["overview"]["transactions"] == events.height

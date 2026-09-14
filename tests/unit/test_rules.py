import json
from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest

from bastion.data.splits import SplitConfig, SplitWindow
from bastion.evaluation.cost import CostModel
from bastion.rules.baseline import (
    RULE_NAMES,
    RuleConfig,
    RuleThresholds,
    apply_rules,
    fit_thresholds,
    load_rule_config,
)
from bastion.rules.evaluate import run_rules_baseline, score_rules, write_report
from bastion.streaming.synthetic import SyntheticConfig, generate

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG = RuleConfig(
    high_amount_quantile=0.99,
    velocity_1h_quantile=0.995,
    new_device_amount_quantile=0.9,
    no_history_amount_quantile=0.95,
    spike_multiplier=5.0,
    spike_min_history=3,
)
COSTS = CostModel(
    currency="USD",
    false_decline_margin_rate=0.1,
    false_decline_fixed_cost=2.0,
    review_cost_per_case=5.0,
    review_catch_rate=0.9,
)
SPLITS = SplitConfig(
    windows=(
        SplitWindow(name="train", start_day=0, end_day=14),
        SplitWindow(name="test", start_day=14, end_day=None),
    )
)
THRESHOLDS = RuleThresholds(
    high_amount=1000.0,
    velocity_1h=3.0,
    new_device_amount=200.0,
    no_history_amount=500.0,
    spike_multiplier=5.0,
    spike_min_history=3,
)


def _feature_rows(**columns: list[object]) -> pl.DataFrame:
    base: dict[str, list[object]] = {
        "amount": [10.0],
        "is_fraud": [False],
        "card_txn_count_1h": [0],
        "card_txn_count_7d": [5],
        "card_amount_sum_7d": [50.0],
        "card_device_seen_before": [True],
    }
    base.update(columns)
    return pl.DataFrame(base, schema_overrides={"card_device_seen_before": pl.Boolean})


@pytest.mark.parametrize(
    ("columns", "expected_rule"),
    [
        ({"amount": [1500.0], "card_amount_sum_7d": [5000.0]}, "high_amount"),
        ({"card_txn_count_1h": [4]}, "velocity_1h"),
        (
            {"amount": [250.0], "card_amount_sum_7d": [1000.0], "card_device_seen_before": [False]},
            "new_device_high_amount",
        ),
        ({"amount": [60.0]}, "amount_spike"),  # mean of 5 prior is 10; 60 > 5 x 10
        (
            {"amount": [600.0], "card_txn_count_7d": [0], "card_amount_sum_7d": [0.0]},
            "no_history_high_amount",
        ),
    ],
)
def test_each_rule_fires_alone(columns: dict[str, list[object]], expected_rule: str) -> None:
    flagged = apply_rules(_feature_rows(**columns), THRESHOLDS).row(0, named=True)
    fired = [name for name in RULE_NAMES if flagged[f"rule_{name}"]]
    assert fired == [expected_rule]
    assert flagged["rules_fired"] == 1


def test_missing_inputs_never_fire_a_rule() -> None:
    frame = _feature_rows(
        amount=[250.0], card_amount_sum_7d=[1000.0], card_device_seen_before=[None]
    )
    assert apply_rules(frame, THRESHOLDS)["rule_new_device_high_amount"].to_list() == [False]


def test_amount_spike_needs_enough_history() -> None:
    frame = _feature_rows(amount=[60.0], card_txn_count_7d=[2], card_amount_sum_7d=[20.0])
    assert apply_rules(frame, THRESHOLDS)["rule_amount_spike"].to_list() == [False]


def test_thresholds_ignore_fraudulent_training_rows() -> None:
    legit = _feature_rows(
        amount=[float(a) for a in range(1, 101)],
        **{
            k: v * 100
            for k, v in {
                "is_fraud": [False],
                "card_txn_count_1h": [0],
                "card_txn_count_7d": [5],
                "card_amount_sum_7d": [50.0],
                "card_device_seen_before": [True],
            }.items()
        },
    )
    fraud = legit.head(10).with_columns(pl.lit(1e6).alias("amount"), pl.lit(True).alias("is_fraud"))
    assert fit_thresholds(legit, CONFIG) == fit_thresholds(pl.concat([legit, fraud]), CONFIG)


@pytest.fixture(scope="module")
def events() -> pl.DataFrame:
    return generate(
        SyntheticConfig(days=21, n_cards=300, n_merchants=40, card_testing_attacks_per_day=2.0)
    )


def test_thresholds_are_fit_on_the_train_window_only(events: pl.DataFrame) -> None:
    # Inflate every amount after the train window: thresholds must not move.
    cutoff = events["event_ts"].min() + timedelta(days=14)  # type: ignore[operator]
    inflated = events.with_columns(
        pl.when(pl.col("event_ts") >= cutoff)
        .then(pl.col("amount") * 100)
        .otherwise(pl.col("amount"))
    )
    _, original = score_rules(events, SPLITS, CONFIG)
    _, after = score_rules(inflated, SPLITS, CONFIG)
    assert original == after


def test_baseline_report_is_complete_and_reproducible(events: pl.DataFrame, tmp_path: Path) -> None:
    result = run_rules_baseline(events, SPLITS, COSTS, CONFIG)
    assert set(result.windows) == {"train", "test"}
    assert set(result.rules) == set(RULE_NAMES)
    test_window = result.windows["test"]
    assert test_window["fraud_value_caught"] + test_window["loss_missed_fraud"] == pytest.approx(
        test_window["fraud_value_total"]
    )

    md = write_report(result, COSTS, tmp_path, dataset="synthetic", source="unit test")
    assert "Headline: `test` window" in md.read_text()
    first = (tmp_path / "rules_baseline.json").read_text()
    write_report(
        run_rules_baseline(events, SPLITS, COSTS, CONFIG),
        COSTS,
        tmp_path,
        dataset="synthetic",
        source="unit test",
    )
    assert (tmp_path / "rules_baseline.json").read_text() == first
    assert json.loads(first)["thresholds"]["spike_min_history"] == 3


def test_repository_rule_config_is_valid() -> None:
    assert load_rule_config(REPO_ROOT / "configs").spike_min_history >= 1

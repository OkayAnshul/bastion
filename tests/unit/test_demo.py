"""The compose demo's bootstrap: everything the scoring service needs, prepared once."""

from pathlib import Path

from bastion.data.splits import SplitConfig
from bastion.demo import DemoInputs, DemoPaths, bootstrap_demo
from bastion.evaluation.cost import load_cost_model
from bastion.policy.config import OverrideConfig, PolicyConfig
from bastion.policy.engine import TunedPolicy, serving_policy
from bastion.rules.baseline import load_rule_config
from bastion.streaming.synthetic import SyntheticConfig
from bastion.training.bundle import ModelBundle
from tests.unit.test_training import LABELS, fast_model_config, small_splits

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_bootstrap_prepares_a_servable_model_and_policy_once(tmp_path: Path) -> None:
    paths = DemoPaths.under(tmp_path / "data" / "processed", tmp_path / "artifacts")
    splits: SplitConfig = small_splits()
    policy = PolicyConfig(
        reviews_per_day=5,
        review_budget_sweep=(0, 5),
        threshold_candidates=10,
        overrides=OverrideConfig(),
        reason_codes_top_k=3,
    )
    inputs = DemoInputs(
        synthetic=SyntheticConfig(days=50, n_cards=400, n_merchants=40),
        splits=splits,
        labels=LABELS,
        model=fast_model_config(),
        rules=load_rule_config(REPO_ROOT / "configs"),
        costs=load_cost_model(REPO_ROOT / "configs"),
        policy=policy,
    )
    tracking = f"sqlite:///{tmp_path / 'mlflow.db'}"

    assert bootstrap_demo(
        paths, inputs, tracking_uri=tracking, artifacts_dir=tmp_path / "artifacts"
    )
    assert paths.ready()
    assert (paths.reports / "phase4" / "policy.md").exists()

    # The files are what the scoring service loads, and its guard accepts them.
    bundle = ModelBundle.load(paths.model)
    parameters = serving_policy(
        policy,
        inputs.costs,
        tuned=TunedPolicy.load(paths.policy),
        model_version=f"local/{bundle.metadata['mlflow_run_id']}",
        spec_fingerprint=bundle.spec.fingerprint(),
    )
    assert parameters.tuned

    # A restarted stack finds everything in place and does no work.
    assert not bootstrap_demo(
        paths, inputs, tracking_uri=tracking, artifacts_dir=tmp_path / "artifacts"
    )

"""A self-contained demo: synthetic events, a trained model and a tuned policy (ARCHITECTURE §5).

``bastion demo bootstrap`` prepares what ``docker compose --profile demo up`` needs:

1. a synthetic event table in the canonical schema, at ``data/processed/events_synthetic.parquet``;
2. a LightGBM bundle trained on it, copied to ``artifacts/demo/model``;
3. the review threshold tuned for that bundle, at ``artifacts/demo/policy.json``.

When all three already exist it does nothing, so restarting the stack is fast. The demo proves the
running system end to end. It is not a benchmark: the synthetic fraud is separable (ARCHITECTURE
§7).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from bastion.data.labels import LabelDelayConfig
from bastion.data.splits import SplitConfig
from bastion.evaluation.cost import CostModel
from bastion.policy.config import PolicyConfig
from bastion.policy.sweep import run_policy_sweep, tuned_policy, write_policy_report
from bastion.rules.baseline import RuleConfig
from bastion.streaming.synthetic import SyntheticConfig, generate
from bastion.training.bundle import ModelBundle
from bastion.training.dataset import ModelConfig
from bastion.training.train import TrainingInputs, train_model

DATASET = "synthetic"


@dataclass(frozen=True)
class DemoPaths:
    events: Path
    model: Path
    policy: Path
    reports: Path

    @classmethod
    def under(cls, processed_dir: Path, artifacts_dir: Path) -> DemoPaths:
        demo = artifacts_dir / "demo"
        return cls(
            events=processed_dir / "events_synthetic.parquet",
            model=demo / "model",
            policy=demo / "policy.json",
            reports=demo / "reports",
        )

    def ready(self) -> bool:
        return self.events.exists() and (self.model / "model.txt").exists() and self.policy.exists()


@dataclass(frozen=True)
class DemoInputs:
    synthetic: SyntheticConfig
    splits: SplitConfig
    labels: LabelDelayConfig
    model: ModelConfig
    rules: RuleConfig
    costs: CostModel
    policy: PolicyConfig


def bootstrap_demo(
    paths: DemoPaths,
    inputs: DemoInputs,
    *,
    tracking_uri: str,
    artifacts_dir: Path,
    force: bool = False,
) -> bool:
    """Prepare the demo. Returns False when it was already prepared and nothing was done."""
    if paths.ready() and not force:
        return False
    if force or not paths.events.exists():
        paths.events.parent.mkdir(parents=True, exist_ok=True)
        generate(inputs.synthetic).write_parquet(paths.events, compression="zstd")
    events = pl.read_parquet(paths.events)

    training = TrainingInputs(
        events,
        inputs.splits,
        inputs.labels,
        inputs.model,
        inputs.rules,
        DATASET,
        str(paths.events),
    )
    trained = train_model(
        training,
        results_dir=paths.reports / "phase1",
        artifacts_dir=artifacts_dir,
        tracking_uri=tracking_uri,
    )
    if paths.model.exists():
        shutil.rmtree(paths.model)
    shutil.copytree(trained.bundle_dir, paths.model)

    bundle = ModelBundle.load(paths.model)
    version = f"local/{bundle.metadata['mlflow_run_id']}"  # the label the scoring service reports
    result = run_policy_sweep(
        events,
        bundle,
        splits=inputs.splits,
        labels=inputs.labels,
        costs=inputs.costs,
        policy=inputs.policy,
        rules=inputs.rules,
    )
    write_policy_report(
        result,
        inputs.costs,
        paths.reports / "phase4",
        dataset=DATASET,
        source=str(paths.events),
        model_version=version,
    )
    tuned_policy(result, bundle=bundle, model_version=version, costs=inputs.costs).save(
        paths.policy
    )
    return True

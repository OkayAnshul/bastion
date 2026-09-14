# Learning log index

One line per entry. The template is in `docs/LEARNING_PLAN.md` §1. Entries explain the concepts
behind code that exists in this repository, using numbers measured on this project's data.

| Entry | Concept | Code it explains | Status |
|---|---|---|---|
| [0.1](phase-0/0.1-fraud-data-landscape.md) | The fraud data landscape: imbalance, cost asymmetry, delayed labels | `bastion.data`, `bastion.eda` | written; IEEE-CIS numbers pending |
| [0.2](phase-0/0.2-baseline-rules.md) | Why a rules baseline comes first | `bastion.rules.baseline`, `bastion.evaluation` | written; IEEE-CIS numbers pending |
| [1.1](phase-1/1.1-temporal-splits.md) | Temporal splits and label maturity | `bastion.data.splits`, `bastion.data.labels`, `bastion.training.dataset` | written; IEEE-CIS numbers pending |
| [1.2](phase-1/1.2-point-in-time-correctness.md) | Point-in-time correctness (flagship entry) | `bastion.features.definitions`, `bastion.features.batch`, `bastion.features.naive` | written; IEEE-CIS leakage gap pending |
| [1.3](phase-1/1.3-gbdt-internals.md) | How gradient-boosted trees split | `bastion.training.models` | written; IEEE-CIS numbers pending |
| [1.4](phase-1/1.4-pr-auc-vs-roc-auc.md) | PR-AUC vs ROC-AUC when fraud is rare | `bastion.evaluation.metrics`, `bastion.training.pipeline` | written; IEEE-CIS numbers pending |
| [1.5](phase-1/1.5-calibration.md) | Calibration for an expected-loss policy | `bastion.evaluation.calibration`, `bastion.training.train` | written; IEEE-CIS numbers pending |

Also:

- [glossary.md](glossary.md): terms that needed looking up.
- [mistakes.md](mistakes.md): bugs that took real debugging, and what they teach.

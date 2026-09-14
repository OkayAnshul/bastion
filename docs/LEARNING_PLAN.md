# Bastion — Learning Plan & Log System

You are building this to learn, not to have built it. The logs are how the learning survives past the commit.

---

## 1. The learning log system

```
/docs/learning/
  00-index.md                 running index, one line per entry
  phase-0/
    0.1-fraud-data-landscape.md
    0.2-baseline-rules.md
  phase-1/
    1.1-temporal-splits.md
    1.2-point-in-time-correctness.md
    1.3-gbdt-internals.md
    1.4-imbalance-and-metrics.md
    1.5-calibration.md
  ...
  /glossary.md                terms you had to look up
  /mistakes.md                things you got wrong and how you found out
```

### Entry template

```markdown
# <Concept>

**Phase:** 1 · **Date:** · **Time spent:**

## What problem does this solve?
One paragraph. If you can't write it without jargon, you don't understand it yet.

## How it works
Your own words. A small worked example with real numbers from our dataset.
Diagram if spatial.

## Why we chose it here
Tie to the ADR. What constraint made this the answer?

## What we rejected and why
The alternative, and the specific reason it loses in our context.

## Where it breaks
Failure modes. When would this be the wrong call?

## The interview answer
3–4 sentences. Out loud. This is what you'll actually say.

## Sources
Links + what each one gave you.

## Open questions
Things still fuzzy. Revisit at phase end.
```

**Rule:** `mistakes.md` is the most valuable file in the repo. Every bug that took over an hour gets an entry: symptom, wrong hypothesis, actual cause, how you'd catch it faster next time. Interviewers ask "tell me about a hard bug" and most candidates have nothing.

---

## 2. Curated references by phase

Read the **core** items. Skim the rest when stuck. Don't read ahead of the phase you're in — context is what makes material stick.

### Cross-cutting spine (start here, return often)

- **📘 Reproducible Machine Learning for Credit Card Fraud Detection** — Le Borgne, Bontempi et al. Free online handbook, Python notebooks, and it covers point-in-time features, imbalance, and cost metrics better than anything else free. `https://fraud-detection-handbook.github.io/fraud-detection-handbook/` — **this is your single most important resource.**
- **📗 Designing Machine Learning Systems** — Chip Huyen. Chapters on feature engineering, train/serve skew, and monitoring map almost 1:1 onto this project.
- **🎓 Stanford CS329S: Machine Learning Systems Design** — free lecture notes online, Chip Huyen.
- **🎓 MLOps Zoomcamp** — DataTalks.Club, free, full course on YouTube. Search: `DataTalksClub MLOps Zoomcamp`.
- **🎓 Full Stack Deep Learning** — free lectures; the deployment, monitoring and testing modules.

---

### Phase 0 — Data, baselines, fraud domain

**Concepts:** class imbalance, cost asymmetry, chargeback lifecycle, rules engines.

| | Resource |
|---|---|
| Core | Fraud Detection Handbook, Ch. 2–3 (baseline + data) |
| Domain | Stripe Radar and Feedzai engineering blogs — search `Stripe Radar how it works`, `Feedzai blog fraud detection machine learning` |
| Video | StatQuest: `StatQuest ROC and AUC clearly explained`, and `StatQuest precision recall` |
| Why rules first | Search: `rules engine vs machine learning fraud detection tradeoffs` |

**Log entries:** `0.1 fraud data landscape`, `0.2 why a rules baseline comes first`.

---

### Phase 1 — Honest modelling *(the most important phase)*

**Concepts:** temporal validation, label maturity, data leakage, as-of joins, GBDT internals, PR-AUC, calibration.

| | Resource |
|---|---|
| **Core** | Fraud Detection Handbook Ch. 5 (validation strategies) — read twice |
| Leakage | Kaufman et al., *Leakage in Data Mining: Formulation, Detection, and Avoidance* (KDD 2011) — the foundational paper |
| Leakage (practical) | Search: `data leakage machine learning time series features`, and the Kaggle "leakage" wiki |
| GBDT | Chen & Guestrin, *XGBoost: A Scalable Tree Boosting System*, arXiv:1603.02754 |
| GBDT | Ke et al., *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*, NeurIPS 2017 |
| GBDT intuition | Video: `StatQuest Gradient Boost Part 1 Regression Main Ideas` (4-part series) |
| Imbalance | `imbalanced-learn` docs, and search `why accuracy is wrong for imbalanced data PR curve` |
| Calibration | Guo et al., *On Calibration of Modern Neural Networks*, arXiv:1706.04599 |
| Calibration | scikit-learn User Guide → "Probability calibration" (worked examples, isotonic vs Platt) |
| As-of joins | pandas `merge_asof` docs; Polars `join_asof` docs |
| Tabular vs deep | Grinsztajn et al., *Why do tree-based models still outperform deep learning on tabular data?*, arXiv:2207.08815 — cite this when asked why no neural net |

**Log entries:** `1.1 temporal splits & label maturity`, `1.2 point-in-time correctness` (your flagship entry — write it as if it were a blog post, because it will become one), `1.3 how GBDTs actually split`, `1.4 PR-AUC vs ROC-AUC under 0.1% positives`, `1.5 calibration and why the policy engine needs it`.

---

### Phase 2 — Streaming & feature stores

**Concepts:** log-structured brokers, partitioning, event time vs processing time, windowing, train/serve skew, online vs offline stores.

| | Resource |
|---|---|
| Core | Apache Kafka docs → "Introduction" and "Design" sections |
| Core | Redpanda docs → quickstart + Kafka compatibility |
| Concepts | Kleppmann, *Designing Data-Intensive Applications*, Ch. 11 (stream processing) — the best explanation of event time vs processing time anywhere |
| Windows | Search: `tumbling vs sliding vs session windows stream processing` |
| Feature stores | Uber engineering blog: `Michelangelo machine learning platform` — where the online/offline split was popularised |
| Feature stores | Feast docs → "What is a feature store" + point-in-time correctness page |
| Skew | Search: `training serving skew feature store` and Google's *Rules of Machine Learning* (Rule #29 is exactly this) |
| Video | Search: `Confluent Kafka fundamentals course`, `Kleppmann turning the database inside out` |
| Redis | Redis docs → pipelining, TTL, hashes |

**Log entries:** `2.1 why a log-structured broker`, `2.2 event time vs processing time`, `2.3 online/offline stores and skew`, `2.4 the parity test`.

---

### Phase 3 — Serving & latency

**Concepts:** async I/O, tail latency, profiling, load testing, p99 vs mean.

| | Resource |
|---|---|
| Core | FastAPI docs → concurrency & `async`/`await` page (explains when async actually helps) |
| Tail latency | Dean & Barroso, *The Tail at Scale* (CACM 2013) — short, classic, quotable in interviews |
| Load testing | Locust or k6 docs → quickstart |
| Profiling | `py-spy` README; search `py-spy flame graph python profiling` |
| Model serving | Search: `ONNX Runtime vs native inference latency benchmark`, `model serving patterns batching` |
| Video | Search: `ArjanCodes FastAPI production`, `Brendan Gregg flame graphs` |

**Log entries:** `3.1 what p99 means and why the mean lies`, `3.2 where my 50ms actually went`.

---

### Phase 4 — Decisioning, cost-sensitive ML, explainability

**Concepts:** expected-loss minimisation, threshold selection, constrained optimisation, SHAP.

| | Resource |
|---|---|
| **Core** | Elkan, *The Foundations of Cost-Sensitive Learning* (IJCAI 2001) — short paper, changes how you think about thresholds |
| Core | Fraud Detection Handbook Ch. 4 (performance metrics, incl. precision@k and card-level metrics) |
| Thresholds | Search: `optimal threshold selection cost matrix classification`, `precision at k fraud review capacity` |
| SHAP | Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions*, arXiv:1705.07874 |
| SHAP | `shap` docs → TreeExplainer; and search `SHAP values explained intuitively` |
| Caution | Search: `limitations of SHAP correlated features` — know the critique, not just the tool |
| Video | Search: `StatQuest SHAP`, `Chip Huyen ML systems design decision thresholds` |

**Log entries:** `4.1 expected loss under a review budget`, `4.2 SHAP: what it does and doesn't prove`, `4.3 why three actions beat two`.

---

### Phase 5 — Graph ML

**Concepts:** entity graphs, message passing, heterogeneous graphs, inductive vs transductive.

| | Resource |
|---|---|
| Core | Hamilton et al., *Inductive Representation Learning on Large Graphs* (GraphSAGE), arXiv:1706.02216 |
| Core | Hamilton, *Graph Representation Learning Book* — free online |
| Course | Stanford CS224W: Machine Learning with Graphs — full lectures free on YouTube. Search: `Stanford CS224W Machine Learning with Graphs` |
| Applied | Search: `graph neural network fraud detection ring`, `PayPal graph fraud detection blog` |
| Tooling | PyTorch Geometric docs → heterogeneous graph learning |
| Reality check | Search: `do GNNs beat feature engineering tabular fraud` — be able to argue both sides |

**Log entries:** `5.1 the entity graph`, `5.2 cheap graph features vs GNN lift`, `5.3 why graph features are snapshotted not real-time`.

---

### Phase 6 — Drift & retraining

**Concepts:** covariate shift, concept drift, PSI, KS test, ADWIN, verification latency.

| | Resource |
|---|---|
| Core | Evidently AI docs + their blog series on data drift metrics (PSI, Wasserstein, KS — and when each misleads) |
| Survey | Lu et al., *Learning under Concept Drift: A Review*, arXiv:2004.05785 |
| Classic | Gama et al., *A Survey on Concept Drift Adaptation* (ACM Computing Surveys 2014) |
| Detectors | Search: `ADWIN drift detection explained`, `Page-Hinkley test drift` |
| PSI | Search: `population stability index credit risk explained` (it comes from credit scoring, worth knowing the origin) |
| Adversarial angle | Search: `adversarial concept drift fraud detection` |
| Video | Search: `Evidently AI ML monitoring tutorial`, `MLOps Zoomcamp monitoring module` |

**Log entries:** `6.1 covariate shift vs concept drift`, `6.2 PSI and its blind spots`, `6.3 the retraining trigger policy`.

---

### Phase 7 — LLM explanation layer & evals

**Concepts:** grounding, faithfulness, hallucination detection, eval harness design.

| | Resource |
|---|---|
| Core | Search: `LLM evaluation harness design faithfulness metrics`, and the Ragas docs on faithfulness |
| Grounding | Search: `grounded generation citation verification LLM` |
| Evals | Hamel Husain's writing on LLM evals — search `Hamel Husain your AI product needs evals` |
| Guardrails | Search: `guardrails LLM output validation structured` |

**Log entries:** `7.1 why the LLM has no decision authority`, `7.2 building a faithfulness eval`.

---

## 3. How to study alongside building

- **Read one core item before writing the code it relates to**, not after. The code then answers a question you already have.
- **Timebox to 90 minutes.** If a concept isn't landing, log it under "open questions" and keep building. It usually clarifies once the code breaks.
- **Explain it in the log before you use it.** If you can't write the "interview answer" section, you're copying, not learning.
- **Once per phase, write the phase summary as a public blog post draft.** Twelve weeks gives you six posts, which is a distribution engine for off-campus outreach.

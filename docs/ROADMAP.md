# Bastion — Roadmap

Twelve weeks at student pace (10–14 hrs/week). **Phases 0–4 are the shippable core.** Everything after is upside.

Rule: no phase starts until the previous phase's exit criteria are met and the learning log for it is written. Scope creep is the failure mode, not lack of ambition.

---

## Phase 0 — Foundations (Week 1)

**Goal:** repo, data, and a dumb thing that works end to end.

- Repo scaffold, `docker-compose.yml`, Makefile, pre-commit, pytest.
- Download IEEE-CIS Fraud Detection. Understand the schema. Write the data dictionary yourself.
- EDA notebook: class balance, amount distribution, temporal density, missingness.
- **Rules baseline**: 5 hand-written rules (amount > X, velocity > Y, new device + high amount...). Measure it.

**Exit criteria:** you can state the fraud rate, the amount distribution of fraud vs non-fraud, and the rules baseline's precision/recall and rupee loss. A model that can't beat this is worthless.

**Stop-line:** do not train an ML model this week.

---

## Phase 1 — Honest offline model (Weeks 2–3)

**Goal:** a model whose numbers you believe.

- Temporal split with a label-maturity buffer (ADR-007).
- Naive feature set → LightGBM → record PR-AUC. *This number is inflated. Expect it.*
- Rebuild every aggregate feature point-in-time correct (ADR-003) via an as-of join.
- Retrain. Record the drop. **Plot both.**
- Calibrate (isotonic). Reliability curve + Brier score.
- MLflow tracking from the first run, not retrofitted.

**Exit criteria:** the leakage comparison plot exists, and you can explain in two sentences why the first number was a lie.

**This is the highest-value week of the whole project.** Do not rush it.

---

## Phase 2 — Streaming path (Weeks 4–5)

**Goal:** the same features, computed live.

- Transaction simulator that replays the dataset in event-time order at configurable speed.
- Redpanda topic, producer, consumer.
- Streaming feature builder with sliding windows → Redis, with TTLs.
- **Parity test**: replay the log, assert streaming features == batch features (ADR-004).

**Exit criteria:** the parity test passes in CI. Without it, everything downstream is untrustworthy.

---

## Phase 3 — Scoring service (Week 6)

**Goal:** sub-50ms decisions over HTTP.

- FastAPI `/v1/score`, model loaded from MLflow registry at startup.
- Pipelined Redis fetch (one round trip, not N).
- Structured decision logging, async.
- Load test (`locust` or `k6`). Record p50/p95/p99 and throughput.
- Profile and fix the slowest component. Write down what it was.

**Exit criteria:** a latency histogram in the README, measured not estimated.

---

## Phase 4 — Policy engine + analyst console (Weeks 7–8)

**Goal:** the part that makes recruiters care.

- Expected-loss threshold optimisation under review capacity (ADR-005). Sweep N, plot fraud-value-caught vs review budget vs false-decline rate.
- Hard-rule overrides layered above the model.
- Reason codes from SHAP top-k.
- Console: live decision feed, review queue sorted by expected loss, case detail page.

**Exit criteria:** you can say "at 200 reviews/day we catch X% of fraud value at a Y% false-decline rate" and point at the curve.

### ⛔ Decision point

**Ship here.** Write the README, record the demo, publish the leakage post. You now have a complete, honest, top-1% project. Phases 5–7 are how you push further — pick based on the roles you're actually seeing, not on completeness.

---

## Phase 5 — Graph layer (Weeks 9–10)

- Build card–device–merchant–IP graph from the offline log.
- Aggregate graph features → LightGBM. Measure the lift. If the lift is small, **say so** rather than hiding it.
- Optional: GraphSAGE / heterogeneous GNN in PyTorch Geometric, compared against the cheap features.
- Console: neighbourhood visualisation for a flagged case. This is the demo shot.

**Exit criteria:** an attributed lift number and a screenshot of a fraud ring.

---

## Phase 6 — Drift + retraining loop (Week 11)

- Inject a novel attack pattern mid-stream via the simulator.
- Evidently PSI/KS on features and scores; calibration drift; decision-rate drift.
- Alert → retraining job → shadow deploy → eval gate → promote.
- **The money plot:** detection rate over time showing degradation, alert firing, and recovery after retrain.

**Exit criteria:** that plot.

---

## Phase 7 — LLM case narratives (Week 12)

- Grounded narrative generation from SHAP + rules + graph context (ADR-010).
- Hallucinated-number guardrail.
- A small eval set of cases with expected key facts; measure narrative faithfulness. **Building the eval is the point**, not the generation.

---

## Continuous, every week

- Learning log entry (see LEARNING_PLAN.md). Non-negotiable.
- Commit history that reads like a story, not `final_v2_fixed`.
- README updated as you go, never at the end.

---

## Checkpoint questions (ask yourself at each phase gate)

1. Can I explain every design choice in this phase to someone who disagrees with it?
2. Is there a number here I haven't personally measured?
3. Did I beat the previous baseline, and do I know *why*?
4. If an interviewer opened this file, would it embarrass me?

---

## Anti-goals

Things to deliberately **not** do, and to be able to say why:

- Kubernetes. Docker Compose is correct for this scale; k8s would be resume-driven development.
- A custom deep tabular architecture. Trees win here.
- Real-time GNN inference. Doesn't fit the latency budget; snapshot features do.
- Scraping real payment data. Ethically and legally wrong, and unnecessary.
- A second project before this one ships.

# Bastion — Architecture & Design Decisions

**Real-time payments fraud & risk decisioning platform.**
Real-world problem, production-grade engineering, public + simulated data. Say exactly that everywhere.

---

## 1. The one-paragraph pitch

Transactions arrive on a stream. Within a fixed latency budget, Bastion computes behavioural features from history, scores the transaction with an ML model, and a policy layer converts that score into a business decision — approve, challenge, or block — under a hard constraint that human analysts can only review N cases per day. Every decision is explained, logged, and monitored for drift. When the fraud pattern shifts, the system notices and retrains.

The model is maybe 20% of this. The other 80% is what gets you hired.

---

## 2. System diagram

```
                         ┌──────────────────────────┐
                         │   Transaction Simulator  │
                         │  (replay + attack        │
                         │   injection)             │
                         └────────────┬─────────────┘
                                      │ JSON events
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        INGEST LAYER                                  │
│   Redpanda / Kafka   topic: transactions.raw                         │
│   partitioned by card_id  (ordering guarantee per entity)            │
└────────────┬───────────────────────────────────┬────────────────────┘
             │                                   │
             ▼                                   ▼
┌────────────────────────────┐      ┌────────────────────────────────┐
│  ONLINE FEATURE BUILDER    │      │   RAW EVENT SINK               │
│  consumer, sliding windows │      │   Parquet on disk / MinIO      │
│  1m / 10m / 1h / 24h / 7d  │      │   = offline store, source of   │
│  writes → Redis            │      │     truth for training         │
└────────────┬───────────────┘      └────────────┬───────────────────┘
             │                                   │
             ▼                                   ▼
┌────────────────────────────┐      ┌────────────────────────────────┐
│  ONLINE STORE (Redis)      │      │  OFFLINE FEATURE PIPELINE      │
│  key: entity:card:{id}     │◄─────┤  MUST be point-in-time correct │
│  TTL'd aggregates          │ same │  (see ADR-003 — the crux)      │
└────────────┬───────────────┘ logic└────────────┬───────────────────┘
             │                                   │
             │                                   ▼
             │                        ┌────────────────────────────────┐
             │                        │  TRAINING                      │
             │                        │  baseline rules → LightGBM     │
             │                        │  → + graph features → GNN      │
             │                        │  MLflow tracking + registry    │
             │                        └────────────┬───────────────────┘
             │                                     │ registered model
             ▼                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      SCORING SERVICE (FastAPI)                       │
│   POST /v1/score  → p99 target < 50ms                                │
│   1. fetch features from Redis (pipelined)                           │
│   2. model.predict_proba  (champion)                                 │
│   3. shadow-score challenger, log only, never decide                 │
│   4. hand score to policy engine                                     │
└────────────┬────────────────────────────────────────────────────────┘
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       POLICY / DECISION ENGINE                       │
│   expected-loss optimisation under review-capacity constraint        │
│   thresholds τ_block, τ_review  (see ADR-005)                        │
│   hard rules override model (velocity caps, blocklists, allowlists)  │
│   emits: decision + reason_codes + SHAP top-k                        │
└────────────┬───────────────────────────────┬────────────────────────┘
             │                               │
             ▼                               ▼
┌────────────────────────────┐   ┌────────────────────────────────────┐
│  DECISION LOG              │   │   ANALYST CONSOLE (React/Streamlit)│
│  every score, features,    │   │   queue sorted by expected loss    │
│  model version, latency    │   │   graph neighbourhood view         │
└────────────┬───────────────┘   │   LLM case narrative (grounded)    │
             │                   └────────────────────────────────────┘
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         MONITORING                                   │
│   Evidently: PSI / KS on feature + score distributions               │
│   calibration drift, decision-rate drift, latency SLO                │
│   alert → triggers retraining DAG                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component contracts

### 3.1 Transaction event (the canonical schema)

```json
{
  "txn_id": "uuid",
  "event_ts": "2026-09-14T10:32:11.412Z",
  "card_id": "c_8821",
  "device_id": "d_331",
  "merchant_id": "m_77",
  "merchant_category": "5814",
  "ip": "49.36.x.x",
  "amount": 2499.00,
  "currency": "INR",
  "channel": "ecom",
  "is_fraud": null
}
```

`is_fraud` is **null at scoring time** and only filled in the offline store, days later, when a chargeback arrives. That delay is not an inconvenience — it is the central fact of fraud ML and it drives ADR-003 and ADR-007.

### 3.2 Feature groups

| Group | Examples | Window |
|---|---|---|
| Velocity | txn count, sum, distinct merchants, distinct devices | 1m, 10m, 1h, 24h, 7d |
| Deviation | amount / rolling mean, z-score vs card's own history | 30d |
| Entity risk | merchant historical fraud rate, device age in system | lifetime |
| Graph | degree, shared-device count, ring size, 2-hop fraud density | snapshot |
| Contextual | hour-of-day, day-of-week, is-new-merchant-for-card | instant |

### 3.3 Latency budget (p99, total 50ms)

```
network in            3ms
Redis feature fetch  12ms   (single pipelined round trip, not N gets)
feature assembly      4ms
model inference       8ms   (LightGBM, no GPU)
policy engine         3ms
logging (async)       0ms   (fire-and-forget to queue)
headroom             20ms
```

Measure this from day one. A latency number you measured yourself is worth more than any accuracy figure.

---

## 4. Decision log (ADRs)

Format: what we chose, why, what we rejected, and what it would cost to change later.

### ADR-001 — Redpanda over Kafka for local dev
**Chose:** Redpanda in Docker Compose, Kafka wire-protocol compatible.
**Why:** single binary, no ZooKeeper/KRaft ceremony, starts in seconds on a laptop. Your code still uses the Kafka client, so the skill transfers and you can say "Kafka" honestly.
**Rejected:** real Kafka (slow local startup, more moving parts); Redis Streams (weaker ordering/partitioning story, and you lose the vocabulary interviewers use).
**Cost to change:** near zero — swap the broker address.

### ADR-002 — LightGBM as primary model, not a neural net
**Chose:** gradient-boosted trees.
**Why:** on tabular, imbalanced, heterogeneous financial data, GBDTs still beat deep nets in accuracy per unit of effort, train in seconds, and give native SHAP explanations which the analyst console needs. Inference is single-digit ms on CPU.
**Rejected:** a tabular transformer (slower, more fragile, no accuracy win here — and choosing it would signal you follow fashion rather than evidence); logistic regression as primary (good baseline, insufficient ceiling).
**Nuance to state in interviews:** "I chose trees because the data is tabular and the latency budget is 50ms on CPU. I'd revisit for sequence modelling of the card's transaction history."

### ADR-003 — Point-in-time correct feature computation (the crux of this project)
**Chose:** every training row's features are computed using **only events with `event_ts` strictly before that row's `event_ts`**, via an as-of join on a time-sorted event log.
**Why:** the naive approach — computing "transactions per card in last 24h" over the whole dataset with a groupby — leaks future information into the past. Models trained that way post spectacular offline scores and collapse in production. This is the single most common silent failure in student fraud projects and in a lot of professional ones.
**Rejected:** naive full-dataset aggregation (leaky); Feast as the feature store (excellent tool, but adopting it early hides the mechanism you're trying to learn — see roadmap, you may adopt it in week 10 *after* hand-rolling it).
**Deliverable:** train both versions. Publish the gap. That plot is your interview centrepiece.

### ADR-004 — Same code path computes online and offline features
**Chose:** one feature-definition module, two executors (streaming consumer, batch backfill).
**Why:** train/serve skew is the second silent killer. If the training features are computed by a pandas script and the serving features by a different Python function, they will diverge, and you will not find out until the model is live.
**Rejected:** separate implementations "because batch is easier in pandas" (this is the trap).
**Test to write:** replay the offline log through the streaming path and assert the Redis values match the batch values within tolerance. This test is itself a portfolio artifact.

### ADR-005 — Optimise expected monetary loss under a review-capacity constraint, not F1
**Chose:** three-way decision (approve / review / block) with thresholds chosen to minimise
`E[loss] = Σ(missed fraud × amount) + Σ(false declines × margin) + (reviews × analyst cost)`
subject to `reviews_per_day ≤ N`.
**Why:** this is what a risk team actually does. F1 treats a ₹200 false negative and a ₹80,000 one identically. Capacity is finite. Framing your results in rupees is what makes a recruiter forward your profile.
**Rejected:** maximising F1 or AUC (business-meaningless); a binary approve/block decision (throws away the highest-value action, which is human review of the uncertain middle).
**Metric to report:** **fraud value caught @ review budget**, plus false-decline rate.

### ADR-006 — Calibration is a first-class requirement
**Chose:** isotonic or Platt calibration on a held-out set; report Brier score and a reliability curve.
**Why:** the policy engine multiplies probability by amount. If your "0.7" isn't really 70%, every expected-loss calculation is wrong. Boosted trees are systematically miscalibrated out of the box.
**Rejected:** using raw model scores as probabilities (the default mistake).

### ADR-007 — Temporal splits only, never random
**Chose:** train on weeks 1–8, validate on week 9, test on weeks 10–12. Plus a label-maturity buffer: exclude the most recent K days from training because chargebacks haven't landed yet.
**Why:** random splits let the model see the future and let the same fraud ring appear in train and test. Reported performance becomes fiction.
**Rejected:** `train_test_split(shuffle=True)`. If you see it in your code, it is a bug.

### ADR-008 — Graph features before a GNN
**Chose:** build the card–device–merchant–IP graph in NetworkX (later igraph for speed), extract aggregate features (shared-device count, 2-hop fraud density, connected-component size), feed them to LightGBM. Only then try a GNN.
**Why:** most of the lift from graphs comes from these cheap features. Starting with a GNN means you can't tell whether the graph or the architecture gave you the gain. Baseline discipline again.
**Rejected:** going straight to PyTorch Geometric (impressive-sounding, harder to attribute, much harder to serve in 50ms).
**Serving note:** graph features are computed on a snapshot schedule, not per-request. Say why: 2-hop traversal doesn't fit the latency budget.

### ADR-009 — Champion/challenger with shadow scoring
**Chose:** new models are deployed in shadow, scoring live traffic but never deciding, until they beat the champion on the eval gate.
**Why:** it's how model deployment actually works in risk, and it's cheap to build here.
**Rejected:** blue/green swaps (fine, but shadow teaches you more and generates a comparison dataset).

### ADR-010 — LLM is confined to explanation, never to decisions
**Chose:** the LLM receives SHAP attributions, the rule hits, and the graph neighbourhood summary, and writes a case narrative for the analyst. It has no authority over approve/block.
**Why:** grounded, auditable, and it survives the obvious interview challenge ("why would you put a non-deterministic model in a regulated decision path?"). The correct answer is that you wouldn't.
**Rejected:** an LLM scoring transactions (latency, cost, non-determinism, no regulatory defensibility).
**Guardrail to build:** the narrative must cite feature values it was given; add a check that flags any number in the output not present in the input payload.

### ADR-011 — Mapping anonymised IEEE-CIS onto the canonical schema
*Added during Phase 0 (implementation decision).*
**Chose:** a deterministic adapter at the dataset boundary (`bastion.data.adapter`) that builds proxy entity ids. `card_id` hashes card1–card6, addr1 and `D1n` (the account's first-seen day). `device_id` is a coarse fingerprint (DeviceInfo, OS, browser, screen) when identity data exists, else null. `merchant_id` is a proxy from ProductCD and the recipient email domain. `ip` stays null. `event_ts` is a fixed anchor (2017-11-30 UTC) plus `TransactionDT`. Every other raw column travels as an `attr_*` vendor attribute. Amounts stay in USD, the dataset's currency.
**Why:** every downstream component (stream, Redis keys, graph) is written against the canonical schema. Mapping once at the boundary keeps the platform dataset-agnostic, and the synthetic generator emits the same table with real entities.
**Rejected:** renaming the canonical schema to IEEE-CIS columns, which couples the whole platform to one anonymised dataset. Raw `card1` as the card id, since one card1 value is shared by many customers and velocity would measure a bank segment, not a customer. Inventing IPs or merchants, since fabricated structure would fabricate graph lift. Converting USD to INR at an assumed rate, which is false precision.
**Cost:** the proxies are weak. Merchant and device nodes are hubs, most rows have no device, and there is no IP. Velocity, entity-risk and graph results on IEEE-CIS therefore understate what real processor data would allow. The README limitations say so.

---

## 5. What "done" looks like

A stranger clones the repo, runs `docker compose up`, opens `localhost:3000`, sees transactions flowing and decisions being made, clicks a blocked transaction and reads why. The README shows the architecture diagram, the leakage plot, and the loss-vs-review-budget curve.

That is the whole game.

---

## 6. Honest limitations (put these in the README yourself)

- Data is public (IEEE-CIS) and synthetic (simulator). Real fraud is harder and adversarial in ways a simulator can't reproduce.
- No real money, no real merchants, no regulatory obligations.
- Single-node throughput; not tested at bank scale.
- Injected attack patterns are designed by me, so drift detection is being graded on a test I wrote.

Stating these makes every other claim you make more believable.

---

## 7. Implementation notes

Deviations and clarifications recorded while building. Each one is the smallest change that resolves a
conflict between this document and reality.

- **Docs location.** Design documents live in `docs/`.
- **Build mode.** The owner chose build-first over the teach-while-building rules in
  `docs/CLAUDE_CODE_KICKOFF.md`. `CLAUDE.md` records the working agreement.
- **Currency.** Monetary results are reported in USD, the IEEE-CIS native currency. The rupee framing
  in ADR-005 is illustrative. See ADR-011.
- **Kafka topics** *(Phase 2, planned).* Created idempotently by application code rather than a
  one-shot compose container, which keeps `docker compose up --wait` simple.
- **Analyst console** *(Phase 4, planned).* Streamlit, served on port 3000 to match §5.
- **Retraining trigger** *(Phase 6, planned).* A Python job invoked by the monitor, not an orchestrator
  DAG. One pipeline does not justify Airflow.
- **Offline store.** Parquet on local disk; no MinIO.
- **Calibrator selection (ADR-006).** ROADMAP names isotonic calibration. On the first synthetic run,
  isotonic lowered test PR-AUC from 0.9998 to 0.9863 and raised log loss from 0.0005 to 0.0040 compared
  with the uncalibrated scores, while Platt matched them. Isotonic regression can map score ranges that
  held no fraud in the calibration window to a probability of exactly 0, and an expected-loss policy
  would approve everything there. Bastion therefore picks isotonic or Platt by log loss on the most
  recent 30% of the calibration window and refits the winner on the whole window
  (`calibration: select`). The test window never influences the choice. On the revised synthetic
  generator the selection chose Platt (holdout log loss 0.00016 vs 0.00022).
- **Synthetic data proves pipelines, not models.** The generator's attacks are separable by design
  (LightGBM test PR-AUC 1.0000 on the 183-day synthetic table), so model quality and the leakage gap
  are reported from IEEE-CIS only. See `docs/learning/mistakes.md`.
- **Phase gate vs. data access.** ROADMAP says no phase starts before the previous phase's exit
  criteria are met. Phase 0's exit numbers need IEEE-CIS, and the download needs the owner's Kaggle
  token. While that is pending, Phase 1 *code* is built and tested on synthetic data. No Phase 1
  result is reported, and no release is tagged, until the Phase 0 numbers exist.

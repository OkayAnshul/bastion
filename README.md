# Bastion

**A real-time payments fraud and risk decisioning platform.**
A real-world problem built with production-style engineering on **public (IEEE-CIS) and simulated data**.
Bastion has never processed real payments, real customers, or real money.

> Status: **Phase 0 — Foundations (in progress).** Every number in this README comes from a
> run recorded in `docs/results/`. Anything not yet measured says **unmeasured**; anything not yet
> built says **planned**.

---

## Problem

A payment arrives, and within a fixed latency budget the system has to answer: approve it, send it to
a human analyst, or block it? Two constraints make this hard:

1. **Labels arrive late.** A fraudulent transaction is confirmed only when a chargeback lands, days
   or weeks later. Anything a model learns from history must respect that delay.
2. **Humans are a finite resource.** Analysts can review only N cases a day. The goal is not a high
   F1 score but minimum expected monetary loss under that review budget.

Bastion is built around those two facts: point-in-time correct features, calibrated probabilities,
an expected-loss policy with a review-capacity constraint, and monitoring that detects when the
fraud pattern shifts.

## Architecture

```
Transaction simulator (replay + attack injection)
        │  POST /v1/score, then publish event
        ▼
Redpanda (Kafka API)  ──►  raw event sink (Parquet, offline store)
        │                          │
        ▼                          ▼
Streaming feature builder    Point-in-time feature pipeline ─► training ─► MLflow registry
        │   (same feature definitions: ADR-004)                              │
        ▼                                                                     ▼
Redis online store  ──────────────►  FastAPI scoring service (LightGBM, calibrated)
                                              │
                                              ▼
                              Policy engine (expected loss @ review budget)
                                              │
                          ┌───────────────────┼────────────────────┐
                          ▼                   ▼                    ▼
                    Decision log      Analyst console       Monitoring → retraining
```

Full design, component contracts, latency budget and ADRs: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Phase status

| Phase | Scope | Status |
|---|---|---|
| 0 | Scaffold, data contracts, EDA, rules baseline | in progress |
| 1 | Temporal validation, point-in-time features, leakage experiment, calibration | planned |
| 2 | Streaming replay, Redpanda, Redis online features, batch/stream parity test | planned |
| 3 | FastAPI scoring service, latency benchmark | planned |
| 4 | Expected-loss policy engine, analyst console | planned |
| 5 | Entity graph features (optional GNN) | planned |
| 6 | Drift detection and retraining loop | planned |
| 7 | Grounded LLM case narratives and faithfulness evals | planned |

## Results

| Result | Status |
|---|---|
| Fraud rate and amount distribution (IEEE-CIS) | unmeasured |
| Rules baseline: precision / recall / monetary loss | unmeasured |
| Leakage experiment: naive vs point-in-time PR-AUC | unmeasured |
| Calibration: Brier score, reliability curve | unmeasured |
| Batch/stream feature parity | unmeasured |
| Scoring latency p50 / p95 / p99, throughput | unmeasured |
| Fraud value caught vs review budget | unmeasured |
| Graph feature lift | unmeasured |
| Drift detection and recovery | unmeasured |
| Narrative faithfulness | unmeasured |

## Local setup

Requirements: [uv](https://docs.astral.sh/uv/), Docker with Compose v2, GNU make, and a Kaggle account
that has accepted the IEEE-CIS Fraud Detection competition rules.

```bash
make setup      # Python 3.12 env via uv + git hooks
make test       # unit tests
make up         # Redpanda, Redis, MLflow
make data       # download IEEE-CIS + build the canonical event table
make eda        # EDA report → docs/results/phase0/
make baseline   # rules baseline → docs/results/phase0/
```

Run `make help` for all targets.

## Data

- **IEEE-CIS Fraud Detection** (Vesta, Kaggle 2019): anonymised e-commerce transactions with fraud
  labels. It is downloaded locally and never redistributed. It lacks explicit card, device, merchant and IP
  identifiers, so Bastion maps it onto its canonical event schema with documented proxies
  (ADR-011 in `docs/ARCHITECTURE.md`, caveats in `docs/data_dictionary.md`).
- **Synthetic transactions** from Bastion's own generator, used for tests, CI, and attack injection.

## Repository layout

```
configs/        versioned experiment, cost and policy parameters
docs/           architecture, roadmap, results, learning logs, mistakes log
src/bastion/    the platform (data, features, training, streaming, serving, policy, ...)
tests/          unit tests; integration tests (need services); data tests (need IEEE-CIS)
```

## Limitations

- Data is public (IEEE-CIS) and synthetic. Real fraud is harder, and adversarial in ways a simulator
  cannot reproduce.
- No real money, no real merchants, no regulatory obligations.
- Single-node throughput, measured on a laptop; not tested at bank scale.
- Injected attack patterns are designed by the author, so drift detection is graded on a test the author wrote.

## Future work

Tracked in [`docs/ROADMAP.md`](docs/ROADMAP.md).

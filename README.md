# Bastion

**A real-time payments fraud and risk decisioning platform.**
A real-world problem built with production-style engineering on **public (IEEE-CIS) and simulated data**.
Bastion has never processed real payments, real customers, or real money.

> Status: **Phases 0–1 code complete and tested on synthetic data; IEEE-CIS results pending the
> dataset download. Phase 2 complete: batch/stream parity passes in CI against Redpanda and Redis.
> Phase 3 complete: scoring latency measured on a laptop (histogram below). Phase 4 code complete:
> expected-loss policy with a review budget, reason codes and an analyst console; its headline policy
> curve waits for IEEE-CIS.** Every number in this README comes from a run recorded in `docs/results/`. Anything not yet measured says **unmeasured**; anything not yet
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
| 0 | Scaffold, data contracts, EDA, rules baseline | code complete; IEEE-CIS results pending dataset download |
| 1 | Temporal validation, point-in-time features, leakage experiment, calibration | code complete; IEEE-CIS results pending dataset download |
| 2 | Streaming replay, Redpanda, Redis online features, batch/stream parity test | complete: parity passes in CI (exit criterion) |
| 3 | FastAPI scoring service, latency benchmark | complete: latency histogram measured (exit criterion); slowest component profiled and fixed |
| 4 | Expected-loss policy engine, analyst console | code complete: policy, review budget, overrides, reason codes, console; exit sentence pending IEEE-CIS |
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
| Batch/stream feature parity | exact equality through Redpanda + Redis in CI (synthetic events; see learning log 2.4) |
| Scoring latency p50 / p95 / p99, throughput | one worker on a laptop, policy engine included: 400 requests/s sustained at p50 2.39 ms, p95 4.96 ms, p99 9.55 ms; 800 requests/s not sustained, with requests beyond the Redis connection pool answered 503 ([report](docs/results/phase4/latency.md)) |
| Fraud value caught vs review budget | unmeasured on IEEE-CIS (`bastion policy sweep` runs end to end on synthetic data, whose numbers are not published) |
| Graph feature lift | unmeasured |
| Drift detection and recovery | unmeasured |
| Narrative faithfulness | unmeasured |

## Scoring latency

![Client-side latency distribution of POST /v1/score at 400 requests/s](docs/results/phase4/figures/latency_histogram.png)

Measured with k6 at a constant arrival rate against one service worker, including the Phase 4
policy engine and reason codes. k6, the service and Redis shared one laptop (Intel i7-1255U), and the
payloads and Redis history are synthetic (271,529 events). At 400 requests/s, the highest rate
sustained with no dropped or failed requests, client-side latency was **p50 2.39 ms, p95 4.96 ms,
p99 9.55 ms** against a 50 ms p99 budget. Three repeats at that rate gave p99 7.44 to 9.39 ms. At 800
requests/s one worker does not keep up; requests that cannot get a Redis connection get a 503, not
an error. Full report: [`docs/results/phase4/latency.md`](docs/results/phase4/latency.md).

**Slowest component: online feature evaluation.** Profiling the running service under load put 40%
of samples in feature evaluation and 5% in LightGBM. Most of that feature time went to input
validation and key building, repeated for every time window. Evaluating all windows in one pass,
with identical outputs, cut feature evaluation from 0.458 to 0.279 ms per call. p99 at 400
requests/s fell from 10.4–50.4 ms to 5.7–7.6 ms across four runs of each version. Details:
[`docs/results/phase3/profiling.md`](docs/results/phase3/profiling.md).

**What the policy engine costs.** Pricing the three actions and checking the review budget takes
0.04 to 0.06 ms at the median. Reason codes (TreeSHAP) take about 2 ms at the median, but only for
reviewed and blocked transactions: about 1% of requests here. Over three repeats at 400 requests/s,
p99 was 5.7 to 7.6 ms before the policy engine and 7.4 to 9.4 ms with it.

These are single-process laptop numbers, not a production capacity claim.

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
make train      # LightGBM + calibration → docs/results/phase1/ (tracked in MLflow)
make experiment-leakage   # naive vs point-in-time features → docs/results/phase1/
make test-integration     # batch/stream parity through Redpanda + Redis (after make up)
make serve                # scoring service on :8000 (BASTION_MODEL_PATH or the MLflow champion)
make bench-prepare && make bench   # k6 latency benchmark → docs/results/phase3/ (needs podman or docker)
make policy               # review-budget sweep → docs/results/phase4/ and a tuned threshold for serving
make console              # analyst console on :3000 over artifacts/decisions/decisions.db
make demo                 # the whole stack on synthetic data (see Demo below)
```

Run `make help` for all targets.

### Try the pipeline without Kaggle

The synthetic generator writes the same canonical event table as the IEEE-CIS adapter, so every
Phase 0 command runs end to end without a Kaggle account:

```bash
uv run bastion data synthetic --out artifacts/demo/events.parquet
uv run bastion eda --events artifacts/demo/events.parquet --out-dir artifacts/demo --dataset synthetic
uv run bastion baseline rules --events artifacts/demo/events.parquet --out-dir artifacts/demo --dataset synthetic
uv run bastion train --events artifacts/demo/events.parquet --out-dir artifacts/demo --dataset synthetic
uv run bastion experiment leakage --events artifacts/demo/events.parquet --out-dir artifacts/demo --dataset synthetic
```

Synthetic results show that the pipeline works. They are not benchmark results: the fraud patterns
were written by the author, so they are easier to catch than real fraud.

## Demo

`make demo` runs `docker compose --profile demo up --build`:

1. `bootstrap` generates a synthetic event table, trains a model and tunes the review threshold.
   It does this once; later starts reuse the files in `data/` and `artifacts/`.
2. `simulator` plays the payment gateway. It replays the transactions in event time, asks
   `POST /v1/score` for each decision, then publishes the transaction to Redpanda.
3. `feature-builder` keeps Redis current, `scoring` decides, and `decision-sink` stores every
   decision for the console.
4. The analyst console at <http://localhost:3000> shows live decisions, the review queue sorted by
   expected fraud loss, and case pages with reason codes and analyst verdicts.

The data is synthetic, so the demo shows the system running, not how much fraud it catches.
The full stack has not been run end to end yet, because the build machine has no Docker. Its parts
are tested separately: unit tests, AppTest renders of every console page, and Redpanda integration
tests in CI.

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
benchmarks/     k6 load script and the feature-evaluation timing script
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

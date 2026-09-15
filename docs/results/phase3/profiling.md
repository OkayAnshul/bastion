# Scoring service profile: the slowest component and the fix

ROADMAP Phase 3: *"Profile and fix the slowest component. Write down what it was."*

**The slowest component was online feature evaluation.** More precisely, it was per-call overhead
repeated for every time window. Each `window_aggregates` and `distinct_in_window` call validated its
inputs with `min()`/`max()` passes, rebuilt the (entity, time) search keys, and sorted or summed the
history again, once per window, for every velocity, distinct-count and device window of every
request. The fix evaluates all windows of a feature group in one call, sharing that work. No feature
value changed (evidence below).

**Setup, for every number on this page.**

- **Machine:** one laptop, 12th Gen Intel i7-1255U (12 logical CPUs), 15.3 GiB RAM, Linux 7.0.2,
  Python 3.12.14, `powersave` CPU governor, on AC power.
- **Service:** one worker process.
- **Redis:** 8.6.6 in rootless Podman.
- **Load:** k6 1.8.1 in a container.
- **Data:** synthetic payloads and history (271,529 events and 248,582 labels in Redis); a 37-input
  model trained on synthetic data.

The latency reports for both versions record revision `0e24265-dirty`. The *before* runs used the
feature code of commit `0e24265`; the *after* runs used the code of the commit that adds this page.
Nothing else differed between the two versions.

## How it was found

The service was started under py-spy (`--rate 250 --format raw`) while k6 sent 200 requests/s for
35 s. Both profiles served 7,001 requests with none failed. `bastion bench profile` groups the
samples by stage. A sample counts toward a stage when any frame of its stack belongs to it, so
nested stages overlap. It converts samples into busy time per request: samples × 4 ms ÷ 7,001.

| Stage | Before: share of samples | Before: ms per request | After: share of samples | After: ms per request |
|---|---|---|---|---|
| Score handler (everything below) | 68.1% | 3.273 | 62.8% | 2.943 |
| `features_from_snapshot` | 40.5% | 1.944 | 29.4% | 1.380 |
| … window aggregates | 14.4% | 0.692 | 10.2% | 0.479 |
| … distinct counts | 17.4% | 0.837 | 10.3% | 0.485 |
| Redis pipeline | 17.0% | 0.818 | 21.2% | 0.992 |
| LightGBM predict | 5.4% | 0.261 | 6.0% | 0.282 |
| Decision log writer | 1.1% | 0.051 | 1.5% | 0.070 |

Samples: 8,406 before, 8,208 after (py-spy reported one sampling error in the after profile).

Before the fix, the hottest leaf frame was `_entity_time_key` (5.3% of all samples), followed by
NumPy's `_amin` (2.8%) and `_amax` (2.5%). That is input validation, not arithmetic. After the fix,
`_entity_time_key` is 2.0% and `_amin`/`_amax` are no longer among the ten hottest leaves. Feature
evaluation is still the largest stage inside the handler.

Two cautions:

- **Profiler overhead.** Profiled busy time includes the profiler's own cost, so it is not the
  service's real latency. It is used here to rank components and to compare the two versions under
  the same load.
- **Noise.** Redis pipeline time rose from 0.818 to 0.992 ms per request although that code did not
  change. Treat differences of that size between single profiles as noise.

## The fix

`window_aggregates_many` and `distinct_in_window_many` in `src/bastion/features/definitions.py` take
a list of windows. They compute input validation, the (entity, time) keys, the sort order,
de-duplication and the cumulative sums once. Only the part that really depends on the window is
computed per window: the lower bound for aggregates, and where each cover interval ends for distinct
counts.

The batch executor and the online path both call these functions, so they still share one definition
(ADR-004). The single-window functions remain as thin wrappers.

Evidence that outputs did not change:

- A hypothesis property test checks every window of both multi-window functions against a
  brute-force count over random histories (`test_multi_window_functions_match_brute_force`). It runs
  alongside the existing brute-force and point-in-time tests.
- The batch/online parity tests and the HTTP-equals-offline serving test pass unchanged.
- On 2,000 real snapshots read from the benchmark Redis, the old and new `features_from_snapshot`
  produce byte-identical feature JSON.

## Feature evaluation in isolation

`benchmarks/feature_eval.py` fetched 2,000 snapshots from the benchmark Redis once. Each version then
evaluated every snapshot 5 times (10,000 calls). The old version was imported from a `git worktree`
of `0e24265`.

| Version | mean | p50 | p95 | p99 |
|---|---|---|---|---|
| Before (`0e24265`) | 0.458 ms | 0.455 ms | 0.517 ms | 0.549 ms |
| After | 0.279 ms | 0.278 ms | 0.312 ms | 0.334 ms |

The mean time per call fell by 39%.

## End to end under load

The first pair of full benchmarks ran *before*, then *after*. To check that the difference was not
noise or thermal drift, 200 and 400 requests/s were then each repeated three times, interleaved, in
the opposite order (*after*, then *before*). For the *before* repeats, the service ran from a
worktree of `0e24265`; everything else was identical. Every run lasted 30 s and was sustained, with
zero failed or dropped requests. Repeat data: [`repeats/before.json`](repeats/before.json),
[`repeats/after.json`](repeats/after.json).

**Client-side latency at 400 requests/s** (12,000–12,001 requests per run)

| Run | Version | p50 ms | p95 ms | p99 ms | max ms | handler p99 ms |
|---|---|---|---|---|---|---|
| First pair | before | 3.36 | 9.17 | 33.16 | 100.16 | 22.43 |
| First pair | after | 2.31 | 4.60 | 6.17 | 12.36 | 4.31 |
| Repeat 1 | before | 3.22 | 6.97 | 10.40 | 21.64 | 7.22 |
| Repeat 2 | before | 3.01 | 16.17 | 50.42 | 68.22 | 33.53 |
| Repeat 3 | before | 2.86 | 6.19 | 11.71 | 101.21 | 7.61 |
| Repeat 1 | after | 2.15 | 3.75 | 5.68 | 62.44 | 3.97 |
| Repeat 2 | after | 2.16 | 3.83 | 5.77 | 16.33 | 4.07 |
| Repeat 3 | after | 2.21 | 4.88 | 7.64 | 21.22 | 5.34 |

**Client-side latency at 200 requests/s** (6,000–6,001 requests per run)

| Version | p50 ms, three repeats | p99 ms, three repeats |
|---|---|---|
| Before | 2.86, 2.55, 2.98 | 6.29, 6.06, 5.94 |
| After | 2.47, 2.49, 2.48 | 5.03, 5.01, 5.09 |

What this shows, and what it doesn't:

- **The tail improved consistently.** At 400 requests/s the old code's p99 ranged from 10.4 to
  50.4 ms over four runs and varied a lot from run to run. The new code's p99 ranged from 5.7 to
  7.6 ms.
- **The mechanism is queueing, not the CPU saving itself.** The CPU saved per request is small next
  to that change. With one event loop, every request's CPU time delays the requests behind it. Close
  to saturation, a small cut in service time therefore removes a lot of waiting. The server-side
  stages fit this picture. The Redis stage, which includes waiting for the event loop, had p99
  5.9–32.1 ms before and 2.8–4.1 ms after at 400 requests/s. This mechanism is an interpretation of
  the measurements, not a separate measurement.
- **The feature stage's median barely moved.** At 400 requests/s its p50 was 0.73–0.83 ms before
  and 0.71–0.85 ms after, while its p99 fell from 1.97–2.13 ms to 1.38–1.59 ms. That stage also
  includes building the model's input row, which did not change.
- **Order was reversed between the pairs.** Runs were sequential, not simultaneous. The order was
  reversed between the first pair and the repeats, and the improvement appeared both times.
- **Direction only.** Three repeats are enough to show a consistent direction, not to put a
  confidence interval on a p99.
- **800 requests/s is not sustained by either version** on one worker. See *Overload* below.

## Overload: 800 requests/s

One worker does not sustain 800 requests/s with either version of the feature code. What mattered
is *how* it failed. The reports count the HTTP status of every request k6 sent, where 0 means no
response. Full reports: [before](before/latency.md), [after](latency.md),
[after, with 503 handling](overload/latency.md).

| Run | Requests | 200 | 500 | 503 | No response | Dropped | p50 ms | p99 ms |
|---|---|---|---|---|---|---|---|---|
| Before the feature fix | 23,734 | 15,193 | 6,106 | 0 | 2,435 | 267 | 343.10 | 680.86 |
| After the feature fix | 23,814 | 22,789 | 727 | 0 | 298 | 187 | 104.41 | 403.63 |
| After the feature fix, with 503 handling | 23,792 | 23,238 | 0 | 554 | 0 | 208 | 35.14 | 395.81 |

- **Every 500 was an unhandled `redis.exceptions.MaxConnectionsError`.** The two service logs hold
  exactly 6,106 and 727 of them. redis-py's asyncio pool holds 100 connections by default and raises
  instead of waiting when all are in use. That happens once more than 100 requests are in flight on
  a saturated event loop.
- **Each 500 also logged a full traceback.** That is extra CPU work in a process that was already
  saturated.
- **The change.** The pool size is now a setting (`BASTION_REDIS_MAX_CONNECTIONS`, default 100).
  The score handler answers Redis connection and timeout errors with `503 online store unavailable`,
  counts them as `store_errors` in `GET /v1/model`, and logs no traceback. In the run with that
  change, the service log held no tracebacks, `store_errors` was 554 (matching the 503s k6 saw), and
  every other request succeeded.
- **Single overload runs vary a lot.** The latency columns do not show that 503 handling made the
  service faster; the claim is only about how requests fail. The disappearance of no-response
  failures (298 to 0) fits with less work per rejected request, but one run cannot establish it.

## Reproduce

```bash
make up                    # Redis and friends (these runs used a separate Redis container)
uv run bastion data synthetic --out artifacts/bench/events.parquet
uv run bastion train --events artifacts/bench/events.parquet --out-dir artifacts/bench/phase1 --dataset synthetic
uv run bastion bench prepare --events artifacts/bench/events.parquet
BASTION_MODEL_PATH=artifacts/models/<run id> uv run bastion serve --port 8010
uv run bastion bench latency --target http://127.0.0.1:8010 --out-dir artifacts/bench/latency

# Profile: start the service under py-spy, load it, then summarise.
BASTION_MODEL_PATH=artifacts/models/<run id> uv run py-spy record --rate 250 --format raw \
    --output artifacts/bench/profile.txt --duration 60 -- .venv/bin/python .venv/bin/bastion serve --port 8011
uv run bastion bench profile artifacts/bench/profile.txt --rate-hz 250 --requests <requests served>

# Feature evaluation in isolation: see the docstring of benchmarks/feature_eval.py.
```

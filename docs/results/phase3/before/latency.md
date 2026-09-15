# Scoring latency: POST /v1/score

Measured by `bastion bench latency` at git revision `0e24265-dirty`. Client and server ran on the same laptop, so the client competes with the service for CPU.

## Machine

| | |
|---|---|
| cpu | 12th Gen Intel(R) Core(TM) i7-1255U |
| logical_cpus | 12 |
| memory | 15.3 GiB |
| os | Linux 7.0.2-arch1-1 |
| python | 3.12.14 |
| cpu_governor | powersave |
| on_ac_power | yes |

## Setup

| | |
|---|---|
| Service | `bastion serve`, 1 worker process(es), JSONL decision log |
| Model | local/da7db7ba6d194aae8cddbe1d35519fc3 (37 inputs), synthetic training data |
| Online store | Redis 8.6.6 in a rootless Podman container, host network |
| History in Redis | 271,529 events and 248,582 labels |
| Requests | 20000 synthetic payloads, cycled |
| Load generator | k6 (docker.io/grafana/k6:1.8.1) in a container, host network, constant arrival rate |
| Per rate | 30 s, after one 10 s warm-up run |

## Results by arrival rate

Client latency is measured by k6 from request start to response end; server stages come from the decision log of the same requests. **Sustained** means every scheduled request started, none failed, and the achieved rate stayed within 2% of the target.

| Target rps | Achieved rps | Sustained | Dropped | Errors | p50 ms | p95 ms | p99 ms | max ms | server p99 ms |
|---|---|---|---|---|---|---|---|---|---|
| 25 | 25.0 | yes | 0 | 0.00% | 7.30 | 9.03 | 9.43 | 10.61 | 7.30 |
| 50 | 50.0 | yes | 0 | 0.00% | 6.75 | 8.77 | 9.45 | 61.04 | 7.31 |
| 100 | 100.0 | yes | 0 | 0.00% | 4.32 | 6.88 | 7.72 | 8.83 | 6.29 |
| 200 | 200.0 | yes | 0 | 0.00% | 2.96 | 5.32 | 6.04 | 11.42 | 4.82 |
| 400 | 399.9 | yes | 0 | 0.00% | 3.36 | 9.17 | 33.16 | 100.16 | 22.43 |
| 800 | 783.2 | no | 267 | 35.99% | 343.10 | 614.45 | 680.86 | 924.56 | 414.90 |

![Latency by arrival rate](figures/latency_vs_rate.png)

## Stage breakdown at 400 requests/s

Client p99 **33.16 ms** against the 50 ms budget: within budget at this rate.

| Stage | Budget (p99) | p50 ms | p95 ms | p99 ms | max ms |
|---|---|---|---|---|---|
| Redis round trip + event-loop wait | 12 ms | 0.67 | 4.79 | 21.28 | 90.32 |
| Feature evaluation + row | 4 ms | 0.83 | 1.80 | 2.13 | 3.23 |
| LightGBM + calibration | 8 ms | 0.15 | 0.27 | 0.38 | 1.27 |
| Handler total | n/a | 2.31 | 6.19 | 22.43 | 91.17 |

The Redis stage runs from the start of the handler until the pipeline's replies have been processed. With one event-loop thread per worker, it therefore also includes time a request waits while other requests run feature evaluation, which is why its tail grows with load.

![Latency distribution](figures/latency_histogram.png)

The handler total excludes HTTP parsing, validation and response serialisation; the gap between it and the client-side latency is the rest of the request path, plus the client's own overhead on a shared machine.

# Glossary

Terms that came up while building Bastion, in plain language, grouped by the phase where they
first matter.

## Phase 0 — fraud domain and baselines

**Chargeback.** A cardholder disputes a transaction and the issuer reverses it. For a fraudulent
transaction, the chargeback is usually how the merchant learns it was fraud, often weeks later.

**Card-not-present (CNP).** A payment where the card is not physically shown: e-commerce, phone
orders. IEEE-CIS transactions are CNP e-commerce.

**Class imbalance.** One class (fraud) is far rarer than the other. Accuracy becomes meaningless:
predicting "legit" for everything scores well while catching nothing.

**Precision.** Of the transactions flagged, the fraction that were actually fraud.

**Recall.** Of the fraudulent transactions, the fraction that were flagged. Recall by *value* (fraud
dollars caught / total fraud dollars) is often the more relevant form.

**PR-AUC (average precision).** The area under the precision-recall curve: ranking quality
concentrated on the positive class. Its no-skill baseline equals the positive rate, unlike ROC-AUC,
whose baseline is always 0.5.

**False-decline rate.** The fraction of legitimate transactions that get declined. Each one is a
lost sale and an annoyed customer.

**Rules engine.** Hand-written if-then conditions (amount above X, velocity above Y). They are
transparent and fast, but brittle, and every threshold is a guess someone has to maintain.

**Label maturity.** A label is *mature* once enough time has passed for its chargeback to have
arrived. Recent transactions look legitimate only because their labels have not landed yet.

**Event time.** When a transaction actually happened, as opposed to when a system processed it
(*processing time*). All of Bastion's windows and splits use event time.

## Phase 3 — serving and latency

**Percentile (p50, p95, p99).** The value below which that share of measurements falls. p50 is the
median; p99 is the latency that 1 request in 100 exceeds. Budgets are set on high percentiles
because every user sees them: at 400 requests/s, "1 in 100" is four slow payments every second.

**Tail latency.** The high percentiles (p99, p99.9, max). Queueing, garbage collection, a busy
event loop and CPU frequency changes all show up here first, long before they move the mean.

**Mean latency.** The average. A few very slow requests barely move it, so it hides exactly the
behaviour a latency budget exists to control. Never report it alone.

**Throughput.** Requests completed per second. Only meaningful together with the latency at which
it was achieved: any service reaches high throughput if responses may take seconds.

**Open vs closed load model.** A *closed* load generator runs N users, each waiting for a response
before sending the next request, so a slow server receives less load. An *open* model sends
requests at a fixed arrival rate regardless of responses, as real payment traffic does. Bastion's
benchmark uses k6's `constant-arrival-rate` executor, an open model.

**Coordinated omission.** The measurement error a closed model makes: while the server stalls, the
generator stops sending, so the requests that would have waited are never measured and the tail
looks better than it is.

**Sampling profiler.** A profiler that periodically records the call stack of a running process
(py-spy reads it from outside, at a fixed rate) instead of instrumenting every call. Overhead is low
enough to profile under load; the share of samples in a function estimates its share of CPU time.

**Flame graph.** A picture of sampled stacks: width is the share of samples, height is call depth.
Wide plateaus near the top are where the CPU actually spends its time.

**Event loop.** The single thread that runs all coroutines in an asyncio process. CPU work inside a
handler blocks every other request on that loop, so it appears as waiting time in their Redis stage.

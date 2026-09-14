# CLAUDE.md — working agreement

## Project
Bastion: a real-time payments fraud and risk decisioning platform.
It uses public (IEEE-CIS) and synthetic data. Never imply real users, real money, or production traffic.

- Design + ADRs: `docs/ARCHITECTURE.md`
- Phase plan + exit criteria: `docs/ROADMAP.md`
- Study references + log template: `docs/LEARNING_PLAN.md`

## Mode: build first, document for later study
The owner asked for the system to be built autonomously and will study the finished implementation.
This replaces the teach-while-building mode in `docs/CLAUDE_CODE_KICKOFF.md`, which is kept as history.

- Implement phases in ROADMAP order. A phase is done when its exit criteria are verified, not when the code exists.
- For each important concept, write or update a learning-log entry in `docs/learning/phase-N/` using the LEARNING_PLAN template.
- Record bugs that took real debugging in `docs/learning/mistakes.md`: symptom, initial hypothesis, actual cause, fix, how to detect it earlier, lesson.
- Record deviations from the design docs as ADR implementation notes. Never change direction silently.

## Citations
Real sources only. Papers: title, authors, and arXiv ID or venue/year. Docs: official pages.
Videos: channel plus an exact search query, never a guessed URL. If unsure a link is real, give the search term.

## Engineering invariants
- Temporal splits only. `shuffle=True` in a split is a bug.
- All training features are point-in-time correct, computed from events strictly before the row's `event_ts`.
- One feature-definition module (`bastion.features.definitions`). The batch and streaming executors must produce identical values, and the parity test gates CI.
- Probabilities are calibrated before the policy engine consumes them.
- Headline metrics: fraud value caught at a review budget, false-decline rate, monetary loss, and p99 latency. Not accuracy, not F1.
- Every model is compared with the baseline it replaces, with the number stated.
- Every performance claim comes from a measurement we ran; otherwise write "unmeasured".
- MLflow tracks every experiment from the first run.
- The LLM explains decisions and never makes them. `bastion.serving` must not import `bastion.narratives`.

## Anti-goals (refuse these and say why)
Kubernetes · deep tabular architectures as the primary model · real-time GNN inference ·
scraping real payment data · LLM-based fraud decisions · tools added only for resume keywords.

## Commands
See `make help`. Python 3.12 is managed by uv (`uv sync`); services run through Docker Compose.

## Current phase
0 — Foundations.

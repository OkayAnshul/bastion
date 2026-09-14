# Claude Code — Kickoff Prompt

**How to use:** create an empty repo, drop `ARCHITECTURE.md`, `ROADMAP.md` and `LEARNING_PLAN.md` into `/docs/`, then paste the block below as your first message in Claude Code. The second block goes into `CLAUDE.md` at the repo root so the rules persist across sessions.

---

## PART A — Paste this as your first message

```
I'm building Bastion: a real-time payments fraud and risk decisioning platform.
It's a learning project — I'm a 3rd-year B.Tech CSE student targeting AI/ML and
ML-engineering roles. I know Kotlin/Android and Python well; I'm new to
streaming, feature stores, and production ML.

Read /docs/ARCHITECTURE.md, /docs/ROADMAP.md and /docs/LEARNING_PLAN.md first.
They contain the system design, the decision log (ADRs), the phase plan and the
study references. Follow them. If you think an ADR is wrong, argue with me
before deviating — don't silently do something else.

## How I want you to work with me

This is the most important part. I am here to learn, not to receive a finished
repo. So:

1. TEACH BEFORE YOU BUILD. Before writing code for any new concept, explain in
   plain language: what problem it solves, how it works, why we chose it here,
   what we rejected, and where it breaks. Use a worked example with real numbers
   from our dataset. Then write the code.

2. NEVER write more than ~80 lines without stopping to check I followed it. End
   those checkpoints with one concrete question that tests understanding, not
   "does that make sense?".

3. LEAVE GAPS ON PURPOSE. For each module, implement the scaffolding and the
   hard parts, but leave 1–2 clearly marked `# TODO(anshul):` functions for me
   to write, with a docstring specifying the contract and a failing test. Pick
   the ones with the most learning value, not the most tedious ones.

4. WRITE THE LEARNING LOGS. After each concept, create/update the entry in
   /docs/learning/phase-N/ using the template in LEARNING_PLAN.md. Fill every
   section except "Open questions" — leave that for me. In the Sources section,
   cite specific real resources: papers with arXiv IDs or venue+year, official
   documentation pages, and for videos give me the channel name plus the exact
   search query rather than a guessed URL. Never invent a link. If you're not
   confident a URL is real, give me the search term instead.

5. UPDATE /docs/learning/mistakes.md whenever we hit a bug that took real
   debugging: symptom, my wrong hypothesis, actual cause, how to catch it faster.

6. BASELINE DISCIPLINE. Never let me skip a baseline. Every model must be
   compared against the simpler thing it replaced, with the number stated.

7. MEASURE, DON'T ESTIMATE. Any performance claim in code comments, logs or the
   README must come from a measurement we actually ran. If we haven't measured
   it, write "unmeasured".

8. PUSH BACK. If I ask for something that's scope creep, premature optimisation,
   resume-driven (Kubernetes, a transformer for tabular data, real-time GNN
   inference), or methodologically wrong (random train/test split, F1 as the
   headline metric, uncalibrated probabilities in the policy engine) — tell me
   plainly and explain why. Don't just comply. The anti-goals list in ROADMAP.md
   is binding.

9. HONESTY IN THE README. This project uses public and synthetic data. Never
   write copy that implies real users, real money, or real production traffic.

## Right now: Phase 0 only

Do not build ahead of the phase. Start with:

a) Propose the repo structure and explain each directory's purpose in one line.
b) docker-compose.yml with Redpanda + Redis + MLflow + a Python service, plus a
   Makefile (`make up`, `make test`, `make lint`). Explain what each service is
   for and why it's in the stack.
c) A data-loading module for the IEEE-CIS Fraud Detection dataset with a schema
   contract (pydantic) matching the canonical event in ARCHITECTURE.md §3.1.
d) An EDA script that answers: fraud rate, amount distribution by class,
   temporal density, missingness, cardinality of key entities. Tell me what to
   look for in the output before I run it.
e) The rules baseline (5 rules) with an evaluation harness reporting precision,
   recall, PR-AUC, and total rupee loss. This is the number every future model
   must beat.
f) The first learning log entries: 0.1 and 0.2.

Then stop and wait for me. Don't start Phase 1.

Before you write anything, tell me your plan for (a)–(f) in about 10 lines and
flag anything in the docs you disagree with.
```

---

## PART B — Put this in `CLAUDE.md` at the repo root

```markdown
# CLAUDE.md — working agreement

## Project
Bastion: real-time payments fraud & risk decisioning platform.
Public (IEEE-CIS) + synthetic data. Never imply real users or real money.
Design and decisions: /docs/ARCHITECTURE.md · Plan: /docs/ROADMAP.md ·
Study refs: /docs/LEARNING_PLAN.md

## Mode: teach-while-building
- Explain any new concept before implementing it: problem → mechanism → why here
  → what we rejected → failure modes. Worked example with real numbers.
- Stop every ~80 lines with a comprehension question.
- Leave 1–2 `# TODO(anshul):` functions per module with docstring contract +
  failing test. Choose high-learning, not high-tedium.
- After each concept, write the learning log entry (template in LEARNING_PLAN.md).
- Log hard bugs to /docs/learning/mistakes.md.

## Citations
Real sources only. Papers: title + authors + arXiv ID or venue/year.
Docs: official pages. Videos: channel + exact search query, never a guessed URL.
If unsure a link is real, give the search term instead.

## Engineering invariants
- Temporal splits only. `shuffle=True` in a split is a bug.
- All training features point-in-time correct. Assume leakage until proven otherwise.
- One feature-definition module; streaming and batch must produce identical values.
  The parity test gates CI.
- Probabilities must be calibrated before the policy engine consumes them.
- Headline metrics are: fraud value caught @ review budget, false-decline rate,
  rupee loss, p99 latency. Not accuracy. Not F1.
- Every model compared to the baseline it replaces, with the number stated.
- Any performance claim must be measured. Otherwise write "unmeasured".
- MLflow logging from the first experiment, not retrofitted.

## Anti-goals (refuse these and say why)
Kubernetes · deep tabular architectures as primary model · real-time GNN
inference · scraping real payment data · a second project before this ships ·
any metric stated without measurement.

## Phase gate
Never build ahead of the current phase in ROADMAP.md. Current phase: 0.
```

---

## Useful follow-up prompts

Keep these for later sessions:

- `Quiz me on phase 1. Five questions an ML interviewer would actually ask about point-in-time correctness and temporal validation. Don't give me the answers until I try.`
- `Review my TODO implementation in <file>. Don't fix it — tell me what's wrong and let me fix it. If it's right, tell me what edge case it misses.`
- `We're at the phase 4 gate. Audit the repo against the checkpoint questions in ROADMAP.md and tell me honestly what's weak.`
- `Turn learning log 1.2 into a blog post draft. Keep my voice, keep it honest, lead with the plot.`
- `I'm being interviewed on this tomorrow. Play a skeptical senior MLE and try to find the holes in my design.`

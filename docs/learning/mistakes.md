# Mistakes log

Bugs that took real debugging, recorded honestly, including the wrong turns. Each entry uses the
same structure so it can be retold in an interview ("tell me about a hard bug").

Template:

```markdown
## <date> — <short title>

**Phase:** N · **Commit(s):** <sha of the fix>

**Symptom.** What was observed, with the actual numbers or error text.

**Initial hypothesis.** What we first thought was wrong, and why it was plausible.

**Actual cause.** What was really wrong.

**Fix.** What changed, plus the regression test that now guards it.

**How we could have detected it earlier.** The check, test or plot that would have caught it sooner.

**What the bug teaches.** The general lesson beyond this codebase.
```

---

## 2026-09-14 — A `.gitignore` rule silently dropped a whole package from a commit

**Phase:** 0 · **Commit(s):** broken `52a62da`; fixed in the next commit, `fix: anchor gitignore
patterns so src/bastion/data is committed`

**Symptom.** Everything was green locally: ruff, strict mypy and all 33 unit tests passed, and the
pre-commit hooks passed. The commit and push succeeded. But the pushed commit held **zero** files
under `src/bastion/data` (`git ls-tree -r HEAD -- src/bastion/data | wc -l` printed `0`), while
`bastion.cli` and `tests/unit/test_adapter.py` import that package. CI failed on the clean
checkout. Its first failing step was ruff's import sorting, which wanted a blank line between
`from bastion.data...` and `from bastion.schemas...`. Ruff classifies an import as first-party by
finding the module under `src/`, so in CI `bastion.data` looked like a third-party package.

**Initial hypothesis.** The staging command ran inside a script that started with `set -e`, so any
failing step (including `git add`) should have aborted before the commit. The first guess was
therefore that `git add` had succeeded, and that the warning it printed ("The following paths are
ignored by one of your .gitignore files: src/bastion/data") was harmless noise. That guess was wrong.

**Actual cause.** Two independent problems combined:

1. `.gitignore` contained `data/`, meant for the top-level dataset directory. A pattern *without a
   leading slash* matches a directory of that name at **any depth**, so it also matched the
   `src/bastion/data/` package.
2. `git add` with an explicitly named ignored path **exits 1 but still stages the other paths**
   (reproduced in a scratch repository). `set -e` should then have stopped the script, but in the
   shell this project's automation runs in, `set -e; false; echo x` still prints `x`: errexit had no
   effect. The commit went ahead with a partial staging area.

**Fix.** Directory patterns are anchored to the repository root (`/data/`, `/artifacts/`,
`/mlruns/`, `/mlartifacts/`, `/profiles/`), and the package is committed. The regression test
`tests/unit/test_repo_hygiene.py` fails if any file under `src/`, `tests/` or `configs/` is
gitignored. Before the fix, the command it runs listed exactly the three missing modules. Commit
chains now use `&&` instead of relying on `set -e`, and check `git diff --cached --name-only` before
committing.

**How we could have detected it earlier.**
- Reading the staging output instead of skimming past it to "hooks passed" and "push ok".
- Checking the staged file list against the files just created before every commit.
- Running tests from a clean checkout. That is exactly what CI does, and CI caught it on the first
  run. The hygiene test now catches this class of bug in `make test`, before a push.

**What the bug teaches.** A green local run proves that the *working tree* works, not that the
*commit* does. The thing you ship is the commit, so verification must run against what was committed.
Ignore rules are code: an overly broad pattern is a silent data-loss bug. And never trust a safety
mechanism (`set -e`) you haven't seen fire in the environment you're actually using.

---

## 2026-09-14: A near-perfect model that had learned the data generator

**Phase:** 1 · **Commit(s):** `fix: remove generator shortcut that made device sharing a perfect fraud flag`

**Symptom.** The first `bastion train` on the 183-day synthetic table scored a test PR-AUC of
**0.9998** for LightGBM and 0.9930 for logistic regression. One feature,
`device_distinct_cards_7d`, carried **54.5%** of total gain. The leakage experiment could not show
anything: the leaky and honest pipelines differed by 0.0002, because every variant sat at the ceiling.

**Initial hypothesis.** The new point-in-time features leaked future information. That was plausible,
since a near-perfect score is the textbook leakage symptom. It did not hold: the no-future-information
property tests passed, and the leaky pipeline was not involved in that run.

**Actual cause.** The generator, not the features. Every legitimate card had two devices of its own
that no other card ever used, while attackers drew from a shared pool of 12 devices. "A device used by
more than one card" was therefore a perfect fraud flag, one that exists nowhere in real payments.
The model learned the generator.

**Fix.** Households of 2–4 cards now share devices and a home connection. 40% of attacks use a
never-seen device, 20% of account takeovers come from the victim's own device, and legitimate
customers make bursts of quick purchases. Regression tests assert that legitimate devices can be
shared and that some fraud uses a device the card already uses. After the fix, device-sharing
features left the top of the importance table (`card_txn_count_1m` 36.0%, `amount` 26.7%).

**What is still true.** Test PR-AUC on synthetic data is still 1.0000. The attacks are extreme by
design: probe bursts seconds apart, tiny amounts, cash-outs many times a card's usual spend. Making
them subtler until a pleasing gap appears would be designing the result. Synthetic runs are
therefore pipeline checks only; model quality and the leakage gap are reported from IEEE-CIS.

**How we could have detected it earlier.**
- Treat a near-perfect score as a bug report, not a result.
- Read the feature-importance table before reading the metric.
- Test the generator for real-world properties it must have ("legitimate devices can be shared"),
  not only for the properties it was built to produce.

**What the bug teaches.** When you generate data, you also generate its shortcuts. A model finds the
cheapest separator you left in, and a perfect score is usually a map of that separator.

---

## 2026-09-14: The CLI loaded the columns rules needed, until features needed one more

**Phase:** 1 · **Commit(s):** `fix: derive CLI column projection from the feature executor`

**Symptom.** After the Phase 1 feature groups landed, all unit tests passed. But
`bastion baseline rules` on a prepared table failed with
`ColumnNotFoundError: unable to find column "merchant_id"`.

**Initial hypothesis.** The rewritten synthetic generator changed the table's schema. It had not: the
table passed the offline contract, and `bastion eda` read the same file without complaint.

**Actual cause.** To save memory on the 434-column IEEE-CIS table, the CLI loads only
`EVENT_COLUMNS`, a hand-written tuple in `bastion.rules.evaluate`. `compute_features` had started
needing `merchant_id` (distinct merchants, merchant familiarity), and the tuple was not updated. The
unit tests handed `run_rules_baseline` full tables, so the projected path was never exercised.

**Fix.** `bastion.features.batch.REQUIRED_EVENT_COLUMNS` is now the single list; `EVENT_COLUMNS`
derives from it. `compute_features` checks its inputs up front and names any missing column. Two
regression tests run the baseline on exactly the projected columns, and drive the CLI end to end
(synthetic data → EDA → rules baseline → training).

**How we could have detected it earlier.** Test through the entry point people run, not only the
function it calls. And keep derived lists derived.

**What the bug teaches.** An optimisation that copies knowledge ("which columns are needed") creates a
second place that must change. The bug shows up in whichever copy nobody tests.

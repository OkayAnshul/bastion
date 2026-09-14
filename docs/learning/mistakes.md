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

"""Repository hygiene: guards on what the published history can silently miss."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src" / "bastion"
LEAKAGE_EXPERIMENT = SRC / "training" / "experiments" / "leakage.py"


@pytest.mark.skipif(
    shutil.which("git") is None or not (REPO_ROOT / ".git").exists(),
    reason="needs a git checkout",
)
def test_no_source_test_or_config_file_is_gitignored() -> None:
    # Regression: an unanchored `data/` pattern once matched the src/bastion/data package, so a
    # commit was pushed without the modules its own tests imported. See docs/learning/mistakes.md.
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "--",
            "src",
            "tests",
            "configs",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    ignored = [
        path
        for path in result.stdout.splitlines()
        if "__pycache__" not in path and not path.endswith(".pyc")
    ]
    assert ignored == []


def test_random_splits_never_appear_outside_the_leakage_experiment() -> None:
    """ADR-007: `train_test_split(shuffle=True)` in this codebase is a bug.

    The one sanctioned shuffle is the leakage experiment, which exists to measure the damage.
    """
    forbidden = re.compile(r"train_test_split|shuffle\s*=\s*True|shuffled_split\(")
    offenders = [
        path.relative_to(REPO_ROOT)
        for path in SRC.rglob("*.py")
        if path != LEAKAGE_EXPERIMENT and forbidden.search(path.read_text())
    ]
    assert offenders == []

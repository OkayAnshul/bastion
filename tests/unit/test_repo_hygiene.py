"""Repository hygiene: guards on what the published history can silently miss."""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


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

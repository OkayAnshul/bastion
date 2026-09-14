"""Provenance for reports and experiment runs: which code produced a result."""

from __future__ import annotations

import subprocess
from pathlib import Path


def git_revision(repo: Path | None = None) -> str:
    """Short commit hash, suffixed ``-dirty`` when tracked files have uncommitted changes.

    Returns ``"unknown"`` outside a git checkout (for example inside the Docker image).
    """

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        sha = git("rev-parse", "--short", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha

"""End-to-end CLI checks: the entry points people actually run, not just the library functions."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bastion.cli import app
from bastion.config import get_settings

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.chdir(REPO_ROOT)  # configs/ are read relative to the repository root
    monkeypatch.setenv("BASTION_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("BASTION_MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def _run(*args: str) -> str:
    result = CliRunner().invoke(app, list(args))
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    return result.output


def test_phase0_and_phase1_commands_run_end_to_end(workspace: Path) -> None:
    events = workspace / "events.parquet"
    _run("data", "synthetic", "--days", "183", "--cards", "400", "--out", str(events))
    common = ["--events", str(events), "--dataset", "synthetic"]
    _run("eda", *common, "--out-dir", str(workspace / "phase0"))
    # Regression: `baseline rules` loads only the columns it needs (docs/learning/mistakes.md).
    _run("baseline", "rules", *common, "--out-dir", str(workspace / "phase0"))
    _run("train", *common, "--out-dir", str(workspace / "phase1"))
    for report in ("phase0/eda.md", "phase0/rules_baseline.md", "phase1/model.md"):
        assert (workspace / report).stat().st_size > 0

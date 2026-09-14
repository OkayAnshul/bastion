from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from bastion.config import Settings, load_config


class _Thresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")
    amount: float


def test_settings_are_overridden_by_bastion_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BASTION_REDIS_URL", "redis://example:6380/1")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.redis_url == "redis://example:6380/1"
    assert settings.raw_dir == Path("data/raw")


def test_load_config_validates_types(tmp_path: Path) -> None:
    (tmp_path / "rules.yaml").write_text("amount: 500.5\n")
    assert load_config("rules", _Thresholds, tmp_path).amount == 500.5


def test_load_config_rejects_misspelled_keys(tmp_path: Path) -> None:
    # A typo must fail loudly rather than silently use a default threshold.
    (tmp_path / "rules.yaml").write_text("amout: 500\n")
    with pytest.raises(ValidationError):
        load_config("rules", _Thresholds, tmp_path)

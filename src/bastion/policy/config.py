"""Policy settings from ``configs/policy.yaml``: review budget, overrides and reason codes."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

from bastion.config import load_config


class OverrideConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    blocked_card_ids: frozenset[str] = frozenset()
    blocked_device_ids: frozenset[str] = frozenset()
    blocked_merchant_ids: frozenset[str] = frozenset()
    allowed_card_ids: frozenset[str] = frozenset()
    velocity_cap_1h: int | None = Field(default=None, ge=1)


class PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reviews_per_day: NonNegativeInt
    review_budget_sweep: tuple[NonNegativeInt, ...] = Field(min_length=1)
    threshold_candidates: int = Field(ge=2)
    overrides: OverrideConfig = OverrideConfig()
    reason_codes_top_k: int = Field(ge=1)


def load_policy_config(config_dir: Path | None = None) -> PolicyConfig:
    return load_config("policy", PolicyConfig, config_dir)

"""Hard rules above the model (ARCHITECTURE §2): blocklists, an allowlist and a velocity cap.

They run before the expected-loss policy and take precedence over it, in this order:

1. A blocklisted card, device or merchant is blocked.
2. Otherwise, an allowlisted card is approved.
3. Otherwise, a card that already made ``velocity_cap_1h`` or more transactions in the previous hour
   is blocked.
4. Otherwise, the model's policy decides.

Blocklists beat the allowlist, so a known-compromised device stops even a trusted card. The same
function runs offline and in the scoring service. Overridden transactions never use review capacity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from bastion.evaluation.cost import Action
from bastion.policy.config import OverrideConfig
from bastion.policy.decide import NO_OVERRIDE
from bastion.policy.expected_loss import StrArray


@dataclass(frozen=True)
class Override:
    action: Action
    rule: str


def override_for(
    *,
    card_id: str,
    device_id: str | None,
    merchant_id: str,
    card_txn_count_1h: int | None,
    config: OverrideConfig,
) -> Override | None:
    if card_id in config.blocked_card_ids:
        return Override(Action.BLOCK, "blocked_card")
    if device_id is not None and device_id in config.blocked_device_ids:
        return Override(Action.BLOCK, "blocked_device")
    if merchant_id in config.blocked_merchant_ids:
        return Override(Action.BLOCK, "blocked_merchant")
    if card_id in config.allowed_card_ids:
        return Override(Action.APPROVE, "allowed_card")
    cap = config.velocity_cap_1h
    if cap is not None and card_txn_count_1h is not None and card_txn_count_1h >= cap:
        return Override(Action.BLOCK, "velocity_cap_1h")
    return None


def overrides_for_rows(
    card_ids: Sequence[str],
    device_ids: Sequence[str | None],
    merchant_ids: Sequence[str],
    card_txn_count_1h: Sequence[int | None],
    config: OverrideConfig,
) -> tuple[StrArray, StrArray]:
    """Offline: the forced action and the rule that fired per row (``""`` where none applies)."""
    if not len(card_ids) == len(device_ids) == len(merchant_ids) == len(card_txn_count_1h):
        raise ValueError("override inputs must have equal lengths")
    actions: list[str] = []
    rules: list[str] = []
    for card, device, merchant, count in zip(
        card_ids, device_ids, merchant_ids, card_txn_count_1h, strict=True
    ):
        found = override_for(
            card_id=card,
            device_id=device,
            merchant_id=merchant,
            card_txn_count_1h=count,
            config=config,
        )
        actions.append(NO_OVERRIDE if found is None else found.action.value)
        rules.append(NO_OVERRIDE if found is None else found.rule)
    return np.asarray(actions, dtype=np.str_), np.asarray(rules, dtype=np.str_)

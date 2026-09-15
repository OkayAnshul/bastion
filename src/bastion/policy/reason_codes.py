"""Reason codes: the model inputs that raised a transaction's fraud score the most.

LightGBM's ``pred_contrib`` returns TreeSHAP values: each input's exact Shapley contribution to the
model's raw margin (log-odds), plus a bias term, which together sum to the margin. Calibration is
monotonic, so an input that raises the margin also raises the calibrated probability. Reason codes
are the ``k`` largest positive contributions.

They explain the model, not the world. A contribution says the model leaned on an input for this
transaction. It does not say the input caused the fraud, and correlated inputs can share or swap
credit (see learning log 4.2).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from bastion.schemas.tables import ATTRIBUTE_PREFIX
from bastion.training.dataset import FeatureSpec


@dataclass(frozen=True)
class ReasonCode:
    feature: str
    value: float | str | None
    contribution: float  # log-odds this input added to the model's margin
    description: str


_WINDOWS = {"1m": "minute", "10m": "10 minutes", "1h": "hour", "24h": "24 hours", "7d": "7 days"}
_WINDOWED = (
    (re.compile(r"^card_txn_count_(\w+)$"), "Card transactions in the previous {window}"),
    (re.compile(r"^card_amount_sum_(\w+)$"), "Card spend in the previous {window}"),
    (
        re.compile(r"^card_distinct_merchants_(\w+)$"),
        "Distinct merchants on this card in the previous {window}",
    ),
    (
        re.compile(r"^card_distinct_devices_(\w+)$"),
        "Distinct devices on this card in the previous {window}",
    ),
    (
        re.compile(r"^device_txn_count_(\w+)$"),
        "Transactions from this device in the previous {window}",
    ),
    (
        re.compile(r"^device_distinct_cards_(\w+)$"),
        "Distinct cards on this device in the previous {window}",
    ),
)
_LABELS = {
    "amount": "Transaction amount",
    "merchant_category": "Merchant category",
    "card_amount_mean_30d": "Card's mean amount over the previous 30 days",
    "card_amount_std_30d": "Spread of the card's amounts over the previous 30 days",
    "amount_to_card_mean_30d": "Amount relative to the card's 30-day mean",
    "amount_zscore_30d": "Amount z-score against the card's previous 30 days",
    "card_age_days": "Days since the card was first seen",
    "device_age_days": "Days since the device was first seen",
    "card_device_seen_before": "Card used this device before",
    "card_merchant_seen_before": "Card used this merchant before",
    "hour_of_day": "Hour of day (UTC)",
    "day_of_week": "Day of week",
    "card_prior_fraud_labels": "Confirmed frauds on this card so far",
    "merchant_known_labels": "Labelled transactions at this merchant so far",
    "merchant_known_fraud_rate": "Merchant's known fraud rate",
}


def describe(feature: str) -> str:
    """A readable name for a model input; unknown inputs keep their column name."""
    if feature in _LABELS:
        return _LABELS[feature]
    for pattern, template in _WINDOWED:
        match = pattern.match(feature)
        if match and match.group(1) in _WINDOWS:
            return template.format(window=_WINDOWS[match.group(1)])
    if feature.startswith(ATTRIBUTE_PREFIX):
        return f"Vendor attribute {feature.removeprefix(ATTRIBUTE_PREFIX)}"
    return feature


def _plain(value: object) -> float | str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    number = float(value)  # type: ignore[arg-type]
    return None if np.isnan(number) else number


def top_reasons(
    contributions: npt.ArrayLike, spec: FeatureSpec, values: Mapping[str, object], *, k: int
) -> list[ReasonCode]:
    """The ``k`` largest positive contributions of one row; its last value is the bias term."""
    row = np.asarray(contributions, dtype=np.float64).reshape(-1)
    n = len(spec.columns)
    if row.size != n + 1:
        raise ValueError(f"expected {n} contributions plus the bias term, got {row.size}")
    reasons: list[ReasonCode] = []
    for index in np.argsort(-row[:n], kind="stable")[:k]:
        if row[index] <= 0:
            break
        feature = spec.columns[index]
        reasons.append(
            ReasonCode(feature, _plain(values.get(feature)), float(row[index]), describe(feature))
        )
    return reasons

"""What each console page shows, computed from the decision store without any UI code.

The Streamlit pages (``bastion.console.app``) only lay these out, so everything an analyst reads
is unit-tested here: the feed summary, the review queue rows, and the case page with its expected
costs, reason codes and comparison against the card's own recent behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from bastion.evaluation.cost import Action
from bastion.schemas.decisions import DecisionRecord, ReasonCodeOut
from bastion.serving.decision_store import AnalystVerdict, DecisionStore, QueueItem

ACTIONS = (Action.APPROVE.value, Action.REVIEW.value, Action.BLOCK.value)


def _money(value: float, currency: str) -> str:
    return f"{value:,.2f} {currency}"


def _percent(value: float) -> str:
    return f"{100 * value:.2f}%"


@dataclass(frozen=True)
class FeedSummary:
    decisions: int
    actions: dict[str, int]
    block_rate: float | None
    review_rate: float | None
    reviews_today: int
    review_budget: int | None


def feed_summary(store: DecisionStore, *, today: date, review_budget: int | None) -> FeedSummary:
    counts = store.action_counts()
    total = sum(counts.values())
    decided = sum(counts.get(action, 0) for action in ACTIONS)
    return FeedSummary(
        decisions=total,
        actions={action: counts.get(action, 0) for action in ACTIONS},
        block_rate=counts.get(Action.BLOCK.value, 0) / decided if decided else None,
        review_rate=counts.get(Action.REVIEW.value, 0) / decided if decided else None,
        reviews_today=store.reviews_on(today),
        review_budget=review_budget,
    )


def feed_rows(records: list[DecisionRecord]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        decision = record.decision
        top_reason = (
            decision.reason_codes[0].description if decision and decision.reason_codes else ""
        )
        rows.append(
            {
                "scored at": record.scored_at.strftime("%Y-%m-%d %H:%M:%S"),
                "transaction": record.txn_id,
                "event time": record.event_ts.strftime("%Y-%m-%d %H:%M"),
                "amount": _money(record.amount, record.currency),
                "fraud probability": _percent(record.fraud_probability),
                "decision": decision.action.value if decision else "none",
                "override": (decision.override or "") if decision else "",
                "top reason": top_reason,
            }
        )
    return rows


def queue_rows(items: list[QueueItem]) -> list[dict[str, Any]]:
    return [
        {
            "transaction": item.txn_id,
            "event time": item.event_ts[:16].replace("T", " "),
            "card": item.card_id,
            "merchant": item.merchant_id,
            "amount": _money(item.amount, item.currency),
            "fraud probability": _percent(item.fraud_probability),
            "expected fraud loss": _money(item.expected_fraud_loss, item.currency),
        }
        for item in items
    ]


@dataclass(frozen=True)
class CaseView:
    record: DecisionRecord
    facts: list[tuple[str, str]]
    expected_costs: list[tuple[str, str]]  # (action, expected cost), cheapest first
    reasons: list[ReasonCodeOut]
    baseline: list[tuple[str, str, str]]  # (measure, this transaction, the card before it)
    history: list[DecisionRecord]
    verdict: AnalystVerdict | None


def _number(features: dict[str, Any], name: str) -> float | None:
    value = features.get(name)
    return None if value is None else float(value)


def _baseline(record: DecisionRecord) -> list[tuple[str, str, str]]:
    features, cur = record.features, record.currency
    rows: list[tuple[str, str, str]] = []
    mean = _number(features, "card_amount_mean_30d")
    zscore = _number(features, "amount_zscore_30d")
    rows.append(
        (
            "Amount",
            _money(record.amount, cur),
            "no spend in the previous 30 days"
            if mean is None
            else f"mean {_money(mean, cur)} over 30 days"
            + ("" if zscore is None else f" (this one is {zscore:+.1f} standard deviations)"),
        )
    )
    hour, day = _number(features, "card_txn_count_1h"), _number(features, "card_txn_count_24h")
    if hour is not None and day is not None:
        rows.append(
            (
                "Card transactions",
                f"{hour:.0f} in the previous hour",
                f"{day:.0f} in the previous 24 hours ({day / 24:.1f} per hour)",
            )
        )
    devices = _number(features, "card_distinct_devices_24h")
    if devices is not None:
        seen = features.get("card_device_seen_before")  # a boolean, or 0/1 after a JSON round trip
        if seen is None:
            device_note = "no device"
        else:
            device_note = "used by this card before" if seen else "new for this card"
        rows.append(("Device", device_note, f"{devices:.0f} distinct devices in 24 hours"))
    seen_merchant = features.get("card_merchant_seen_before")
    if seen_merchant is not None:
        rows.append(
            (
                "Merchant",
                record.merchant_id,
                "used by this card before" if seen_merchant else "new for this card",
            )
        )
    age = _number(features, "card_age_days")
    if age is not None:
        rows.append(("Card age", f"{age:.1f} days since first seen", ""))
    return rows


def case_view(store: DecisionStore, txn_id: str, *, history: int = 10) -> CaseView | None:
    record = store.get(txn_id)
    if record is None:
        return None
    decision = record.decision
    cur = record.currency
    facts = [
        ("Transaction", record.txn_id),
        ("Event time (UTC)", record.event_ts.strftime("%Y-%m-%d %H:%M:%S")),
        ("Amount", _money(record.amount, cur)),
        ("Card", record.card_id),
        ("Merchant", record.merchant_id),
        ("Device", record.device_id or "none"),
        ("Calibrated fraud probability", _percent(record.fraud_probability)),
        ("Model", record.model_version),
    ]
    expected: list[tuple[str, str]] = []
    reasons: list[ReasonCodeOut] = []
    if decision is not None:
        facts += [
            ("Decision", decision.action.value),
            ("Override", decision.override or "none"),
            ("Review wanted but budget used", "yes" if decision.review_capped else "no"),
        ]
        expected = [
            (action, _money(cost, cur))
            for action, cost in sorted(decision.expected_cost.items(), key=lambda item: item[1])
        ]
        reasons = list(decision.reason_codes)
    return CaseView(
        record=record,
        facts=facts,
        expected_costs=expected,
        reasons=reasons,
        baseline=_baseline(record),
        history=store.card_history(record.card_id, before=record.event_ts, limit=history),
        verdict=store.verdict(txn_id),
    )

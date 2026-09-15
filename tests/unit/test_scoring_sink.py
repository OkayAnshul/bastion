"""The simulator's gateway role: score first, publish afterwards, never drop a payment."""

import json
from datetime import UTC, datetime

import httpx
import pytest

from bastion.schemas.events import LabelEvent, TransactionEvent
from bastion.streaming.decision_sink import decode_decision
from bastion.streaming.replay import ScoringSink


class RecordingSink:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def send_transaction(self, event: TransactionEvent) -> None:
        self.calls.append(f"publish {event.txn_id}")

    def send_label(self, label: LabelEvent) -> None:
        self.calls.append(f"label {label.txn_id}")

    def flush(self) -> None:
        self.calls.append("flush")


def _event(txn_id: str) -> TransactionEvent:
    return TransactionEvent.model_validate(
        {
            "txn_id": txn_id,
            "event_ts": datetime(2026, 6, 1, tzinfo=UTC),
            "card_id": "c1",
            "device_id": None,
            "merchant_id": "m1",
            "merchant_category": "5411",
            "ip": None,
            "amount": 12.5,
            "currency": "USD",
            "channel": "ecom",
            "is_fraud": None,
            "attributes": {},
        }
    )


def test_each_transaction_is_scored_before_it_is_published() -> None:
    inner = RecordingSink()
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body["txn_id"])
        inner.calls.append(f"score {body['txn_id']}")
        action = "block" if body["txn_id"] == "t2" else "approve"
        return httpx.Response(200, json={"decision": {"action": action}})

    sink = ScoringSink(
        "http://scoring:8000/", inner, client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    sink.send_transaction(_event("t1"))
    sink.send_transaction(_event("t2"))
    sink.flush()
    assert inner.calls == ["score t1", "publish t1", "score t2", "publish t2", "flush"]
    assert sink.actions == {"approve": 1, "block": 1}
    assert sink.failures == 0


def test_a_refused_or_failed_score_is_counted_and_the_payment_still_published() -> None:
    inner = RecordingSink()
    responses = iter([httpx.Response(503), httpx.ConnectError("scoring is down")])

    def handler(request: httpx.Request) -> httpx.Response:
        outcome = next(responses)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    sink = ScoringSink(
        "http://scoring:8000", inner, client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    sink.send_transaction(_event("t1"))
    sink.send_transaction(_event("t2"))
    assert sink.failures == 2
    assert inner.calls == ["publish t1", "publish t2"]


def test_empty_decision_messages_are_rejected() -> None:
    with pytest.raises(ValueError, match="empty message"):
        decode_decision(None)

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from bastion.console.views import case_view, feed_rows, feed_summary, queue_rows
from bastion.schemas.decisions import DecisionRecord
from bastion.serving.decision_store import DecisionStore
from tests.unit.test_decision_store import T0, _record


@pytest.fixture
def store(tmp_path: Path) -> DecisionStore:
    store = DecisionStore(tmp_path / "decisions.db")
    store.write(
        [
            _record("a1", probability=0.01, amount=20.0, action="approve"),
            _record("b1", probability=0.97, amount=900.0, action="block"),
            _record("r1", probability=0.30, amount=400.0, action="review"),
            _record(
                "later",
                probability=0.20,
                amount=80.0,
                action="review",
                event_ts=T0 + timedelta(hours=3),
            ),
        ]
    )
    return store


def test_feed_summary_counts_actions_and_todays_reviews(store: DecisionStore) -> None:
    summary = feed_summary(store, today=date(2026, 6, 1), review_budget=200)
    assert summary.decisions == 4
    assert summary.actions == {"approve": 1, "review": 2, "block": 1}
    assert summary.block_rate == pytest.approx(0.25)
    assert summary.review_rate == pytest.approx(0.5)
    assert summary.reviews_today == 2


def test_feed_summary_of_an_empty_store_has_no_rates(tmp_path: Path) -> None:
    summary = feed_summary(
        DecisionStore(tmp_path / "d.db"), today=date(2026, 6, 1), review_budget=None
    )
    assert summary.decisions == 0
    assert summary.block_rate is None


def test_feed_and_queue_rows_are_readable(store: DecisionStore) -> None:
    feed = feed_rows(store.latest())
    assert {row["decision"] for row in feed} == {"approve", "review", "block"}
    assert feed[0]["transaction"] == "later"  # newest first
    queue = queue_rows(store.review_queue())
    assert [row["transaction"] for row in queue] == ["r1", "later"]  # 120 USD before 16 USD
    assert queue[0]["expected fraud loss"] == "120.00 USD"


def test_case_view_explains_the_decision_against_the_card_history(store: DecisionStore) -> None:
    view = case_view(store, "later")
    assert view is not None
    facts = dict(view.facts)
    assert facts["Decision"] == "review"
    assert facts["Calibrated fraud probability"] == "20.00%"
    assert [action for action, _ in view.expected_costs] == ["block", "review", "approve"]
    assert [record.txn_id for record in view.history] == ["r1", "b1", "a1"]
    measures = {row[0] for row in view.baseline}
    assert "Amount" in measures  # the test records carry no 24-hour count, so no velocity row
    assert view.verdict is None


def test_case_view_of_an_unknown_transaction_is_none(store: DecisionStore) -> None:
    assert case_view(store, "missing") is None


def test_case_view_works_for_records_written_before_the_policy_engine(tmp_path: Path) -> None:
    store = DecisionStore(tmp_path / "d.db")
    old: DecisionRecord = _record("old", probability=0.4, amount=50.0, action=None)
    store.write([old])
    view = case_view(store, "old")
    assert view is not None
    assert view.expected_costs == []
    assert view.reasons == []
    assert "Decision" not in dict(view.facts)


def test_verdicts_show_on_the_case(store: DecisionStore) -> None:
    store.record_verdict("r1", "legitimate", analyst="a1", now=datetime(2026, 6, 2, tzinfo=UTC))
    view = case_view(store, "r1")
    assert view is not None
    assert view.verdict is not None
    assert view.verdict.verdict == "legitimate"

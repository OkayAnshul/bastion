"""The analyst console renders every page over a real decision store (Streamlit's AppTest)."""

from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from bastion.config import get_settings
from bastion.serving.decision_store import DecisionStore
from tests.unit.test_decision_store import T0, _record

APP = Path(__file__).resolve().parents[2] / "src" / "bastion" / "console" / "app.py"


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    path = tmp_path / "decisions.db"
    monkeypatch.setenv("BASTION_DECISION_DB_PATH", str(path))
    monkeypatch.setenv("BASTION_SCORING_URL", "http://127.0.0.1:9")  # nothing listens there
    get_settings.cache_clear()
    DecisionStore(path).write(
        [
            _record("a1", probability=0.01, amount=20.0, action="approve"),
            _record("b1", probability=0.97, amount=900.0, action="block"),
            _record(
                "r1",
                probability=0.30,
                amount=400.0,
                action="review",
                event_ts=T0 + timedelta(hours=1),
            ),
        ]
    )
    yield path
    get_settings.cache_clear()


def _page(page: str | None = None, **state: object) -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=30)
    for key, value in state.items():
        app.session_state[key] = value
    if page is not None:
        app.switch_page(page)
    return app.run()


def test_live_decisions_count_actions_and_the_days_reviews(db: Path) -> None:
    app = _page()
    assert not app.exception
    assert [metric.label for metric in app.metric] == [
        "Decisions",
        "Block rate",
        "Review rate",
        "Reviews on 2026-06-01",
    ]
    assert app.metric[0].value == "3"
    assert app.dataframe[0].value["transaction"].tolist() == ["r1", "b1", "a1"]


def test_review_queue_lists_reviews_without_a_verdict(db: Path) -> None:
    app = _page("pages/queue.py")
    assert not app.exception
    assert app.dataframe[0].value["transaction"].tolist() == ["r1"]


def test_an_analyst_verdict_is_saved_from_the_case_page(db: Path) -> None:
    app = _page("pages/case.py", case_txn_id="r1")
    assert not app.exception
    assert "Against the card's own recent behaviour" in [s.value for s in app.subheader]

    app.radio[0].set_value("fraud")
    app.text_input[1].input("analyst-1")
    app.text_area[0].input("same device as last week's chargeback")
    app.button[0].click().run()
    assert not app.exception

    verdict = DecisionStore(db).verdict("r1")
    assert verdict is not None
    assert (verdict.verdict, verdict.analyst) == ("fraud", "analyst-1")
    assert DecisionStore(db).review_queue() == []


def test_model_page_says_when_the_scoring_service_does_not_answer(db: Path) -> None:
    app = _page("pages/model.py")
    assert not app.exception
    assert app.warning[0].value.startswith("No answer from")

"""The operational decision store: SQLite, read by the analyst console (Phase 4).

Decision records arrive either straight from the scoring service
(``BASTION_DECISION_SINK=sqlite``) or from the decisions topic through ``bastion stream decisions``.
Analyst verdicts on reviewed cases are stored next to them. Writes are upserts keyed by transaction
id, so at-least-once delivery cannot duplicate a decision. WAL mode lets the console read while a
writer appends.

This is a single-node operational store for the console's queue and case pages. It is not the
offline store: monitoring and retraining (Phase 6) read the event log.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

from bastion.schemas.decisions import DecisionRecord

Verdict = Literal["fraud", "legitimate"]
_TIMESTAMP = "%Y-%m-%dT%H:%M:%S.%fZ"  # fixed width, so text order is time order

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    txn_id TEXT PRIMARY KEY,
    event_ts TEXT NOT NULL,
    scored_at TEXT NOT NULL,
    card_id TEXT NOT NULL,
    merchant_id TEXT NOT NULL,
    device_id TEXT,
    amount REAL NOT NULL,
    currency TEXT NOT NULL,
    model_version TEXT NOT NULL,
    fraud_probability REAL NOT NULL,
    action TEXT,
    override TEXT,
    review_capped INTEGER,
    expected_fraud_loss REAL NOT NULL,
    record TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS decisions_queue ON decisions (action, expected_fraud_loss DESC);
CREATE INDEX IF NOT EXISTS decisions_scored_at ON decisions (scored_at);
CREATE INDEX IF NOT EXISTS decisions_card ON decisions (card_id, event_ts);
CREATE TABLE IF NOT EXISTS analyst_verdicts (
    txn_id TEXT PRIMARY KEY REFERENCES decisions (txn_id),
    verdict TEXT NOT NULL CHECK (verdict IN ('fraud', 'legitimate')),
    note TEXT NOT NULL DEFAULT '',
    analyst TEXT NOT NULL,
    decided_at TEXT NOT NULL
);
"""
_COLUMNS = (
    "txn_id",
    "event_ts",
    "scored_at",
    "card_id",
    "merchant_id",
    "device_id",
    "amount",
    "currency",
    "model_version",
    "fraud_probability",
    "action",
    "override",
    "review_capped",
    "expected_fraud_loss",
    "record",
)
_UPSERT = (
    f"INSERT INTO decisions ({', '.join(_COLUMNS)}) VALUES ({', '.join('?' for _ in _COLUMNS)})"
    " ON CONFLICT (txn_id) DO UPDATE SET "
    + ", ".join(f"{column} = excluded.{column}" for column in _COLUMNS[1:])
)


def timestamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime(_TIMESTAMP)


def _row(record: DecisionRecord) -> tuple[object, ...]:
    decision = record.decision
    return (
        record.txn_id,
        timestamp(record.event_ts),
        timestamp(record.scored_at),
        record.card_id,
        record.merchant_id,
        record.device_id,
        record.amount,
        record.currency,
        record.model_version,
        record.fraud_probability,
        None if decision is None else decision.action.value,
        None if decision is None else decision.override,
        None if decision is None else int(decision.review_capped),
        record.fraud_probability * record.amount,  # the expected loss of approving it
        record.model_dump_json(),
    )


@dataclass(frozen=True)
class QueueItem:
    txn_id: str
    event_ts: str
    card_id: str
    merchant_id: str
    amount: float
    currency: str
    fraud_probability: float
    expected_fraud_loss: float


@dataclass(frozen=True)
class AnalystVerdict:
    txn_id: str
    verdict: Verdict
    note: str
    analyst: str
    decided_at: str


class DecisionStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._connect() as db:
            db.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # A connection per call: the logger writes from a worker thread and the console reads from
        # its own process, and SQLite connections must not be shared across threads.
        db = sqlite3.connect(self.path, timeout=10)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA foreign_keys=ON")
            with db:  # one transaction: committed on success, rolled back on error
                yield db
        finally:
            db.close()

    def write(self, records: Sequence[DecisionRecord]) -> None:
        if records:
            with self._connect() as db:
                db.executemany(_UPSERT, [_row(record) for record in records])

    def get(self, txn_id: str) -> DecisionRecord | None:
        with self._connect() as db:
            row = db.execute("SELECT record FROM decisions WHERE txn_id = ?", (txn_id,)).fetchone()
        return None if row is None else DecisionRecord.model_validate_json(row["record"])

    def latest(self, limit: int = 50) -> list[DecisionRecord]:
        """The most recently scored decisions, newest first."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT record FROM decisions ORDER BY scored_at DESC, txn_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [DecisionRecord.model_validate_json(row["record"]) for row in rows]

    def review_queue(self, limit: int = 100) -> list[QueueItem]:
        """Reviews without a verdict, largest expected fraud loss first."""
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT d.txn_id, d.event_ts, d.card_id, d.merchant_id, d.amount, d.currency,
                       d.fraud_probability, d.expected_fraud_loss
                FROM decisions AS d LEFT JOIN analyst_verdicts AS v USING (txn_id)
                WHERE d.action = 'review' AND v.txn_id IS NULL
                ORDER BY d.expected_fraud_loss DESC, d.event_ts
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [QueueItem(**dict(row)) for row in rows]

    def record_verdict(
        self,
        txn_id: str,
        verdict: Verdict,
        *,
        analyst: str,
        note: str = "",
        now: datetime | None = None,
    ) -> None:
        """Store (or replace) an analyst's verdict. Unknown transactions raise IntegrityError."""
        decided_at = timestamp(now or datetime.now(UTC))
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO analyst_verdicts (txn_id, verdict, note, analyst, decided_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (txn_id) DO UPDATE SET verdict = excluded.verdict,
                    note = excluded.note, analyst = excluded.analyst,
                    decided_at = excluded.decided_at
                """,
                (txn_id, verdict, note, analyst, decided_at),
            )

    def verdict(self, txn_id: str) -> AnalystVerdict | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM analyst_verdicts WHERE txn_id = ?", (txn_id,)
            ).fetchone()
        return None if row is None else AnalystVerdict(**dict(row))

    def action_counts(self, *, scored_since: datetime | None = None) -> dict[str, int]:
        """Decisions per action; records written before Phase 4 count as ``"none"``."""
        query = "SELECT coalesce(action, 'none') AS action, count(*) AS n FROM decisions"
        parameters: tuple[object, ...] = ()
        if scored_since is not None:
            query += " WHERE scored_at >= ?"
            parameters = (timestamp(scored_since),)
        with self._connect() as db:
            rows = db.execute(query + " GROUP BY 1", parameters).fetchall()
        return {row["action"]: row["n"] for row in rows}

    def reviews_on(self, day: date) -> int:
        """Reviews granted for transactions on a UTC event day: that day's budget use."""
        with self._connect() as db:
            row = db.execute(
                "SELECT count(*) AS n FROM decisions WHERE action = 'review' AND event_ts LIKE ?",
                (f"{day.isoformat()}T%",),
            ).fetchone()
        return int(row["n"])

    def card_history(
        self, card_id: str, *, before: datetime, limit: int = 20
    ) -> list[DecisionRecord]:
        """The card's decisions strictly before ``before``, most recent first."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT record FROM decisions WHERE card_id = ? AND event_ts < ?"
                " ORDER BY event_ts DESC LIMIT ?",
                (card_id, timestamp(before), limit),
            ).fetchall()
        return [DecisionRecord.model_validate_json(row["record"]) for row in rows]


class SqliteDecisionSink:
    """A ``DecisionSink`` that writes straight into the decision store."""

    def __init__(self, path: Path) -> None:
        self.store = DecisionStore(path)

    def write(self, records: Sequence[DecisionRecord]) -> None:
        self.store.write(records)

    def flush(self) -> None:
        return None

from datetime import UTC, datetime, timedelta, timezone

import polars as pl
import pytest
from pydantic import ValidationError

from bastion.schemas.events import LabelEvent, ScoreRequest, TransactionEvent
from bastion.schemas.tables import EventTableError, row_to_event, validate_event_table

# The canonical example from ARCHITECTURE.md §3.1.
EXAMPLE = {
    "txn_id": "0b6f4c1e-5d2a-4f5e-9d61-3f1f7a2a9c10",
    "event_ts": "2026-09-14T10:32:11.412Z",
    "card_id": "c_8821",
    "device_id": "d_331",
    "merchant_id": "m_77",
    "merchant_category": "5814",
    "ip": "49.36.x.x",
    "amount": 2499.00,
    "currency": "INR",
    "channel": "ecom",
    "is_fraud": None,
}


def test_architecture_example_is_a_valid_event() -> None:
    event = TransactionEvent.model_validate(EXAMPLE)
    assert event.event_ts == datetime(2026, 9, 14, 10, 32, 11, 412000, tzinfo=UTC)


def test_event_survives_json_round_trip() -> None:
    # The stream carries JSON; decoding must reproduce the exact event.
    event = TransactionEvent.model_validate({**EXAMPLE, "attributes": {"C1": 2.0, "M4": None}})
    assert TransactionEvent.model_validate_json(event.model_dump_json()) == event


def test_naive_timestamps_are_rejected() -> None:
    with pytest.raises(ValidationError):
        TransactionEvent.model_validate({**EXAMPLE, "event_ts": "2026-09-14T10:32:11"})


def test_offset_timestamps_are_normalised_to_utc() -> None:
    ist = timezone(timedelta(hours=5, minutes=30))
    event = TransactionEvent.model_validate(
        {**EXAMPLE, "event_ts": datetime(2026, 9, 14, 16, 2, 11, tzinfo=ist)}
    )
    assert event.event_ts == datetime(2026, 9, 14, 10, 32, 11, tzinfo=UTC)
    assert event.event_ts.utcoffset() == timedelta(0)


@pytest.mark.parametrize("amount", [0.0, -10.0, float("nan"), float("inf")])
def test_invalid_amounts_are_rejected(amount: float) -> None:
    with pytest.raises(ValidationError):
        TransactionEvent.model_validate({**EXAMPLE, "amount": amount})


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        TransactionEvent.model_validate({**EXAMPLE, "isFraud": 0})


def test_nan_attributes_are_rejected() -> None:
    with pytest.raises(ValidationError, match="non-finite"):
        TransactionEvent.model_validate({**EXAMPLE, "attributes": {"D1": float("nan")}})


def test_score_request_rejects_a_label() -> None:
    ScoreRequest.model_validate(EXAMPLE)  # no label: fine
    with pytest.raises(ValidationError, match="null at scoring time"):
        ScoreRequest.model_validate({**EXAMPLE, "is_fraud": True})


def test_label_cannot_precede_its_transaction() -> None:
    base = {
        "txn_id": "t1",
        "card_id": "c1",
        "merchant_id": "m1",
        "event_ts": "2026-09-14T10:00:00Z",
        "is_fraud": True,
    }
    LabelEvent.model_validate({**base, "label_ts": "2026-09-20T10:00:00Z"})
    with pytest.raises(ValidationError, match="precedes"):
        LabelEvent.model_validate({**base, "label_ts": "2026-09-13T10:00:00Z"})


def _table(**overrides: object) -> pl.DataFrame:
    ts = [datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=i) for i in range(3)]
    columns: dict[str, object] = {
        "txn_id": ["t1", "t2", "t3"],
        "event_ts": pl.Series(ts, dtype=pl.Datetime("ms", "UTC")),
        "card_id": ["c1", "c1", "c2"],
        "device_id": ["d1", None, "d2"],
        "merchant_id": ["m1", "m2", "m1"],
        "merchant_category": ["W", "C", "W"],
        "ip": [None, None, None],
        "amount": [10.0, 250.5, 3.25],
        "currency": ["USD"] * 3,
        "channel": ["ecom"] * 3,
        "is_fraud": [False, True, False],
        "attr_C1": [1.0, float("nan"), 3.0],
    }
    columns.update(overrides)
    return pl.DataFrame(columns, schema_overrides={"ip": pl.String})


def test_valid_table_passes() -> None:
    validate_event_table(_table())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"txn_id": ["t1", "t1", "t3"]}, "not unique"),
        ({"amount": [10.0, 0.0, 3.25]}, "non-positive"),
        ({"card_id": ["c1", None, "c2"]}, "nulls"),
        ({"isFraud": [0, 1, 0]}, "unexpected columns"),
        (
            {
                "event_ts": pl.Series(
                    [datetime(2026, 1, 2, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC)] * 1
                    + [datetime(2026, 1, 3, tzinfo=UTC)],
                    dtype=pl.Datetime("ms", "UTC"),
                )
            },
            "not sorted",
        ),
    ],
)
def test_table_violations_are_reported(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(EventTableError, match=message):
        validate_event_table(_table(**overrides))


def test_naive_event_ts_dtype_is_a_violation() -> None:
    naive = pl.Series([datetime(2026, 1, 1, 0, i) for i in range(3)], dtype=pl.Datetime("ms"))
    with pytest.raises(EventTableError, match="wrong dtypes"):
        validate_event_table(_table(event_ts=naive))


def test_row_to_event_strips_label_and_maps_nan_to_none() -> None:
    row = _table().row(1, named=True)
    event = row_to_event(row)
    assert event.is_fraud is None
    assert event.attributes == {"C1": None}
    assert row_to_event(row, include_label=True).is_fraud is True

from datetime import timedelta

import polars as pl
import pytest

from bastion.data.adapter import ANCHOR, SECONDS_PER_DAY, to_canonical
from bastion.data.ieee_cis import downcast_floats
from bastion.schemas.tables import CANONICAL_SCHEMA

DAY = SECONDS_PER_DAY


def _raw() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Six IEEE-CIS-shaped rows.

    Rows 1 and 2 are one customer: same card fields, and D1 grows with the day, so D1n is equal.
    Row 3 has identical card fields but a different first-seen day, so it is a different customer.
    """
    transactions = pl.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5, 6],
            "isFraud": [0, 1, 0, 0, 1, 0],
            "TransactionDT": [10 * DAY, 12 * DAY, 12 * DAY + 5, 11 * DAY, 11 * DAY, 30 * DAY],
            "TransactionAmt": [49.99, 1200.0, 15.5, 7.25, 300.0, 89.1],
            "ProductCD": ["W", "W", "C", "C", "H", "W"],
            "card1": [9500, 9500, 9500, 4774, 4774, 13926],
            "card2": [111.0, 111.0, 111.0, 555.0, 555.0, None],
            "card3": [150.0] * 6,
            "card4": ["visa", "visa", "visa", "mastercard", "mastercard", "discover"],
            "card5": [226.0] * 6,
            "card6": ["debit", "debit", "debit", "credit", "credit", "credit"],
            "addr1": [315.0, 315.0, 315.0, 204.0, 204.0, None],
            "addr2": [87.0] * 6,
            "D1": [5.0, 7.0, 0.0, 1.0, 1.0, None],
            "P_emaildomain": ["gmail.com", "gmail.com", None, "yahoo.com", None, None],
            "R_emaildomain": [None, None, "gmail.com", "gmail.com", None, None],
            "C1": [1.0, 2.0, 1.0, 1.0, 3.0, 1.0],
        }
    )
    identity = pl.DataFrame(
        {
            "TransactionID": [2, 4, 5],
            "DeviceInfo": ["Windows", None, None],
            "id_30": ["Windows 10", None, None],
            "id_31": ["chrome 63.0", None, "safari"],
            "id_33": ["1920x1080", None, None],
            "DeviceType": ["desktop", "mobile", "mobile"],
        }
    )
    return downcast_floats(transactions), identity


def _by_txn(df: pl.DataFrame) -> dict[str, dict[str, object]]:
    return {row["txn_id"]: row for row in df.iter_rows(named=True)}


@pytest.fixture
def canonical() -> pl.DataFrame:
    return to_canonical(*_raw())


def test_output_follows_the_offline_contract(canonical: pl.DataFrame) -> None:
    # to_canonical validates internally; check the layout explicitly as well.
    assert canonical.height == 6
    assert list(canonical.columns[: len(CANONICAL_SCHEMA)]) == list(CANONICAL_SCHEMA)
    extras = canonical.columns[len(CANONICAL_SCHEMA) :]
    assert all(c.startswith("attr_") for c in extras)
    assert {"attr_C1", "attr_card1", "attr_DeviceType"} <= set(extras)
    assert not {"attr_TransactionDT", "attr_isFraud", "attr_D1n"} & set(extras)


def test_event_time_is_anchor_plus_transaction_dt(canonical: pl.DataFrame) -> None:
    row = _by_txn(canonical)["ieee-3"]
    assert row["event_ts"] == ANCHOR + timedelta(days=12, seconds=5)


def test_rows_are_sorted_by_event_time(canonical: pl.DataFrame) -> None:
    assert canonical["event_ts"].is_sorted()
    assert canonical["txn_id"].to_list()[:3] == ["ieee-1", "ieee-4", "ieee-5"]


def test_labels_are_mapped(canonical: pl.DataFrame) -> None:
    rows = _by_txn(canonical)
    assert rows["ieee-2"]["is_fraud"] is True
    assert rows["ieee-1"]["is_fraud"] is False


def test_card_id_uses_first_seen_day_to_separate_customers(canonical: pl.DataFrame) -> None:
    rows = _by_txn(canonical)
    assert rows["ieee-1"]["card_id"] == rows["ieee-2"]["card_id"]  # D1n = 5 for both
    assert rows["ieee-3"]["card_id"] != rows["ieee-1"]["card_id"]  # same card fields, D1n = 12


def test_missing_key_parts_still_produce_an_id(canonical: pl.DataFrame) -> None:
    row = _by_txn(canonical)["ieee-6"]  # card2, addr1 and D1 are all null
    assert isinstance(row["card_id"], str)
    assert row["card_id"].startswith("c_")


def test_device_is_null_without_a_fingerprint(canonical: pl.DataFrame) -> None:
    rows = _by_txn(canonical)
    assert rows["ieee-1"]["device_id"] is None  # no identity row
    assert rows["ieee-4"]["device_id"] is None  # identity row, but no fingerprint fields
    assert rows["ieee-5"]["device_id"] is not None  # browser alone is a (coarse) fingerprint
    assert rows["ieee-2"]["device_id"] != rows["ieee-5"]["device_id"]


def test_merchant_proxy_groups_product_and_recipient_domain(canonical: pl.DataFrame) -> None:
    rows = _by_txn(canonical)
    assert rows["ieee-3"]["merchant_id"] == rows["ieee-4"]["merchant_id"]  # C + gmail.com
    assert rows["ieee-1"]["merchant_id"] == rows["ieee-6"]["merchant_id"]  # W + no recipient
    assert rows["ieee-1"]["merchant_category"] == "W"


def test_ids_do_not_depend_on_input_row_order() -> None:
    transactions, identity = _raw()
    forward = to_canonical(transactions, identity)
    shuffled = to_canonical(transactions.sample(fraction=1.0, shuffle=True, seed=7), identity)
    columns = ["txn_id", "card_id", "device_id", "merchant_id"]
    assert forward.select(columns).equals(shuffled.select(columns))


def test_identity_join_never_duplicates_transactions() -> None:
    transactions, identity = _raw()
    with pytest.raises(pl.exceptions.ComputeError):
        to_canonical(transactions, pl.concat([identity, identity.head(1)]))

import ipaddress
from datetime import timedelta

import polars as pl
import pytest

from bastion.features.batch import compute_features
from bastion.streaming.synthetic import IP_PREFIX, SyntheticConfig, generate

SMALL = SyntheticConfig(
    days=14,
    n_cards=300,
    n_merchants=40,
    card_testing_attacks_per_day=3.0,
    account_takeovers_per_day=3.0,
)


@pytest.fixture(scope="module")
def events() -> pl.DataFrame:
    return generate(SMALL)  # generate() validates the offline contract itself


def test_same_seed_gives_identical_tables(events: pl.DataFrame) -> None:
    assert generate(SMALL).equals(events)
    assert not generate(SMALL.model_copy(update={"seed": SMALL.seed + 1})).equals(events)


def test_both_classes_are_present(events: pl.DataFrame) -> None:
    n_fraud = events["is_fraud"].sum()
    assert 0 < n_fraud < events.height / 2


def test_every_event_falls_inside_the_configured_period(events: pl.DataFrame) -> None:
    # Regression: attack bursts starting near the end once spilled past it, creating a final
    # day of 1-2 events with a 100% fraud rate.
    end = SMALL.start + timedelta(days=SMALL.days)
    assert events["event_ts"].min() >= SMALL.start  # type: ignore[operator]
    assert events["event_ts"].max() < end  # type: ignore[operator]


def test_timestamps_have_whole_second_resolution(events: pl.DataFrame) -> None:
    # Like IEEE-CIS, so same-timestamp handling gets exercised.
    assert events.filter(pl.col("event_ts").dt.millisecond() != 0).is_empty()


def test_ips_come_only_from_the_documentation_prefix(events: pl.DataFrame) -> None:
    addresses = events["ip"].drop_nulls().unique().to_list()
    assert addresses
    assert all(ipaddress.ip_address(a) in IP_PREFIX for a in addresses)


def test_entity_ids_do_not_reveal_the_label(events: pl.DataFrame) -> None:
    for column, pattern in [
        ("card_id", r"^c_\d{6}$"),
        ("device_id", r"^d_\d{7}$"),
        ("merchant_id", r"^m_\d{5}$"),
    ]:
        ids = events[column].drop_nulls()
        assert ids.str.contains(pattern).all(), column


def test_legitimate_devices_can_be_shared_between_cards(events: pl.DataFrame) -> None:
    # Regression: when only attackers shared devices, "device used by two cards" was a perfect
    # fraud flag and the model learned the generator instead of behaviour.
    shared = (
        events.filter(~pl.col("is_fraud") & pl.col("device_id").is_not_null())
        .group_by("device_id")
        .agg(pl.col("card_id").n_unique().alias("cards"))
        .filter(pl.col("cards") > 1)
    )
    assert shared.height > 0


def test_some_fraud_uses_a_device_the_card_already_uses(events: pl.DataFrame) -> None:
    legit_pairs = events.filter(~pl.col("is_fraud")).select("card_id", "device_id").unique()
    fraud_on_known_device = events.filter(pl.col("is_fraud")).join(
        legit_pairs, on=["card_id", "device_id"], how="semi"
    )
    assert fraud_on_known_device.height > 0


def test_fraud_patterns_are_behavioural(events: pl.DataFrame) -> None:
    # Card testing produces bursts, so fraud should carry more recent same-card activity than
    # legitimate traffic. The signal lives in behaviour, not in a flag.
    features = compute_features(events).join(events.select("txn_id", "is_fraud"), on="txn_id")
    means = features.group_by("is_fraud").agg(pl.col("card_txn_count_10m").mean()).sort("is_fraud")
    legit_mean, fraud_mean = means["card_txn_count_10m"].to_list()
    assert fraud_mean > legit_mean

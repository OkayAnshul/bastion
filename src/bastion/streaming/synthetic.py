"""Synthetic transactions with known, behavioural fraud patterns.

Why this exists: CI cannot download IEEE-CIS, the batch/stream parity test needs a small replayable
log with real entity ids, and Phase 6 needs attacks injected on demand. The generator writes the
same canonical event table as the IEEE-CIS adapter.

The fraud patterns are designed by the author. A model that catches them shows the pipeline works;
it says nothing about catching real fraud.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from bastion.schemas.tables import validate_event_table

SECONDS_PER_DAY = 86_400
MCC_CODES = ("5411", "5812", "5311", "5999", "4814", "5732", "5816", "7995")
PROBE_MCCS = ("5816", "4814")  # digital goods, telecom: cheap targets to test a stolen card
CASH_OUT_MCCS = ("5732", "5999")  # electronics, miscellaneous retail: resellable goods
EMAIL_DOMAINS = ("gmail.com", "yahoo.com", "outlook.com", "icloud.com", "proton.me")

# Relative volume per UTC hour: quiet overnight, busiest in the evening.
HOUR_WEIGHTS = np.array(
    [2, 1, 1, 1, 1, 2, 4, 6, 8, 9, 9, 10, 11, 10, 9, 9, 10, 11, 12, 12, 11, 9, 6, 4],
    dtype=np.float64,
)

# RFC 3849 reserves 2001:db8::/32 for documentation, so these addresses can never belong to anyone.
IP_PREFIX = ipaddress.IPv6Network("2001:db8::/32")
_ATTACKER_IP_OFFSET = 1 << 32  # attacker addresses live in a separate block of the prefix


class SyntheticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = 7
    start: AwareDatetime = datetime(2026, 1, 1, tzinfo=UTC)
    days: int = Field(default=30, ge=1)
    n_cards: int = Field(default=1_000, ge=1)
    n_merchants: int = Field(default=200, ge=len(MCC_CODES))
    txns_per_card_per_day: float = Field(default=0.5, gt=0)
    card_testing_attacks_per_day: float = Field(default=1.0, ge=0)
    account_takeovers_per_day: float = Field(default=1.0, ge=0)
    # A small pool of attacker devices reused across attacks, so fraud rings share infrastructure.
    attacker_devices: int = Field(default=12, ge=1)
    currency: str = Field(default="USD", pattern=r"^[A-Z]{3}$")


@dataclass(frozen=True)
class _Batch:
    seconds: npt.NDArray[np.int64]
    card: npt.NDArray[np.int64]
    merchant: npt.NDArray[np.int64]
    device: npt.NDArray[np.int64]  # -1 = no device
    ip: npt.NDArray[np.int64]
    amount: npt.NDArray[np.float64]
    fraud: npt.NDArray[np.bool_]

    def within(self, max_seconds: int) -> _Batch:
        """Drop events at or after ``max_seconds`` (attack bursts can run past the period's end)."""
        keep = self.seconds < max_seconds
        return _Batch(*(getattr(self, f)[keep] for f in _Batch.__dataclass_fields__))

    @staticmethod
    def concat(parts: list[_Batch]) -> _Batch:
        if not parts:
            empty = np.zeros(0, dtype=np.int64)
            return _Batch(
                empty, empty, empty, empty, empty, np.zeros(0), np.zeros(0, dtype=np.bool_)
            )
        return _Batch(
            *(np.concatenate([getattr(p, f) for p in parts]) for f in _Batch.__dataclass_fields__)
        )


@dataclass(frozen=True)
class _World:
    """Static entities: card spending scales, favourite merchants, merchant categories."""

    card_scale: npt.NDArray[np.float64]
    card_merchants: npt.NDArray[np.int64]
    merchant_popularity: npt.NDArray[np.float64]
    merchant_mcc: npt.NDArray[np.str_]
    probe_merchants: npt.NDArray[np.int64]
    cash_out_merchants: npt.NDArray[np.int64]
    card_email: npt.NDArray[np.str_]


def _world(cfg: SyntheticConfig, rng: np.random.Generator) -> _World:
    # Every category appears at least once: cycle the codes, then shuffle.
    mcc = rng.permutation(np.resize(np.array(MCC_CODES), cfg.n_merchants))
    rank = rng.permutation(cfg.n_merchants)
    popularity = np.asarray(1.0 / (rank + 1.0) ** 1.1, dtype=np.float64)
    popularity /= popularity.sum()
    return _World(
        card_scale=rng.lognormal(mean=3.5, sigma=0.8, size=cfg.n_cards),
        card_merchants=rng.choice(cfg.n_merchants, size=(cfg.n_cards, 5), p=popularity),
        merchant_popularity=popularity,
        merchant_mcc=mcc,
        probe_merchants=np.flatnonzero(np.isin(mcc, PROBE_MCCS)),
        cash_out_merchants=np.flatnonzero(np.isin(mcc, CASH_OUT_MCCS)),
        card_email=rng.choice(np.array(EMAIL_DOMAINS), size=cfg.n_cards),
    )


def _legitimate(cfg: SyntheticConfig, world: _World, rng: np.random.Generator) -> _Batch:
    counts = rng.poisson(cfg.txns_per_card_per_day * cfg.days, size=cfg.n_cards)
    card = np.repeat(np.arange(cfg.n_cards, dtype=np.int64), counts)
    n = card.size
    day = rng.integers(0, cfg.days, n)
    hour = rng.choice(24, size=n, p=HOUR_WEIGHTS / HOUR_WEIGHTS.sum())
    seconds = day * SECONDS_PER_DAY + hour * 3_600 + rng.integers(0, 3_600, n)
    favourite = world.card_merchants[card, rng.integers(0, world.card_merchants.shape[1], n)]
    anywhere = rng.choice(cfg.n_merchants, size=n, p=world.merchant_popularity)
    merchant = np.where(rng.random(n) < 0.8, favourite, anywhere)
    # Each card owns two devices (ids 2*card and 2*card+1); 5% of payments carry no device data.
    device = np.where(rng.random(n) < 0.05, -1, 2 * card + (rng.random(n) < 0.3))
    amount = np.maximum(0.5, np.round(world.card_scale[card] * rng.lognormal(0.0, 0.5, n), 2))
    return _Batch(
        seconds.astype(np.int64),
        card,
        merchant.astype(np.int64),
        device.astype(np.int64),
        card.copy(),  # home IP index = card index
        amount,
        np.zeros(n, dtype=np.bool_),
    )


def _attack(
    cfg: SyntheticConfig,
    rng: np.random.Generator,
    seconds: npt.NDArray[np.int64],
    merchants: npt.NDArray[np.int64],
    amounts: npt.NDArray[np.float64],
    card: int,
) -> _Batch:
    k = seconds.size
    attacker = int(rng.integers(cfg.attacker_devices))
    return _Batch(
        seconds.astype(np.int64),
        np.full(k, card, dtype=np.int64),
        merchants.astype(np.int64),
        np.full(k, 2 * cfg.n_cards + attacker, dtype=np.int64),
        np.full(k, _ATTACKER_IP_OFFSET + attacker, dtype=np.int64),
        amounts,
        np.ones(k, dtype=np.bool_),
    )


def _card_testing(cfg: SyntheticConfig, world: _World, rng: np.random.Generator) -> list[_Batch]:
    """A stolen card probed with a burst of tiny payments, sometimes followed by a cash-out."""
    attacks = []
    for _ in range(int(rng.poisson(cfg.card_testing_attacks_per_day * cfg.days))):
        card = int(rng.integers(cfg.n_cards))
        probes = int(rng.integers(5, 16))
        gaps = rng.integers(0, 60, probes)  # zero gaps create same-second events
        gaps[0] = 0
        seconds = int(rng.integers(0, cfg.days * SECONDS_PER_DAY)) + np.cumsum(gaps)
        merchants = rng.choice(world.probe_merchants, probes)
        amounts = np.round(rng.uniform(0.5, 5.0, probes), 2)
        if rng.random() < 0.5:
            seconds = np.append(seconds, seconds[-1] + int(rng.integers(60, 1_800)))
            merchants = np.append(merchants, rng.choice(world.cash_out_merchants))
            amounts = np.append(
                amounts, round(float(world.card_scale[card]) * rng.uniform(10, 30), 2)
            )
        attacks.append(_attack(cfg, rng, seconds, merchants, amounts, card))
    return attacks


def _account_takeover(
    cfg: SyntheticConfig, world: _World, rng: np.random.Generator
) -> list[_Batch]:
    """Several large purchases within hours, from an attacker device, at resellable goods."""
    attacks = []
    for _ in range(int(rng.poisson(cfg.account_takeovers_per_day * cfg.days))):
        card = int(rng.integers(cfg.n_cards))
        n = int(rng.integers(1, 5))
        start = int(rng.integers(0, cfg.days * SECONDS_PER_DAY))
        seconds = start + np.sort(rng.integers(0, 6 * 3_600, n))
        merchants = rng.choice(world.cash_out_merchants, n)
        amounts = np.round(world.card_scale[card] * rng.uniform(5, 20, n), 2)
        attacks.append(_attack(cfg, rng, seconds, merchants, amounts, card))
    return attacks


def _ip_string(index: int) -> str:
    return str(IP_PREFIX.network_address + index + 1)


def generate(config: SyntheticConfig | None = None) -> pl.DataFrame:
    """Generate a validated canonical event table. The same config always yields the same table."""
    cfg = config or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    world = _world(cfg, rng)
    batch = _Batch.concat(
        [
            _legitimate(cfg, world, rng),
            *_card_testing(cfg, world, rng),
            *_account_takeover(cfg, world, rng),
        ]
    ).within(cfg.days * SECONDS_PER_DAY)

    ip_names = {int(i): _ip_string(int(i)) for i in np.unique(batch.ip)}
    device = pl.col("_device")
    frame = pl.DataFrame(
        {
            "_seconds": batch.seconds,
            "_card": batch.card,
            "_merchant": batch.merchant,
            "_device": batch.device,
            "_ip": batch.ip,
            "amount": batch.amount,
            "is_fraud": batch.fraud,
            "merchant_category": world.merchant_mcc[batch.merchant],
            "attr_email_domain": world.card_email[batch.card],
        }
    ).sort("_seconds", "_card", "_merchant", "amount", maintain_order=True)

    events = frame.select(
        pl.format("syn-{}", pl.int_range(pl.len()).cast(pl.String).str.zfill(8)).alias("txn_id"),
        (pl.lit(cfg.start.astimezone(UTC)) + pl.duration(seconds=pl.col("_seconds")))
        .cast(pl.Datetime("ms", "UTC"))
        .alias("event_ts"),
        pl.format("c_{}", pl.col("_card").cast(pl.String).str.zfill(6)).alias("card_id"),
        pl.when(device >= 0)
        .then(pl.format("d_{}", device.cast(pl.String).str.zfill(7)))
        .alias("device_id"),
        pl.format("m_{}", pl.col("_merchant").cast(pl.String).str.zfill(5)).alias("merchant_id"),
        "merchant_category",
        pl.col("_ip").replace_strict(ip_names, return_dtype=pl.String).alias("ip"),
        "amount",
        pl.lit(cfg.currency).alias("currency"),
        pl.lit("ecom").alias("channel"),
        "is_fraud",
        "attr_email_domain",
        pl.when(device >= 0)
        .then(pl.when(device % 2 == 0).then(pl.lit("mobile")).otherwise(pl.lit("desktop")))
        .alias("attr_device_type"),
    )
    validate_event_table(events)
    return events

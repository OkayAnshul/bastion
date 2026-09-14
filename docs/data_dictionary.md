# Data dictionary

Bastion uses two data sources: the public IEEE-CIS Fraud Detection dataset and its own synthetic
generator. Neither contains real payment information from Bastion's side. IEEE-CIS is used under the
competition's rules and is never redistributed. Measured statistics (row counts, fraud rate,
missingness) are in the EDA report (`docs/results/phase0/eda.md`), not here, so this page never
states a number nobody measured.

---

## 1. IEEE-CIS Fraud Detection (raw)

- **Provider:** Vesta Corporation, via the IEEE Computational Intelligence Society Kaggle
  competition (2019).
- **Files used:** `train_transaction.csv` and `train_identity.csv`, joined on `TransactionID`.
  The `test_*` files have no labels, so Bastion never uses them.
- **Source of the descriptions below:** the provider's own column notes in the competition's
  discussion forum. Search: `IEEE-CIS Fraud Detection "Data Description (Details and Discussion)" Kaggle`.
  Most column meanings are deliberately masked by the provider.

### 1.1 Transaction table

| Column(s) | Provider description | Notes for Bastion |
|---|---|---|
| `TransactionID` | Row identifier | Becomes `txn_id = "ieee-<id>"`. |
| `isFraud` | Label | See **labeling logic** below. It is critical. |
| `TransactionDT` | Time delta from an undisclosed reference datetime, in seconds | Not a real timestamp. Mapped with a fixed anchor (ADR-011). |
| `TransactionAmt` | Payment amount in USD | Monetary loss is reported in USD. |
| `ProductCD` | Product code | Five opaque codes. Used as the merchant-category proxy. |
| `card1`–`card6` | Payment card information (card type, category, issuing bank, country, …) | Part of the `card_id` key. |
| `addr1`, `addr2` | Purchaser billing region and billing country | `addr1` is part of the `card_id` key. |
| `dist1`, `dist2` | Distances between addresses, IP, phone area, … | Passthrough attributes. |
| `P_emaildomain`, `R_emaildomain` | Purchaser and recipient email domains | `R_emaildomain` feeds the merchant proxy. |
| `C1`–`C14` | Counts (e.g. addresses associated with the card); meaning masked | Vendor-computed at authorisation time. |
| `D1`–`D15` | Time deltas (e.g. days since the previous transaction) | `D1` is used to stabilise `card_id` (see ADR-011). |
| `M1`–`M9` | Match flags (e.g. name on card vs address) | Strings `T`/`F` (`M4`: `M0`/`M1`/`M2`). |
| `V1`–`V339` | Vesta-engineered features: ranks, counts, entity relations | Passthrough. Their point-in-time status is unknown (see caveats). |

### 1.2 Identity table (only some transactions have a row)

| Column(s) | Provider description | Notes for Bastion |
|---|---|---|
| `id_01`–`id_38` | Network connection (IP, ISP, proxy) and digital signature (UA, browser, OS) information, masked | `id_30` (OS), `id_31` (browser), `id_33` (screen) feed the device fingerprint. |
| `DeviceType` | `mobile` / `desktop` | Passthrough. |
| `DeviceInfo` | Device model / OS string | Part of the device fingerprint. |

### 1.3 Labeling logic (read this before trusting any metric)

According to the provider, a transaction is labeled fraud when a chargeback is reported on the card.
**Later transactions** directly linked to it through the same user account, email address or billing
address are labeled fraud too. A transaction with no such report within 120 days is labeled legitimate.

Consequences:

1. **Labels propagate forward within an account.** Once an account is marked fraudulent, its later
   transactions share the label. A feature like "this card's historical fraud rate" is therefore
   extremely predictive, *if* it uses labels that would not yet exist at decision time.
   This is exactly the leakage ADR-003 guards against, and it is why labels enter features only
   through the simulated label-arrival time (`label_ts`).
2. **Labels mature slowly.** Real chargeback delays run to weeks. Bastion's split keeps a
   label-maturity gap between training and evaluation (ADR-007).

---

## 2. Canonical event table (Bastion's offline store)

Contract: `bastion.schemas.tables` (validated on every write). One row per transaction, sorted by
`event_ts`.

| Column | Type | Nullable | IEEE-CIS mapping (ADR-011) | Synthetic generator |
|---|---|---|---|---|
| `txn_id` | string | no | `"ieee-" + TransactionID` | UUID-like id |
| `event_ts` | datetime(ms, UTC) | no | anchor `2017-11-30T00:00:00Z` + `TransactionDT` seconds | simulated |
| `card_id` | string | no | stable hash of `card1..card6`, `addr1`, `D1n` (see below) | real entity |
| `device_id` | string | yes | stable hash of `DeviceInfo`, `id_30`, `id_31`, `id_33`; null without an identity row or when all four are null | real entity |
| `merchant_id` | string | no | **proxy:** stable hash of `ProductCD`, `R_emaildomain` | real entity |
| `merchant_category` | string | no | `ProductCD` | simulated MCC |
| `ip` | string | yes | always null (IEEE-CIS exposes no usable IP) | synthetic IP |
| `amount` | float64 | no | `TransactionAmt` | simulated |
| `currency` | string | no | `"USD"` | configurable |
| `channel` | string | no | `"ecom"` (card-not-present e-commerce) | configurable |
| `is_fraud` | boolean | yes | `isFraud == 1` | simulated |
| `attr_*` | varies | yes | every other raw column, prefixed | a few vendor-like fields |

`D1n = floor(TransactionDT / 86400) - D1`, the day the account was first seen. Adding it to the
card key separates different customers who share card and address fields. The idea comes from the
Kaggle community's winning solutions (search: `IEEE-CIS Fraud Detection 1st place solution uid`).
It is an approximation: because the reference time is unknown, `D1n` can be off by one day across
midnight boundaries, which splits one customer into two keys.

### Caveats that limit every downstream claim

- **The entity ids are proxies.** There is no merchant in IEEE-CIS; `merchant_id` is a
  product/recipient-domain segment. The device fingerprint is coarse, so many users share it, and
  most transactions have no device at all. Graph and velocity features built on these ids are
  weaker than they would be on real processor data.
- **Hour of day is relative.** The anchor is arbitrary, so `hour_of_day` is shifted by an unknown
  constant. Trees are insensitive to that shift; human interpretation of it is not.
- **Vendor attributes are opaque.** `C`, `D` and `V` columns are computed by Vesta, and whether all
  of them are strictly point-in-time is not documented. Bastion's own aggregate features are the
  ones whose correctness it can prove; experiments hold the vendor attributes fixed when comparing
  feature pipelines.

"""IEEE-CIS Fraud Detection: download and typed loading of the labelled (train) files.

The competition rules forbid redistribution, so the data lives only under ``data/raw``, which is
gitignored. Column meanings: ``docs/data_dictionary.md``.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import polars as pl

from bastion.log import get_logger

log = get_logger(__name__)

COMPETITION = "ieee-fraud-detection"
TRANSACTION_CSV = "train_transaction.csv"
IDENTITY_CSV = "train_identity.csv"
RULES_URL = "https://www.kaggle.com/competitions/ieee-fraud-detection/rules"

# Columns forced to String: T/F flags, email domains, browser/OS strings. Everything else is
# inferred from the full file, since columns that are sparse early in the file would otherwise be
# mis-inferred.
TRANSACTION_STRING_COLUMNS = (
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    *(f"M{i}" for i in range(1, 10)),
)
IDENTITY_STRING_COLUMNS = (
    "id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29", "id_30",
    "id_31", "id_33", "id_34", "id_35", "id_36", "id_37", "id_38",
    "DeviceType", "DeviceInfo",
)  # fmt: skip

# Money keeps float64; the ~400 anonymised float columns are downcast to float32, which halves
# memory with no loss that matters for tree models.
FULL_PRECISION_COLUMNS = ("TransactionAmt",)


class DatasetUnavailableError(RuntimeError):
    """IEEE-CIS is not on disk and could not be downloaded. The message says how to fix it."""


def _setup_instructions(raw_dir: Path) -> str:
    return (
        "IEEE-CIS is not available locally and could not be downloaded.\n"
        f"  1. Accept the competition rules: {RULES_URL}\n"
        "  2. Create an API token at https://www.kaggle.com/settings/api and either\n"
        "       export KAGGLE_API_TOKEN=<token>   or save it to ~/.kaggle/access_token\n"
        "     (a legacy ~/.kaggle/kaggle.json also works)\n"
        "  3. Re-run `make data`.\n"
        f"Alternatively, place {COMPETITION}.zip in {raw_dir}/ yourself."
    )


def dataset_present(raw_dir: Path) -> bool:
    return all((raw_dir / name).exists() for name in (TRANSACTION_CSV, IDENTITY_CSV))


def download(raw_dir: Path) -> None:
    """Ensure the train CSVs exist in ``raw_dir``: reuse, unzip, or download, in that order."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    if dataset_present(raw_dir):
        log.info("ieee_cis.already_present", raw_dir=str(raw_dir))
        return

    archive = raw_dir / f"{COMPETITION}.zip"
    if not archive.exists():
        _download_archive(raw_dir)

    with zipfile.ZipFile(archive) as zf:
        for name in (TRANSACTION_CSV, IDENTITY_CSV):
            log.info("ieee_cis.extracting", file=name)
            zf.extract(name, raw_dir)


def _download_archive(raw_dir: Path) -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        log.info("ieee_cis.downloading", competition=COMPETITION, dest=str(raw_dir))
        api.competition_download_files(COMPETITION, path=str(raw_dir), quiet=False)
    # The Kaggle client may call sys.exit on authentication failure, hence SystemExit.
    except (Exception, SystemExit) as exc:
        raise DatasetUnavailableError(_setup_instructions(raw_dir)) from exc


def load_raw(raw_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return ``(transactions, identity)`` with explicit string columns and float32 downcasting."""
    if not dataset_present(raw_dir):
        raise DatasetUnavailableError(_setup_instructions(raw_dir))

    transactions = pl.read_csv(
        raw_dir / TRANSACTION_CSV,
        infer_schema_length=None,
        schema_overrides=dict.fromkeys(TRANSACTION_STRING_COLUMNS, pl.String),
    )
    identity = pl.read_csv(
        raw_dir / IDENTITY_CSV,
        infer_schema_length=None,
        schema_overrides=dict.fromkeys(IDENTITY_STRING_COLUMNS, pl.String),
    )
    log.info(
        "ieee_cis.loaded",
        transactions=transactions.shape,
        identity=identity.shape,
    )
    return downcast_floats(transactions), downcast_floats(identity)


def downcast_floats(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col(name).cast(pl.Float32)
        for name, dtype in df.schema.items()
        if dtype == pl.Float64 and name not in FULL_PRECISION_COLUMNS
    )

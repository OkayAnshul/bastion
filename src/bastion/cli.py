"""Command-line entry point: ``bastion <command>``.

Heavy libraries are imported inside commands so ``bastion --help`` stays fast.
"""

from __future__ import annotations

import typer

from bastion.config import get_settings
from bastion.log import configure_logging

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Bastion: fraud and risk decisioning on public and simulated data.",
)


@app.callback()
def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)


data_app = typer.Typer(no_args_is_help=True, help="Dataset download and preparation.")
app.add_typer(data_app, name="data")


@app.command()
def info() -> None:
    """Show the resolved settings."""
    for name, value in get_settings().model_dump().items():
        typer.echo(f"{name:22} {value}")


@data_app.command("download")
def data_download() -> None:
    """Download the IEEE-CIS train files into data/raw (needs a Kaggle token)."""
    from bastion.data.ieee_cis import DatasetUnavailableError, download

    try:
        download(get_settings().raw_dir)
    except DatasetUnavailableError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


@data_app.command("prepare")
def data_prepare() -> None:
    """Build the canonical event table (data/processed/events_ieee.parquet)."""
    from bastion.data.adapter import prepare

    settings = get_settings()
    path = prepare(settings.raw_dir, settings.processed_dir)
    typer.echo(f"wrote {path}")

"""Command-line entry point: ``bastion <command>``.

Heavy libraries are imported inside commands so ``bastion --help`` stays fast.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from bastion.config import get_settings
from bastion.log import configure_logging

if TYPE_CHECKING:
    import polars as pl

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Bastion: fraud and risk decisioning on public and simulated data.",
)
data_app = typer.Typer(no_args_is_help=True, help="Dataset download and preparation.")
baseline_app = typer.Typer(no_args_is_help=True, help="Baselines every model must beat.")
app.add_typer(data_app, name="data")
app.add_typer(baseline_app, name="baseline")

EventsOption = Annotated[
    Path | None,
    typer.Option("--events", help="Canonical event table (default: the prepared IEEE-CIS table)."),
]
OutDirOption = Annotated[
    Path | None, typer.Option("--out-dir", help="Report directory (default: docs/results/phase0).")
]
DatasetOption = Annotated[str, typer.Option("--dataset", help="Dataset name shown in reports.")]


@app.callback()
def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)


@app.command()
def info() -> None:
    """Show the resolved settings."""
    for name, value in get_settings().model_dump().items():
        typer.echo(f"{name:22} {value}")


def _read_events(path: Path | None, columns: list[str] | None = None) -> tuple[pl.DataFrame, Path]:
    import polars as pl

    from bastion.data.adapter import EVENTS_FILE

    source = path or get_settings().processed_dir / EVENTS_FILE
    if not source.exists():
        typer.echo(f"{source} does not exist. Run `make data` first, or pass --events.", err=True)
        raise typer.Exit(code=1)
    return pl.read_parquet(source, columns=columns), source


def _results_dir(out_dir: Path | None) -> Path:
    return out_dir or get_settings().results_dir / "phase0"


# ------------------------------------------------------------------------------ data


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


@data_app.command("synthetic")
def data_synthetic(
    days: int = 183,
    cards: int = 5_000,
    seed: int = 7,
    out: Annotated[Path | None, typer.Option(help="Output parquet path.")] = None,
) -> None:
    """Generate a synthetic canonical event table (for demos and smoke runs, not benchmarks)."""
    from bastion.streaming.synthetic import SyntheticConfig, generate

    events = generate(SyntheticConfig(days=days, n_cards=cards, seed=seed))
    path = out or get_settings().processed_dir / "events_synthetic.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    events.write_parquet(path, compression="zstd")
    typer.echo(f"wrote {path} ({events.height:,} events, {int(events['is_fraud'].sum()):,} fraud)")


# ------------------------------------------------------------------------------ phase 0


@app.command()
def eda(
    events: EventsOption = None,
    out_dir: OutDirOption = None,
    dataset: DatasetOption = "IEEE-CIS",
) -> None:
    """Write the EDA report (markdown, JSON, figures)."""
    from bastion.data.splits import load_split_config
    from bastion.eda import write_eda_report

    frame, source = _read_events(events)
    path = write_eda_report(
        frame, load_split_config(), _results_dir(out_dir), dataset=dataset, source=str(source)
    )
    typer.echo(f"wrote {path}")


@baseline_app.command("rules")
def baseline_rules(
    events: EventsOption = None,
    out_dir: OutDirOption = None,
    dataset: DatasetOption = "IEEE-CIS",
) -> None:
    """Fit rule thresholds on the train window and evaluate the rules baseline on every window."""
    from bastion.data.splits import load_split_config
    from bastion.evaluation.cost import load_cost_model
    from bastion.rules.baseline import load_rule_config
    from bastion.rules.evaluate import EVENT_COLUMNS, run_rules_baseline, write_report

    frame, source = _read_events(events, columns=list(EVENT_COLUMNS))
    costs = load_cost_model()
    result = run_rules_baseline(frame, load_split_config(), costs, load_rule_config())
    path = write_report(result, costs, _results_dir(out_dir), dataset=dataset, source=str(source))
    typer.echo(f"wrote {path}")

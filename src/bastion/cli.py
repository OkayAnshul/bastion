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

    from bastion.features.online import OnlineFeatureStore

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Bastion: fraud and risk decisioning on public and simulated data.",
)
data_app = typer.Typer(no_args_is_help=True, help="Dataset download and preparation.")
baseline_app = typer.Typer(no_args_is_help=True, help="Baselines every model must beat.")
experiment_app = typer.Typer(no_args_is_help=True, help="Experiments with published results.")
stream_app = typer.Typer(
    no_args_is_help=True, help="Streaming path: topics, replay, feature builder."
)
app.add_typer(data_app, name="data")
app.add_typer(baseline_app, name="baseline")
app.add_typer(experiment_app, name="experiment")
app.add_typer(stream_app, name="stream")

EventsOption = Annotated[
    Path | None,
    typer.Option("--events", help="Canonical event table (default: the prepared IEEE-CIS table)."),
]
OutDirOption = Annotated[
    Path | None,
    typer.Option("--out-dir", help="Report directory (default: docs/results/<phase>)."),
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


def _results_dir(out_dir: Path | None, phase: str) -> Path:
    return out_dir or get_settings().results_dir / phase


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
        frame,
        load_split_config(),
        _results_dir(out_dir, "phase0"),
        dataset=dataset,
        source=str(source),
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
    path = write_report(
        result, costs, _results_dir(out_dir, "phase0"), dataset=dataset, source=str(source)
    )
    typer.echo(f"wrote {path}")


# ------------------------------------------------------------------------------ phase 1


@app.command()
def train(
    events: EventsOption = None,
    out_dir: OutDirOption = None,
    dataset: DatasetOption = "IEEE-CIS",
) -> None:
    """Train, calibrate and evaluate LightGBM on point-in-time features; log it all to MLflow."""
    from bastion.data.labels import load_label_delay_config
    from bastion.data.splits import load_split_config
    from bastion.rules.baseline import load_rule_config
    from bastion.training.dataset import load_model_config
    from bastion.training.train import TrainingInputs, train_model

    frame, source = _read_events(events)
    settings = get_settings()
    inputs = TrainingInputs(
        frame,
        load_split_config(),
        load_label_delay_config(),
        load_model_config(),
        load_rule_config(),
        dataset,
        str(source),
    )
    result = train_model(
        inputs,
        results_dir=_results_dir(out_dir, "phase1"),
        artifacts_dir=settings.artifacts_dir,
        tracking_uri=settings.mlflow_tracking_uri,
    )
    typer.echo(
        f"wrote {result.report_path} · MLflow run {result.run_id} · bundle {result.bundle_dir}"
    )


@experiment_app.command("leakage")
def experiment_leakage(
    events: EventsOption = None,
    out_dir: OutDirOption = None,
    dataset: DatasetOption = "IEEE-CIS",
) -> None:
    """Naive vs point-in-time features, shuffled vs temporal split (ADR-003, ADR-007)."""
    from bastion.data.labels import load_label_delay_config
    from bastion.data.splits import load_split_config
    from bastion.training.dataset import load_model_config
    from bastion.training.experiments.leakage import run_leakage_experiment

    frame, source = _read_events(events)
    settings = get_settings()
    results = run_leakage_experiment(
        frame,
        load_split_config(),
        load_label_delay_config(),
        load_model_config(),
        dataset=dataset,
        source=str(source),
        results_dir=_results_dir(out_dir, "phase1"),
        artifacts_dir=settings.artifacts_dir,
        tracking_uri=settings.mlflow_tracking_uri,
    )
    for result in results:
        typer.echo(f"{result.name:22} test PR-AUC {result.pr_auc:.4f}")


# ------------------------------------------------------------------------------ phase 2


def _online_store() -> OnlineFeatureStore:
    import redis

    from bastion.features.online import OnlineFeatureStore
    from bastion.streaming.config import load_streaming_config

    client = redis.Redis.from_url(get_settings().redis_url, decode_responses=True)
    return OnlineFeatureStore(client, load_streaming_config().online_store)


@stream_app.command("topics")
def stream_topics() -> None:
    """Create the Kafka topics if they do not exist yet."""
    from bastion.streaming.config import load_streaming_config
    from bastion.streaming.topics import ensure_topics

    topics = load_streaming_config().topics
    created = ensure_topics(get_settings().kafka_bootstrap, [topics.transactions, topics.labels])
    typer.echo(f"created: {', '.join(created) if created else 'none (all topics exist)'}")


@stream_app.command("replay")
def stream_replay(
    events: EventsOption = None,
    speedup: Annotated[
        float, typer.Option(help="Event-seconds per wall-second (0 = as fast as possible).")
    ] = 0.0,
    limit: Annotated[int | None, typer.Option(help="Stop after this many transactions.")] = None,
    labels: Annotated[
        bool, typer.Option("--labels/--no-labels", help="Also publish simulated label arrivals.")
    ] = True,
    direct: Annotated[
        bool, typer.Option("--direct", help="Write straight into Redis instead of Redpanda.")
    ] = False,
) -> None:
    """Replay an event table in event-time order into Redpanda, or straight into Redis."""
    from bastion.data.labels import label_events, load_label_delay_config
    from bastion.streaming.config import load_streaming_config
    from bastion.streaming.replay import EventSink, KafkaSink, StoreSink, replay, timeline
    from bastion.streaming.topics import ensure_topics

    frame, source = _read_events(events)
    settings = get_settings()
    topics = load_streaming_config().topics
    arrivals = label_events(frame, load_label_delay_config()) if labels else None
    sink: EventSink
    if direct:
        sink = StoreSink(_online_store())
    else:
        ensure_topics(settings.kafka_bootstrap, [topics.transactions, topics.labels])
        sink = KafkaSink(settings.kafka_bootstrap, topics)
    stats = replay(timeline(frame, arrivals), sink, speedup=speedup or None, limit=limit)
    typer.echo(
        f"replayed {stats.transactions:,} transactions and {stats.labels:,} labels from {source}:"
        f" {stats.event_span_ms / 86_400_000:.1f} event-days in {stats.wall_seconds:.1f} s"
    )


@stream_app.command("features")
def stream_features(
    max_messages: Annotated[int | None, typer.Option(help="Stop after this many messages.")] = None,
    idle_timeout: Annotated[
        float | None, typer.Option(help="Stop after this many seconds without messages.")
    ] = None,
) -> None:
    """Run the streaming feature builder: consume events and labels into the online store."""
    from bastion.streaming.config import load_streaming_config
    from bastion.streaming.feature_builder import run_feature_builder
    from bastion.streaming.topics import ensure_topics

    settings = get_settings()
    config = load_streaming_config()
    ensure_topics(settings.kafka_bootstrap, [config.topics.transactions, config.topics.labels])
    processed = run_feature_builder(
        settings.kafka_bootstrap,
        config.consumer_group,
        config.topics,
        _online_store(),
        max_messages=max_messages,
        idle_timeout_s=idle_timeout,
    )
    typer.echo(f"processed {processed:,} messages")

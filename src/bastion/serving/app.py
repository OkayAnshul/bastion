"""FastAPI scoring service: ``POST /v1/score`` (ROADMAP Phase 3).

One request does one pipelined Redis read, evaluates the shared feature definitions, runs the
calibrated model on one thread, queues a decision record without waiting, and returns. Each stage is
timed on the server; the load test measures end to end.

Run with ``bastion serve``. Tests pass a ready ``ServingState`` to ``create_app`` instead of letting
the lifespan load a model and connect to Redis.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

import redis.asyncio
from fastapi import FastAPI, Request, Response

from bastion.config import get_settings
from bastion.features.online import read_snapshot_async
from bastion.schemas.decisions import DecisionRecord, ScoreResponse, StageTimings
from bastion.schemas.events import ScoreRequest
from bastion.serving.decision_log import (
    DecisionLogger,
    DecisionSink,
    JsonlDecisionSink,
    KafkaDecisionSink,
    NullDecisionSink,
)
from bastion.serving.scorer import Scorer


@dataclass
class ServingState:
    scorer: Scorer
    model_version: str
    redis: redis.asyncio.Redis
    prefix: str
    decisions: DecisionLogger


def _decision_sink() -> DecisionSink:
    from bastion.streaming.config import load_streaming_config

    settings = get_settings()
    if settings.decision_sink == "kafka":
        topic = load_streaming_config().topics.decisions.name
        return KafkaDecisionSink(settings.kafka_bootstrap, topic)
    if settings.decision_sink == "jsonl":
        return JsonlDecisionSink(settings.decision_log_path)
    return NullDecisionSink()


def _build_state() -> ServingState:
    from bastion.streaming.config import load_streaming_config
    from bastion.training.bundle import ModelBundle

    settings = get_settings()
    if settings.model_path is not None:
        bundle = ModelBundle.load(settings.model_path)
        version = f"local/{bundle.metadata.get('mlflow_run_id', settings.model_path.name)}"
    else:
        import mlflow

        from bastion.training.registry import load_registered_bundle

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        bundle, version = load_registered_bundle(settings.model_name, settings.model_alias)
    return ServingState(
        scorer=Scorer(bundle),
        model_version=version,
        redis=redis.asyncio.Redis.from_url(settings.redis_url, decode_responses=True),
        prefix=load_streaming_config().online_store.prefix,
        decisions=DecisionLogger(_decision_sink()),
    )


def create_app(state: ServingState | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = state is None
        serving = state if state is not None else await asyncio.to_thread(_build_state)
        app.state.serving = serving
        drain = asyncio.create_task(serving.decisions.run())
        try:
            yield
        finally:
            serving.decisions.close()
            await drain
            if owned:
                await serving.redis.aclose()

    app = FastAPI(title="Bastion scoring service", lifespan=lifespan)

    @app.post("/v1/score", response_model=ScoreResponse)
    async def score(event: ScoreRequest, request: Request) -> ScoreResponse:
        serving: ServingState = request.app.state.serving
        started = time.perf_counter_ns()
        snapshot = await read_snapshot_async(serving.redis, serving.prefix, event)
        fetched = time.perf_counter_ns()
        result = serving.scorer.score(event, snapshot)
        timings = StageTimings(
            redis_ms=(fetched - started) / 1e6,
            features_ms=result.features_ms,
            model_ms=result.model_ms,
            total_ms=(time.perf_counter_ns() - started) / 1e6,
        )
        serving.decisions.log(
            DecisionRecord(
                txn_id=event.txn_id,
                event_ts=event.event_ts,
                scored_at=datetime.now(UTC),
                card_id=event.card_id,
                merchant_id=event.merchant_id,
                device_id=event.device_id,
                amount=event.amount,
                currency=event.currency,
                model_version=serving.model_version,
                fraud_probability=result.fraud_probability,
                raw_score=result.raw_score,
                features=result.features,
                timings=timings,
            )
        )
        return ScoreResponse(
            txn_id=event.txn_id,
            model_version=serving.model_version,
            fraud_probability=result.fraud_probability,
            raw_score=result.raw_score,
            timings=timings,
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request, response: Response) -> dict[str, object]:
        serving: ServingState = request.app.state.serving
        try:
            redis_ok = bool(await serving.redis.ping())
        except (redis.RedisError, OSError):
            redis_ok = False
        response.status_code = 200 if redis_ok else 503
        return {"redis": redis_ok, "model_version": serving.model_version}

    @app.get("/v1/model")
    async def model(request: Request) -> dict[str, object]:
        serving: ServingState = request.app.state.serving
        return {
            "model_version": serving.model_version,
            "spec_fingerprint": serving.scorer.bundle.spec.fingerprint(),
            "inputs": len(serving.scorer.bundle.spec.columns),
            "decision_log": serving.decisions.stats(),
        }

    return app

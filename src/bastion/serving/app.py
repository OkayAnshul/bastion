"""FastAPI scoring service: ``POST /v1/score`` (ROADMAP Phases 3 and 4).

One request does a pipelined Redis read, evaluates the shared feature definitions, and runs the
calibrated model on one thread. It then decides (ADR-005): overrides first, then approve, review or
block by expected cost, with a review only while the day's budget lasts. Reviewed and blocked
transactions get reason codes. The decision record is queued without waiting, and the response
returns. Each stage is timed on the server; the load test measures end to end.

Run with ``bastion serve``. Tests pass a ready ``ServingState`` to ``create_app`` instead of letting
the lifespan load a model and connect to Redis.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import NoReturn

import redis.asyncio
from fastapi import FastAPI, HTTPException, Request, Response

from bastion.config import get_settings
from bastion.evaluation.cost import Action
from bastion.features.online import read_snapshot_async
from bastion.policy.capacity import ReviewCapacity, event_day
from bastion.policy.engine import PolicyParameters, propose
from bastion.policy.overrides import override_for
from bastion.schemas.decisions import (
    DecisionRecord,
    PolicyDecision,
    ReasonCodeOut,
    ScoreResponse,
    StageTimings,
)
from bastion.schemas.events import ScoreRequest
from bastion.serving.decision_log import (
    DecisionLogger,
    DecisionSink,
    JsonlDecisionSink,
    KafkaDecisionSink,
    NullDecisionSink,
)
from bastion.serving.scorer import Scorer

STORE_ERRORS = (redis.ConnectionError, redis.TimeoutError)


@dataclass
class ServingState:
    scorer: Scorer
    model_version: str
    redis: redis.asyncio.Redis
    prefix: str
    decisions: DecisionLogger
    policy: PolicyParameters
    capacity: ReviewCapacity
    store_errors: int = 0  # score requests answered 503 because Redis was unavailable


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
    from bastion.evaluation.cost import load_cost_model
    from bastion.policy.config import load_policy_config
    from bastion.policy.engine import TunedPolicy, serving_policy
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
    policy = serving_policy(
        load_policy_config(),
        load_cost_model(),
        tuned=None if settings.policy_path is None else TunedPolicy.load(settings.policy_path),
        model_version=version,
        spec_fingerprint=bundle.spec.fingerprint(),
    )
    client = redis.asyncio.Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        max_connections=settings.redis_max_connections,
    )
    prefix = load_streaming_config().online_store.prefix
    return ServingState(
        scorer=Scorer(bundle),
        model_version=version,
        redis=client,
        prefix=prefix,
        decisions=DecisionLogger(_decision_sink()),
        policy=policy,
        capacity=ReviewCapacity(client, prefix, policy.reviews_per_day),
    )


def _store_unavailable(serving: ServingState, exc: Exception) -> NoReturn:
    # Includes an exhausted connection pool under overload. A 503 tells the caller to use its
    # fallback, and unlike an unhandled error it doesn't make an already saturated process format a
    # traceback for every rejected request.
    serving.store_errors += 1
    raise HTTPException(status_code=503, detail="online store unavailable") from exc


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
        policy = serving.policy
        started = time.perf_counter_ns()
        try:
            snapshot = await read_snapshot_async(serving.redis, serving.prefix, event)
        except STORE_ERRORS as exc:
            _store_unavailable(serving, exc)
        fetched = time.perf_counter_ns()
        result = serving.scorer.score(event, snapshot)
        scored = time.perf_counter_ns()

        velocity = result.features.get("card_txn_count_1h")
        override = override_for(
            card_id=event.card_id,
            device_id=event.device_id,
            merchant_id=event.merchant_id,
            card_txn_count_1h=None if velocity is None else int(velocity),
            config=policy.overrides,
        )
        proposal = propose(
            result.fraud_probability,
            event.amount,
            override,
            threshold=policy.review_threshold,
            costs=policy.costs,
        )
        action, capped = proposal.action, False
        if action is Action.REVIEW:
            try:
                granted = await serving.capacity.try_reserve(event_day(event.event_ts))
            except STORE_ERRORS as exc:
                _store_unavailable(serving, exc)
            if not granted:
                action, capped = proposal.fallback, True
        decided = time.perf_counter_ns()
        reasons = (
            []
            if action is Action.APPROVE
            else serving.scorer.explain(result, k=policy.reason_codes_top_k)
        )
        finished = time.perf_counter_ns()

        decision = PolicyDecision(
            action=action,
            override=proposal.override_rule,
            review_capped=capped,
            review_benefit=proposal.review_benefit,
            review_threshold=policy.review_threshold,
            expected_cost=proposal.expected_cost,
            reason_codes=[ReasonCodeOut(**asdict(reason)) for reason in reasons],
        )
        timings = StageTimings(
            redis_ms=(fetched - started) / 1e6,
            features_ms=result.features_ms,
            model_ms=result.model_ms,
            policy_ms=(decided - scored) / 1e6,
            explain_ms=(finished - decided) / 1e6,
            total_ms=(finished - started) / 1e6,
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
                decision=decision,
                timings=timings,
            )
        )
        return ScoreResponse(
            txn_id=event.txn_id,
            model_version=serving.model_version,
            fraud_probability=result.fraud_probability,
            raw_score=result.raw_score,
            decision=decision,
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
        policy = serving.policy
        return {
            "model_version": serving.model_version,
            "spec_fingerprint": serving.scorer.bundle.spec.fingerprint(),
            "inputs": len(serving.scorer.bundle.spec.columns),
            "policy": {
                "reviews_per_day": policy.reviews_per_day,
                "review_threshold": policy.review_threshold,
                "tuned": policy.tuned,
                "reason_codes_top_k": policy.reason_codes_top_k,
                "costs": policy.costs.model_dump(),
            },
            "decision_log": serving.decisions.stats(),
            "store_errors": serving.store_errors,
        }

    return app

"""The scoring service end to end: HTTP scores must equal the offline model's scores."""

from pathlib import Path

import fakeredis
import httpx
import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

from bastion.data.labels import label_events
from bastion.evaluation.calibration import IsotonicCalibrator
from bastion.features.batch import compute_features, feature_names
from bastion.features.online import OnlineFeatureStore, OnlineStoreConfig
from bastion.schemas.decisions import DecisionRecord
from bastion.schemas.events import LabelEvent, TransactionEvent
from bastion.schemas.tables import row_to_event
from bastion.serving.app import ServingState, create_app
from bastion.serving.decision_log import DecisionLogger, JsonlDecisionSink
from bastion.serving.scorer import Scorer
from bastion.training.bundle import ModelBundle
from bastion.training.dataset import (
    FeatureSpec,
    build_model_frame,
    fit_feature_spec,
    row_vector,
    to_matrix,
)
from bastion.training.pipeline import fit_lightgbm_on_frame, labels_of, split_rows
from tests.unit.test_training import LABELS, fast_model_config, small_splits, synthetic_events


@pytest.fixture(scope="module")
def trained() -> tuple[ModelBundle, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    events = synthetic_events()
    labels = label_events(events, LABELS)
    config = fast_model_config()
    frame = build_model_frame(
        events,
        compute_features(events, labels, label_strength=config.label_strength),
        small_splits(),
    )
    fitted = fit_lightgbm_on_frame(frame, feature_names(with_labels=True), config)
    calibration = split_rows(frame, "calibration")
    calibrator = IsotonicCalibrator.fit(fitted.scores(calibration), labels_of(calibration))
    bundle = ModelBundle.from_booster(
        fitted.booster, calibrator, fitted.spec, {"label_strength": config.label_strength}
    )
    return bundle, events, labels, frame


def _post(client: TestClient, event: TransactionEvent) -> httpx.Response:
    # Regression: without this header FastAPI reads the body as a string and every request is a
    # 422, which once made the label-rejection test pass for the wrong reason (mistakes.md).
    return client.post(
        "/v1/score",
        content=event.model_dump_json(),
        headers={"Content-Type": "application/json"},
    )


def _serving(bundle: ModelBundle, server: fakeredis.FakeServer, log_path: Path) -> ServingState:
    return ServingState(
        scorer=Scorer(bundle),
        model_version="test/1",
        redis=fakeredis.FakeAsyncRedis(server=server, decode_responses=True),
        prefix="bastion",
        decisions=DecisionLogger(JsonlDecisionSink(log_path), flush_interval_s=0.05),
    )


def test_http_scores_equal_offline_model_scores_exactly(
    trained: tuple[ModelBundle, pl.DataFrame, pl.DataFrame, pl.DataFrame], tmp_path: Path
) -> None:
    bundle, events, labels, frame = trained
    server = fakeredis.FakeServer()
    writer = OnlineFeatureStore(
        fakeredis.FakeRedis(server=server, decode_responses=True),
        OnlineStoreConfig(trim_history=False),
    )
    for row in labels.iter_rows(named=True):
        writer.write_label(LabelEvent.model_validate(row))
    for row in events.iter_rows(named=True):
        writer.write_transaction(row_to_event(row))

    test = split_rows(frame, "test").head(150)
    expected = bundle.predict_proba(test)
    by_id = {row["txn_id"]: row for row in events.iter_rows(named=True)}
    log_path = tmp_path / "decisions.jsonl"
    with TestClient(create_app(_serving(bundle, server, log_path))) as client:
        responses = [_post(client, row_to_event(by_id[txn_id])) for txn_id in test["txn_id"]]
    assert all(r.status_code == 200 for r in responses)
    np.testing.assert_array_equal([r.json()["fraud_probability"] for r in responses], expected)

    # The lifespan drained the queue on shutdown: one decision record per request, nothing dropped.
    records = [
        DecisionRecord.model_validate_json(line) for line in log_path.read_text().splitlines()
    ]
    assert [r.txn_id for r in records] == test["txn_id"].to_list()
    assert all(r.timings.total_ms >= r.timings.redis_ms for r in records)


def test_a_request_carrying_a_label_is_rejected(
    trained: tuple[ModelBundle, pl.DataFrame, pl.DataFrame, pl.DataFrame], tmp_path: Path
) -> None:
    bundle, events, _, _ = trained
    event = row_to_event(events.row(0, named=True), include_label=True)
    with TestClient(
        create_app(_serving(bundle, fakeredis.FakeServer(), tmp_path / "d.jsonl"))
    ) as client:
        response = _post(client, event)
        assert response.status_code == 422
        assert "null at scoring time" in response.text  # rejected for the right reason
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/readyz").status_code == 200


def test_a_model_needing_unknown_inputs_is_refused(
    trained: tuple[ModelBundle, pl.DataFrame, pl.DataFrame, pl.DataFrame],
) -> None:
    bundle = trained[0]
    spec = FeatureSpec((*bundle.spec.numeric, "feature_from_the_future"), bundle.spec.categorical)
    tampered = ModelBundle(bundle.model_text, bundle.calibrator, spec, bundle.metadata)
    with pytest.raises(ValueError, match="feature_from_the_future"):
        Scorer(tampered)


def test_row_encoding_matches_the_training_matrix(
    trained: tuple[ModelBundle, pl.DataFrame, pl.DataFrame, pl.DataFrame],
) -> None:
    """Training/serving skew guard for model inputs: row by row, identical to to_matrix."""
    bundle, _, _, frame = trained
    sample = split_rows(frame, "test").head(300)
    rows = np.vstack([row_vector(row, bundle.spec) for row in sample.iter_rows(named=True)])
    np.testing.assert_array_equal(rows, to_matrix(sample, bundle.spec))


def test_unseen_categories_encode_like_training(
    trained: tuple[ModelBundle, pl.DataFrame, pl.DataFrame, pl.DataFrame],
) -> None:
    frame = trained[3]
    spec = fit_feature_spec(split_rows(frame, "train"), [], ["^email_domain$"], 2)
    probe = pl.DataFrame(
        {
            "amount": [1.0, 2.0],
            "merchant_category": ["never-seen", None],
            "attr_email_domain": ["x.org", None],
        }
    )
    rows = np.vstack([row_vector(row, spec) for row in probe.iter_rows(named=True)])
    np.testing.assert_array_equal(rows, to_matrix(probe, spec))


def test_decision_logger_drops_instead_of_blocking(tmp_path: Path) -> None:
    logger = DecisionLogger(JsonlDecisionSink(tmp_path / "d.jsonl"), max_queue=1)
    record = DecisionRecord.model_validate_json(
        '{"txn_id":"t","event_ts":"2026-01-01T00:00:00Z","scored_at":"2026-01-01T00:00:00Z",'
        '"card_id":"c","merchant_id":"m","device_id":null,"amount":1.0,"currency":"USD",'
        '"model_version":"v","fraud_probability":0.1,"raw_score":0.1,"features":{},'
        '"timings":{"redis_ms":0,"features_ms":0,"model_ms":0,"total_ms":0}}'
    )
    logger.log(record)
    logger.log(record)  # queue already full: counted and dropped, never awaited
    assert logger.dropped == 1

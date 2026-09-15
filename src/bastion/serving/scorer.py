"""From a transaction and its online snapshot to a calibrated fraud probability.

The scorer does no I/O: the service fetches the snapshot first, then calls ``score``, which is pure
CPU work, timed per stage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from bastion.features.batch import DEFAULT_LABEL_STRENGTH, feature_names
from bastion.features.online import OnlineSnapshot, features_from_snapshot
from bastion.schemas.events import TransactionEvent
from bastion.schemas.tables import ATTRIBUTE_PREFIX
from bastion.training.bundle import ModelBundle
from bastion.training.dataset import row_vector

# Columns the service can produce: the event itself, every online feature, and any vendor attribute.
PRODUCIBLE_COLUMNS = frozenset({"amount", "merchant_category", *feature_names(with_labels=True)})


@dataclass(frozen=True)
class ScoreResult:
    fraud_probability: float
    raw_score: float
    features: dict[str, Any]
    features_ms: float
    model_ms: float


class Scorer:
    def __init__(self, bundle: ModelBundle) -> None:
        unknown = [
            column
            for column in bundle.spec.columns
            if column not in PRODUCIBLE_COLUMNS and not column.startswith(ATTRIBUTE_PREFIX)
        ]
        if unknown:
            # ADR-004 tripwire: never serve a model that needs inputs this service cannot build.
            raise ValueError(f"model needs inputs the scoring service cannot compute: {unknown}")
        self.bundle = bundle
        self.label_strength = float(bundle.metadata.get("label_strength", DEFAULT_LABEL_STRENGTH))

    def score(self, event: TransactionEvent, snapshot: OnlineSnapshot) -> ScoreResult:
        started = time.perf_counter_ns()
        features = features_from_snapshot(event, snapshot, label_strength=self.label_strength)
        values: dict[str, object] = {
            "amount": event.amount,
            "merchant_category": event.merchant_category,
            **features,
            **{f"{ATTRIBUTE_PREFIX}{name}": value for name, value in event.attributes.items()},
        }
        row = row_vector(values, self.bundle.spec)
        assembled = time.perf_counter_ns()
        raw = float(self.bundle.raw_scores_from_matrix(row)[0])
        probability = float(self.bundle.calibrator.predict([raw])[0])
        finished = time.perf_counter_ns()
        return ScoreResult(
            fraud_probability=probability,
            raw_score=raw,
            features=features,
            features_ms=(assembled - started) / 1e6,
            model_ms=(finished - assembled) / 1e6,
        )

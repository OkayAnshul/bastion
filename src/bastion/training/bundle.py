"""The model bundle: everything needed to score a transaction, saved as plain files.

``model.txt`` (LightGBM text format), ``calibrator.json``, ``feature_spec.json`` and
``metadata.json``.
No pickles, so a bundle is readable and diffable, and loadable without the training code's classes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import polars as pl

from bastion.evaluation.calibration import Calibrator, calibrator_from_dict
from bastion.training.dataset import FeatureSpec, to_matrix

MODEL_FILE = "model.txt"
CALIBRATOR_FILE = "calibrator.json"
SPEC_FILE = "feature_spec.json"
METADATA_FILE = "metadata.json"


@dataclass(frozen=True)
class ModelBundle:
    model_text: str
    calibrator: Calibrator
    spec: FeatureSpec
    metadata: Mapping[str, Any]
    _booster: lgb.Booster = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_booster", lgb.Booster(model_str=self.model_text))

    @classmethod
    def from_booster(
        cls,
        booster: lgb.Booster,
        calibrator: Calibrator,
        spec: FeatureSpec,
        metadata: Mapping[str, Any],
    ) -> ModelBundle:
        # Keep only the trees up to the early-stopping optimum.
        text = booster.model_to_string(num_iteration=booster.best_iteration)
        return cls(text, calibrator, spec, {**metadata, "spec_fingerprint": spec.fingerprint()})

    def raw_scores(self, frame: pl.DataFrame) -> npt.NDArray[np.float64]:
        return np.asarray(self._booster.predict(to_matrix(frame, self.spec)), dtype=np.float64)

    def predict_proba(self, frame: pl.DataFrame) -> npt.NDArray[np.float64]:
        """Calibrated fraud probabilities (ADR-006): the only scores the policy engine may use."""
        return self.calibrator.predict(self.raw_scores(frame))

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / MODEL_FILE).write_text(self.model_text)
        (directory / CALIBRATOR_FILE).write_text(json.dumps(self.calibrator.to_dict(), indent=2))
        (directory / SPEC_FILE).write_text(json.dumps(self.spec.to_dict(), indent=2))
        (directory / METADATA_FILE).write_text(
            json.dumps(dict(self.metadata), indent=2, sort_keys=True, default=str)
        )
        return directory

    @classmethod
    def load(cls, directory: Path) -> ModelBundle:
        spec = FeatureSpec.from_dict(json.loads((directory / SPEC_FILE).read_text()))
        metadata = json.loads((directory / METADATA_FILE).read_text())
        expected = metadata.get("spec_fingerprint")
        if expected is not None and expected != spec.fingerprint():
            raise ValueError(
                "feature_spec.json does not match the fingerprint recorded at training"
            )
        return cls(
            (directory / MODEL_FILE).read_text(),
            calibrator_from_dict(json.loads((directory / CALIBRATOR_FILE).read_text())),
            spec,
            metadata,
        )

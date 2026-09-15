"""MLflow model registry: register bundles, and load the one an alias points to (ADR-009).

A bundle is logged as an MLflow ``pyfunc`` model so the registry, UI and aliases work normally. The
scoring service does not use the pyfunc runtime, though: it downloads the registered artifacts and
loads the plain-file bundle directly, with no extra wrapper on the request path.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import pandas as pd
import polars as pl
import yaml
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import PythonModel, PythonModelContext  # type: ignore[attr-defined]
from mlflow.tracking import MlflowClient

from bastion.training.bundle import ModelBundle

CHAMPION = "champion"
CHALLENGER = "challenger"
_ARTIFACT_KEY = "bundle"


class BundlePythonModel(PythonModel):
    """pyfunc wrapper so MLflow can load a bundle anywhere; Bastion's own service bypasses it."""

    def load_context(self, context: PythonModelContext) -> None:
        self._bundle = ModelBundle.load(Path(context.artifacts[_ARTIFACT_KEY]))

    def predict(
        self,
        context: PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> Any:
        return self._bundle.predict_proba(pl.from_pandas(model_input))


@dataclass(frozen=True)
class RegisteredModel:
    name: str
    version: str
    alias: str


def register_bundle(bundle_dir: Path, model_name: str) -> RegisteredModel:
    """Log the bundle in the active MLflow run and register it.

    The first registered version becomes ``champion``; later ones become ``challenger`` and must
    earn promotion through shadow evaluation (Phase 6) rather than replace the champion directly.
    """
    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=BundlePythonModel(),
        artifacts={_ARTIFACT_KEY: str(bundle_dir)},
        registered_model_name=model_name,
        pip_requirements=[f"lightgbm=={lgb.__version__}", f"polars=={pl.__version__}"],
    )
    version = str(info.registered_model_version)
    client = MlflowClient()
    alias = CHALLENGER if _has_alias(client, model_name, CHAMPION) else CHAMPION
    client.set_registered_model_alias(model_name, alias, version)
    return RegisteredModel(model_name, version, alias)


def _has_alias(client: MlflowClient, model_name: str, alias: str) -> bool:
    try:
        client.get_model_version_by_alias(model_name, alias)
    except MlflowException:
        return False
    return True


def load_registered_bundle(model_name: str, alias: str) -> tuple[ModelBundle, str]:
    """Download the bundle an alias points to; return it with its "name/version" label."""
    version = MlflowClient().get_model_version_by_alias(model_name, alias)
    destination = Path(tempfile.mkdtemp(prefix="bastion-model-"))
    local = Path(
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{model_name}@{alias}", dst_path=str(destination)
        )
    )
    mlmodel = yaml.safe_load((local / "MLmodel").read_text())
    relative = mlmodel["flavors"]["python_function"]["artifacts"][_ARTIFACT_KEY]["path"]
    return ModelBundle.load(local / relative), f"{model_name}/{version.version}"

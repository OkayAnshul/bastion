"""Model training, calibration, experiment tracking and experiments (Phase 1)."""

import os

# MLflow 3 logs an assistant hint when imported; in pipeline logs it is noise.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

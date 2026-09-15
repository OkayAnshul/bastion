"""What the console pages share: the decision store, the scoring service, and small formatting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import streamlit as st

from bastion.config import get_settings
from bastion.serving.decision_store import DecisionStore

DISCLAIMER = "Public and simulated data only: no real customers, cards or payments."
CASE_KEY = "case_txn_id"
CASE_PAGE = "pages/case.py"  # relative to the console's main script


@st.cache_resource
def _decision_store(path: str) -> DecisionStore:
    return DecisionStore(Path(path))


def store() -> DecisionStore:
    return _decision_store(str(get_settings().decision_db_path))


@st.cache_data(ttl=30, show_spinner=False)
def model_info(url: str) -> dict[str, Any] | None:
    """``GET /v1/model`` from the scoring service, or None when it doesn't answer."""
    try:
        response = httpx.get(f"{url.rstrip('/')}/v1/model", timeout=2.0)
        response.raise_for_status()
    except httpx.HTTPError:
        return None
    info: dict[str, Any] = response.json()
    return info


def scoring_url() -> str:
    return str(st.session_state.get("scoring_url", get_settings().scoring_url))


def rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"

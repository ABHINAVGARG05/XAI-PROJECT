"""Append-only JSONL audit log of every explanation shown (GDPR Art. 22)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from src.paths import EXPLANATION_LOG_PATH


def _to_jsonable(obj):
    """Convert numpy scalars/arrays that the json module cannot serialise."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def log_explanation(
    transaction_id: str,
    shap_values: dict,
    counterfactual: dict,
    fraud_prob: float,
    model_version: str = "xgb_v1",
) -> None:
    """Append one explanation record to ``logs/explanation_log.jsonl``."""
    record = {
        "transaction_id": transaction_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "fraud_prob": float(fraud_prob),
        "shap_values": shap_values,
        "counterfactual": counterfactual,
    }
    EXPLANATION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EXPLANATION_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=_to_jsonable) + "\n")

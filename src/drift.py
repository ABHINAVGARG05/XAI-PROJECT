"""Drift detection via shifts in per-feature SHAP importance."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.explainability import compute_shap_values
from src.paths import DRIFT_LOG_PATH, SEED, TARGET

COHORTS = ("test_early", "test_mid", "test_late")
BASELINE_SAMPLE = 20_000
EPS = 1e-9


def mean_abs_shap(model, X: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """Mean |SHAP| per feature, in ``feature_names`` order."""
    return np.abs(compute_shap_values(model, X[feature_names])).mean(axis=0)


def compute_drift_signal(
    model,
    baseline_X: pd.DataFrame,
    cohort_X: pd.DataFrame,
    feature_names: list,
    threshold: float = 0.20,
    baseline_importance: np.ndarray | None = None,
    cohort_name: str = "cohort",
) -> dict:
    """Relative shift in mean |SHAP| between a baseline and a cohort.

    ``baseline_importance`` may be passed to reuse a precomputed baseline
    across cohorts. Alerts are appended to ``logs/drift_log.json``.

    Returns:
    {
      "feature_shifts": {feature_name: {"baseline": float, "cohort": float, "shift_pct": float}},
      "alerts": [{"feature": str, "shift_pct": float}],
      "cohort_size": int,
    }
    """
    if baseline_importance is None:
        baseline_importance = mean_abs_shap(model, baseline_X, feature_names)
    cohort_importance = mean_abs_shap(model, cohort_X, feature_names)

    feature_shifts = {}
    alerts = []
    for feat, base, cur in zip(feature_names, baseline_importance, cohort_importance):
        shift_pct = float(abs(cur - base) / (base + EPS))
        feature_shifts[feat] = {"baseline": float(base), "cohort": float(cur), "shift_pct": shift_pct}
        if shift_pct > threshold:
            alerts.append({"feature": feat, "shift_pct": shift_pct})
    alerts.sort(key=lambda a: a["shift_pct"], reverse=True)

    signal = {"feature_shifts": feature_shifts, "alerts": alerts, "cohort_size": len(cohort_X)}
    _append_drift_log(cohort_name, signal, threshold)
    return signal


def _append_drift_log(cohort_name: str, signal: dict, threshold: float) -> None:
    """Append one entry per run to the JSON list in ``logs/drift_log.json``."""
    DRIFT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        history = json.loads(DRIFT_LOG_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        history = []
    history.append(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "cohort": cohort_name,
            "cohort_size": signal["cohort_size"],
            "threshold": threshold,
            "alerts": signal["alerts"],
        }
    )
    DRIFT_LOG_PATH.write_text(json.dumps(history, indent=2))


def run_all_cohorts(model, splits: dict, threshold: float = 0.20) -> dict:
    """Drift signal for each temporal test cohort against a train baseline."""
    feature_names = [c for c in splits["train"].columns if c != TARGET]
    train = splits["train"]
    baseline_X = train.sample(min(BASELINE_SAMPLE, len(train)), random_state=SEED)
    baseline_importance = mean_abs_shap(model, baseline_X, feature_names)

    return {
        name: compute_drift_signal(
            model,
            baseline_X,
            splits[name],
            feature_names,
            threshold=threshold,
            baseline_importance=baseline_importance,
            cohort_name=name,
        )
        for name in COHORTS
    }

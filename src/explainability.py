"""SHAP attributions and DiCE counterfactual explanations."""

from __future__ import annotations

import dice_ml
import numpy as np
import pandas as pd
import shap

from src.paths import SEED, TARGET

CF_TOLERANCE = 1e-6


def get_explainer(model) -> shap.TreeExplainer:
    """Exact TreeSHAP explainer (log-odds output for XGBoost)."""
    return shap.TreeExplainer(model)


def base_value(model) -> float:
    """Expected model output (log-odds) over the background distribution."""
    expected = get_explainer(model).expected_value
    return float(np.ravel(expected)[-1])


def compute_shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """Exact Shapley values, shape (n_rows, n_features)."""
    values = get_explainer(model).shap_values(X)
    if isinstance(values, list):  # some shap versions return one array per class
        values = values[-1]
    return np.asarray(values)


def global_importance(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Mean |SHAP| per feature, sorted descending."""
    return (
        pd.DataFrame(
            {"feature": feature_names, "mean_abs_shap": np.abs(shap_values).mean(axis=0)}
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def local_waterfall_data(
    shap_values_row: np.ndarray,
    base_value: float,
    feature_names: list[str],
    top_k: int | None = None,
) -> dict:
    """Per-feature contributions for one row, ordered by |SHAP| descending.

    With ``top_k`` the remaining features are folded into a single
    "other features" entry so the contributions still sum to the prediction.
    """
    row = np.asarray(shap_values_row, dtype=float)
    order = np.argsort(-np.abs(row))
    features = [feature_names[i] for i in order]
    vals = row[order].tolist()

    if top_k is not None and len(features) > top_k:
        rest = float(sum(vals[top_k:]))
        features = features[:top_k] + [f"{len(vals) - top_k} other features"]
        vals = vals[:top_k] + [rest]

    return {
        "features": features,
        "shap_vals": vals,
        "base_value": float(base_value),
        "prediction": float(base_value + row.sum()),
    }


class ThresholdedModel:
    """Re-centres a binary classifier so its decision threshold maps to 0.5.

    DiCE picks the original class with argmax(predict_proba) and judges a flip
    against a 0.5 probability cutoff, so wrapping ``predict`` alone is not
    enough. A monotone piecewise-linear remap of P(fraud) keeps the ranking and
    makes both rules agree with the tuned ``threshold``.
    """

    def __init__(self, model, threshold: float):
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        self.model = model
        self.threshold = threshold
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X) -> np.ndarray:
        p = self.model.predict_proba(X)[:, 1]
        t = self.threshold
        p_adj = np.where(p < t, p / (2 * t), 0.5 + (p - t) / (2 * (1 - t)))
        return np.column_stack([1 - p_adj, p_adj])

    def predict(self, X) -> np.ndarray:
        return (self.model.predict_proba(X)[:, 1] >= self.threshold).astype(int)


def _with_label(model, X_train: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """DiCE needs the outcome column; fall back to model labels if it is absent."""
    if TARGET in X_train.columns:
        return X_train[feature_names + [TARGET]]
    labelled = X_train[feature_names].copy()
    labelled[TARGET] = model.predict(labelled)
    return labelled


def generate_counterfactual(
    model,
    X_train: pd.DataFrame,
    instance_df: pd.DataFrame,
    feature_names: list[str],
    threshold: float = 0.5,
) -> dict:
    """One DiCE counterfactual flipping the predicted class of ``instance_df``.

    The class is decided at ``threshold`` (the val-tuned decision threshold),
    so the counterfactual explains the same decision the evaluated system makes.
    Returns ``{feature: (original_val, cf_val)}`` for changed features only, or
    ``{"error": ...}`` if DiCE cannot find one. Never raises.
    """
    try:
        decision_model = ThresholdedModel(model, threshold)
        data = dice_ml.Data(
            dataframe=_with_label(decision_model, X_train, feature_names),
            continuous_features=list(feature_names),
            outcome_name=TARGET,
        )
        dice = dice_ml.Dice(
            data, dice_ml.Model(model=decision_model, backend="sklearn"), method="random"
        )
        query = instance_df[feature_names].iloc[[0]]
        result = dice.generate_counterfactuals(
            query, total_CFs=1, desired_class="opposite", random_seed=SEED, verbose=False
        )
        cfs = result.cf_examples_list[0].final_cfs_df
        if cfs is None or cfs.empty:
            return {"error": "counterfactual not found"}

        original, cf = query.iloc[0], cfs.iloc[0]
        changed = {
            feat: (float(original[feat]), float(cf[feat]))
            for feat in feature_names
            if abs(float(cf[feat]) - float(original[feat])) > CF_TOLERANCE
        }
        return changed or {"error": "counterfactual not found"}
    except Exception:  # DiCE fails in many ways; the dashboard must survive all of them
        return {"error": "counterfactual not found"}

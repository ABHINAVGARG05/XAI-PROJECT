"""Train an XGBoost fraud classifier with BorderlineSMOTE + class weighting."""

from __future__ import annotations

import json

import joblib
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE
from xgboost import XGBClassifier

from src.paths import FEATURE_NAMES_PATH, MODEL_PATH, MODELS_DIR, SEED
from src.preprocess import load_and_split, split_xy


def resample_train(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Oversample the minority class near the decision border (train split only)."""
    X_res, y_res = BorderlineSMOTE(random_state=SEED).fit_resample(X, y)
    return X_res, y_res


def compute_scale_pos_weight(y: pd.Series) -> float:
    """count(negatives) / count(positives) on the labels the model is fitted on."""
    positives = int((y == 1).sum())
    return float((y == 0).sum() / max(positives, 1))


def train_model(splits: dict) -> XGBClassifier:
    """Fit the classifier on resampled train data, early-stopping on val PR-AUC."""
    X_train, y_train = split_xy(splits["train"])
    X_val, y_val = split_xy(splits["val"])

    X_res, y_res = resample_train(X_train, y_train)
    scale_pos_weight = compute_scale_pos_weight(y_res)
    print(
        f"Train rows {len(X_train):,} -> {len(X_res):,} after BorderlineSMOTE; "
        f"scale_pos_weight={scale_pos_weight:.3f}"
    )

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=SEED,
        device="cpu",
    )
    model.fit(X_res, y_res, eval_set=[(X_val, y_val)], verbose=False)
    print(f"Best iteration {model.best_iteration}, val aucpr {model.best_score:.4f}")

    save_model(model, list(X_train.columns))
    return model


def save_model(model: XGBClassifier, feature_names: list[str]) -> None:
    """Persist the model and its feature order under models/."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    FEATURE_NAMES_PATH.write_text(json.dumps(feature_names, indent=2))


def load_model() -> tuple[XGBClassifier, list[str]]:
    """Load the persisted model and feature names."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found. Run: python -m src.train")
    return joblib.load(MODEL_PATH), json.loads(FEATURE_NAMES_PATH.read_text())


if __name__ == "__main__":
    train_model(load_and_split())

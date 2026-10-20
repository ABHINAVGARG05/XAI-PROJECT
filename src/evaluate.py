"""Cohort-wise evaluation at a validation-tuned decision threshold."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_curve,
)

from src.paths import EVAL_REPORT_PATH
from src.preprocess import split_xy

EVAL_SPLITS = {
    "val": "val",
    "test_early": "test_early",
    "test_mid": "test_mid",
    "test_late": "test_late",
    "test_full": "test",
}
TARGET_FPR = 0.01


def find_optimal_threshold(model, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """Probability threshold maximising F1 on the validation set."""
    proba = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, proba)
    # The last precision/recall point has no threshold attached.
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return float(thresholds[int(np.argmax(f1))])


def recall_at_fpr(y_true: pd.Series, proba: np.ndarray, max_fpr: float = TARGET_FPR) -> float:
    """Highest recall (TPR) achievable while keeping FPR <= ``max_fpr``."""
    fpr, tpr, _ = roc_curve(y_true, proba)
    return float(tpr[fpr <= max_fpr].max())


def compute_metrics(y_true: pd.Series, proba: np.ndarray, threshold: float) -> dict:
    """PR-AUC, F1, Recall@1%FPR and MCC for one split."""
    pred = (proba >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, proba)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "recall_at_1pct_fpr": recall_at_fpr(y_true, proba),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "n": int(len(y_true)),
        "n_fraud": int(y_true.sum()),
    }


def evaluate_all(model, splits: dict, threshold: float) -> dict:
    """Metrics for val, each test cohort and the full test set; saved as JSON."""
    metrics = {}
    for label, key in EVAL_SPLITS.items():
        X, y = split_xy(splits[key])
        metrics[label] = compute_metrics(y, model.predict_proba(X)[:, 1], threshold)

    report = {"threshold": threshold, "metrics": metrics}
    EVAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_REPORT_PATH.write_text(json.dumps(report, indent=2))
    return report


def format_markdown_table(report: dict) -> str:
    """Render the report as a markdown table."""
    lines = [
        f"Decision threshold (tuned on val): {report['threshold']:.4f}",
        "",
        "| Split | N | Fraud | PR-AUC | F1 | Recall@1%FPR | MCC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, m in report["metrics"].items():
        lines.append(
            f"| {split} | {m['n']:,} | {m['n_fraud']} | {m['pr_auc']:.4f} | {m['f1']:.4f} "
            f"| {m['recall_at_1pct_fpr']:.4f} | {m['mcc']:.4f} |"
        )
    return "\n".join(lines)


def run_evaluation(model, splits: dict) -> dict:
    """Tune the threshold on val, evaluate every split and print the table."""
    threshold = find_optimal_threshold(model, *split_xy(splits["val"]))
    report = evaluate_all(model, splits, threshold)
    print(format_markdown_table(report))
    return report


if __name__ == "__main__":
    from src.preprocess import load_and_split
    from src.train import load_model

    run_evaluation(load_model()[0], load_and_split())

# XCCFDS — Explainable Credit Card Fraud Detection System

A weekend research project: an XGBoost fraud detector on the
[ULB credit card dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
with three explainability additions:

1. **Drift-aware SHAP monitoring** — compares mean |SHAP| per feature between the
   training baseline and successive temporal test cohorts, and alerts when a
   feature's importance shifts by more than 20%.
2. **DiCE counterfactuals** — "what would need to change for this transaction to
   be classified differently?"
3. **GDPR-style audit logging** — every explanation shown to a user can be appended
   to an immutable JSONL log (Art. 22 "right to explanation").

## Setup

```bash
uv venv
uv pip install -e .
```

Download `creditcard.csv` (284,807 rows) into `data/raw/`, e.g. with the Kaggle CLI:

```bash
kaggle datasets download mlg-ulb/creditcardfraud -p data/raw --unzip
```

## Usage

```bash
# Full pipeline: preprocess -> train -> evaluate -> print report
uv run python -m src.main

# Individual stages
uv run python -m src.train
uv run python -m src.evaluate

# Dashboard
uv run streamlit run app.py
```

## Pipeline

| Stage | Module | Notes |
|---|---|---|
| Preprocess | `src/preprocess.py` | `Amount_log = log1p(Amount)`, cyclic hour-of-day encoding, temporal 70/15/15 split (no shuffle), test split into early/mid/late cohorts |
| Train | `src/train.py` | BorderlineSMOTE on train only, XGBoost with `scale_pos_weight`, early stopping on val PR-AUC |
| Explain | `src/explainability.py` | Exact TreeSHAP, global importance, waterfall data, DiCE counterfactuals |
| Drift | `src/drift.py` | Per-feature relative shift in mean \|SHAP\|, alerts appended to `logs/drift_log.json` |
| Evaluate | `src/evaluate.py` | PR-AUC, F1 @ val-tuned threshold, Recall @ 1% FPR, MCC per cohort |
| Audit log | `src/logger.py` | Append-only `logs/explanation_log.jsonl` |
| Dashboard | `app.py` | Streamlit: Global View, Transaction Explainer, Drift Monitor |

## Design notes

- **Temporal split.** Rows are kept in `Time` order so validation and test
  always lie in the future relative to training — no leakage from shuffling.
- **Imbalance handling.** `scale_pos_weight` is computed on the labels the model
  is actually fitted on (post-SMOTE). With BorderlineSMOTE's default
  `sampling_strategy="auto"` the classes are balanced, so the weight is ≈ 1;
  computing it on the raw labels would double-correct (≈ 578× oversampling *and*
  ≈ 578× up-weighting) and wreck precision.
- **SHAP units.** TreeSHAP values for XGBoost are in log-odds (margin) space;
  the waterfall chart starts at the model's base value in log-odds.
- **Drift baseline.** The SHAP baseline is a seeded 20k-row sample of the
  (un-resampled) training split to keep the dashboard responsive.

## Layout

```
data/raw/            creditcard.csv (not committed)
data/processed/      parquet splits (generated)
models/              xgb_model.joblib, feature_names.json (generated)
results/             evaluation_report.json
logs/                explanation_log.jsonl, drift_log.json (generated)
src/                 pipeline modules
app.py               Streamlit dashboard
```

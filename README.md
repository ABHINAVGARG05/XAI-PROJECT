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

## Demo walkthrough (≈5 minutes)

Open the **Live Demo** tab (`uv run streamlit run app.py`).

1. **🚨 Fraud caught** — the model blocks the payment (87% vs an 85.3% threshold).
   "Why it looks suspicious" lists V14, V12 and V10 at values lower than
   virtually all normal transactions.
2. **What if?** — open the expander and drag **V14** to the right. The fraud
   probability falls live, the decision flips to APPROVED and the reasons
   re-rank: the explanation tracks the model, not a canned story.
3. **What would change the decision?** — click the DiCE button. It answers
   with the smallest change that flips the outcome, e.g. *"would have been
   BLOCKED if V14 were −15.70 instead of 5.28"*.
4. **⚖️ Borderline call** — a legitimate payment scoring ~76%: suspicious,
   but under the threshold, so it is approved. Shows why the threshold matters.
5. **🕳️ Fraud missed** — an honest failure case: a real fraud the model
   approves, and the explanation shows which features made it look normal.
6. **🧾 Audit trail** — record the decision; the stored JSON record (SHAP
   values, counterfactual, probability, model version, UTC timestamp) appears
   underneath. This is the GDPR Art. 22 "right to explanation" evidence.

Then use **Global View** (what the model relies on overall), **Transaction
Explainer** (full SHAP waterfall for any test transaction) and **Drift
Monitor** (feature-importance shift across time cohorts) for the research
results.

> V1–V28 are PCA components anonymised by the dataset provider, so the
> explanations can say a value is *unusual*, not what it means in banking terms.

## Pipeline

| Stage | Module | Notes |
|---|---|---|
| Preprocess | `src/preprocess.py` | `Amount_log = log1p(Amount)`, cyclic hour-of-day encoding, temporal 70/15/15 split (no shuffle), test split into early/mid/late cohorts |
| Train | `src/train.py` | BorderlineSMOTE on train only, XGBoost with `scale_pos_weight`, early stopping on val PR-AUC |
| Explain | `src/explainability.py` | Exact TreeSHAP, global importance, waterfall data, DiCE counterfactuals |
| Drift | `src/drift.py` | Per-feature relative shift in mean \|SHAP\|, alerts appended to `logs/drift_log.json` |
| Evaluate | `src/evaluate.py` | PR-AUC, F1 @ val-tuned threshold, Recall @ 1% FPR, MCC per cohort |
| Audit log | `src/logger.py` | Append-only `logs/explanation_log.jsonl` |
| Narrative | `src/narrative.py` | SHAP and counterfactuals turned into plain-English sentences |
| Dashboard | `app.py`, `src/demo_view.py`, `src/charts.py` | Streamlit: Live Demo, Global View, Transaction Explainer, Drift Monitor |

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

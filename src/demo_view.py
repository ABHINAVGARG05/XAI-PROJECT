"""Live Demo tab: one transaction, the AI's decision, and why — in plain English."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from src.charts import reasons_chart
from src.explainability import compute_shap_values, generate_counterfactual
from src.logger import log_explanation
from src.narrative import (
    TIME_FEATURES,
    build_reference,
    counterfactual_sentences,
    decision_label,
    display_name,
    explain_reasons,
)
from src.paths import EXPLANATION_LOG_PATH, TARGET

N_SLIDERS = 3


@st.cache_data
def get_reference(train: pd.DataFrame, features: tuple[str, ...]) -> dict[str, np.ndarray]:
    return build_reference(train, list(features))


@st.cache_data(show_spinner="Asking DiCE what would change the decision...")
def get_counterfactual(_model, _train: pd.DataFrame, instance: pd.DataFrame, features, threshold):
    return generate_counterfactual(_model, _train, instance, list(features), threshold)


def pick_examples(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, int]:
    """Representative test transactions for each outcome the demo should show."""

    def median_of(mask: np.ndarray) -> int | None:
        idx = np.flatnonzero(mask)
        return int(idx[np.argsort(prob[idx])[len(idx) // 2]]) if len(idx) else None

    def extreme_of(mask: np.ndarray, highest: bool) -> int | None:
        idx = np.flatnonzero(mask)
        return int(idx[np.argmax(prob[idx]) if highest else np.argmin(prob[idx])]) if len(idx) else None

    def first_found(*candidates: int | None) -> int | None:
        return next((c for c in candidates if c is not None), None)

    flagged = prob >= threshold
    examples = {
        "🚨 Fraud caught": median_of((y == 1) & flagged),
        "✅ Normal purchase": median_of(y == 0),
        # A legitimate payment in the grey zone: suspicious-looking but under the threshold.
        "⚖️ Borderline call": first_found(
            median_of((y == 0) & ~flagged & (prob >= 0.5)), extreme_of((y == 0) & ~flagged, highest=True)
        ),
        "🕳️ Fraud missed": median_of((y == 1) & ~flagged),
    }
    return {k: v for k, v in examples.items() if v is not None}


def what_if_sliders(row: pd.DataFrame, features: list[str], shap_row: np.ndarray, reference, key: str) -> pd.DataFrame:
    """Sliders for the most influential features plus the amount; returns the edited row."""
    ranked = [features[i] for i in np.argsort(-np.abs(shap_row)) if features[i] not in TIME_FEATURES]
    editable = ranked[:N_SLIDERS] + (["Amount_log"] if "Amount_log" not in ranked[:N_SLIDERS] else [])
    edited = row.copy()
    cols = st.columns(len(editable))
    for col, feat in zip(cols, editable):
        value = float(row[feat].iloc[0])
        if feat == "Amount_log":
            euros = float(np.expm1(value))
            new = col.slider("Amount (€)", 0.0, max(2000.0, euros * 1.5), euros, key=f"{key}_{feat}")
            edited[feat] = np.log1p(new)
            continue
        ref = reference[feat]
        lo = min(float(np.quantile(ref, 0.001)), value)
        hi = max(float(np.quantile(ref, 0.999)), value)
        pad = 0.1 * (hi - lo)
        edited[feat] = col.slider(feat, lo - pad, hi + pad, value, key=f"{key}_{feat}")
    return edited




def decision_card(prob: float, threshold: float, original_prob: float | None, truth: str | None) -> None:
    label = decision_label(prob, threshold)
    (st.error if label == "BLOCKED" else st.success)(
        f"### {'🚨' if label == 'BLOCKED' else '✅'} Decision: {label}"
    )
    c1, c2, c3 = st.columns(3)
    delta = None if original_prob is None else f"{(prob - original_prob) * 100:+.1f} pts"
    c1.metric("Fraud probability", f"{prob:.1%}", delta, delta_color="inverse")
    c2.metric("Blocking threshold", f"{threshold:.1%}")
    c3.metric("Actual outcome", truth or "— (edited)")


def demo_tab(model, splits: dict, features: list[str], threshold: float, test_shap: np.ndarray) -> None:
    test, train = splits["test"], splits["train"]
    X_test = test[features]
    prob_all = model.predict_proba(X_test)[:, 1]
    examples = pick_examples(test[TARGET].to_numpy(), prob_all, threshold)
    reference = get_reference(train, tuple(features))

    st.markdown(
        "Pick a transaction. The model decides whether to **block** it, then explains **why** "
        "and **what would have changed the decision**. _V1–V28 are anonymised (PCA) bank features._"
    )
    cols = st.columns(len(examples))
    for col, (name, pos) in zip(cols, examples.items()):
        if col.button(name, width="stretch"):
            st.session_state["demo_idx"] = pos
    idx = st.session_state.setdefault("demo_idx", next(iter(examples.values())))

    row = X_test.iloc[[idx]]
    with st.expander("🎛️ What if? Change the transaction and watch the decision update"):
        edited = what_if_sliders(row, features, test_shap[idx], reference, key=f"demo_{idx}")
    is_edited = not np.allclose(edited.to_numpy(), row.to_numpy())

    render_decision(
        model,
        edited,
        features,
        threshold,
        reference,
        train,
        transaction_id=f"demo_test_{test.index[idx]}" + ("_whatif" if is_edited else ""),
        key="demo",
        shap_row=None if is_edited else test_shap[idx],
        truth=None if is_edited else ("Fraud" if test[TARGET].iloc[idx] == 1 else "Legitimate"),
        original_prob=float(prob_all[idx]) if is_edited else None,
    )


def render_decision(
    model,
    row: pd.DataFrame,
    features: list[str],
    threshold: float,
    reference: dict[str, np.ndarray],
    train: pd.DataFrame,
    *,
    transaction_id: str,
    key: str,
    shap_row: np.ndarray | None = None,
    truth: str | None = None,
    original_prob: float | None = None,
) -> None:
    """Decision card, plain-English reasons, DiCE counterfactual and audit logging for one row."""
    prob = float(model.predict_proba(row)[0, 1])
    if shap_row is None:
        shap_row = compute_shap_values(model, row)[0]
    decision_card(prob, threshold, original_prob, truth)

    reasons = explain_reasons(shap_row, row.iloc[0], features, reference)
    left, right = st.columns(2)
    with left:
        st.markdown("#### 🔺 Why it looks suspicious")
        for r in reasons["for_fraud"] or [{"sentence": "_Nothing notable._"}]:
            st.markdown(f"- {r['sentence']}")
        st.markdown("#### 🔻 Why it looks normal")
        for r in reasons["against_fraud"] or [{"sentence": "_Nothing notable._"}]:
            st.markdown(f"- {r['sentence']}")
    with right:
        st.plotly_chart(reasons_chart(reasons), width="stretch", key=f"{key}_reasons")

    st.markdown("#### 🔄 What would change the decision?")
    cf_key = ("cf", tuple(np.round(row.to_numpy()[0], 6)))
    if st.button("Find the smallest change that flips it (DiCE)", key=f"{key}_dice"):
        st.session_state[cf_key] = get_counterfactual(model, train, row, tuple(features), threshold)
    cf = st.session_state.get(cf_key)
    if cf is not None:
        lines = counterfactual_sentences(cf, prob, threshold)
        st.markdown("\n".join(lines) if lines else "DiCE could not find a counterfactual for this one.")

    st.markdown("#### 🧾 Audit trail (GDPR Art. 22)")
    if st.button("Record this decision and its explanation", key=f"{key}_log"):
        log_explanation(
            transaction_id=transaction_id,
            shap_values=dict(zip(features, map(float, shap_row))),
            counterfactual=cf or {},
            fraud_prob=prob,
        )
        last = EXPLANATION_LOG_PATH.read_text(encoding="utf-8").splitlines()[-1]
        st.success("Appended to logs/explanation_log.jsonl")
        st.json(json.loads(last), expanded=False)

"""XCCFDS Streamlit dashboard (run with: streamlit run app.py)."""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.charts import FRAUD_RED, LEGIT_BLUE, drift_chart, gauge, waterfall
from src.demo_view import demo_tab
from src.scenario_view import scenario_tab
from src.drift import run_all_cohorts
from src.explainability import (
    base_value,
    compute_shap_values,
    generate_counterfactual,
    global_importance,
    local_waterfall_data,
)
from src.logger import log_explanation
from src.paths import DRIFT_LOG_PATH, EVAL_REPORT_PATH, TARGET
from src.preprocess import load_and_split
from src.train import load_model

PR_AUC_DROP_ALERT = 0.05
DRIFT_THRESHOLD = 0.20

st.set_page_config(page_title="XCCFDS", page_icon="🔍", layout="wide")


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_splits() -> dict[str, pd.DataFrame]:
    return load_and_split()


@st.cache_data
def get_test_shap(_model, X: pd.DataFrame):
    return compute_shap_values(_model, X)


@st.cache_data
def get_base_value(_model) -> float:
    return base_value(_model)


@st.cache_data(show_spinner="Searching for a counterfactual with DiCE...")
def get_counterfactual(_model, _train: pd.DataFrame, idx: int, instance: pd.DataFrame, features, threshold):
    return generate_counterfactual(_model, _train, instance, list(features), threshold)


def load_report() -> dict | None:
    if not EVAL_REPORT_PATH.exists():
        return None
    return json.loads(EVAL_REPORT_PATH.read_text())


def global_view(model, X_test: pd.DataFrame, features: list[str]) -> None:
    importance = global_importance(get_test_shap(model, X_test), features).head(15)
    fig = go.Figure(
        go.Bar(
            x=importance["mean_abs_shap"][::-1],
            y=importance["feature"][::-1],
            orientation="h",
            marker_color=LEGIT_BLUE,
        )
    )
    fig.update_layout(
        title="Top-15 features by mean |SHAP| (full test set)",
        xaxis_title="mean |SHAP| (log-odds)",
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, width="stretch")

    report = load_report()
    if report is None:
        st.info("No evaluation report yet. Run `python -m src.main` first.")
        return
    table = pd.DataFrame(report["metrics"]).T[["pr_auc", "f1", "mcc", "recall_at_1pct_fpr"]]
    val_pr_auc = table.loc["val", "pr_auc"]

    def highlight_drift(row: pd.Series) -> list[str]:
        dropped = row.name != "val" and row["pr_auc"] < val_pr_auc * (1 - PR_AUC_DROP_ALERT)
        return [f"background-color: {FRAUD_RED}55" if dropped else ""] * len(row)

    st.subheader("Cohort metrics")
    st.caption(
        f"Threshold {report['threshold']:.4f} (tuned on val). Rows in red: "
        f"PR-AUC more than {PR_AUC_DROP_ALERT:.0%} below validation."
    )
    st.dataframe(table.style.apply(highlight_drift, axis=1).format("{:.4f}"))






def transaction_explainer(model, splits: dict, features: list[str], threshold: float) -> None:
    test = splits["test"]
    X_test = test[features]
    idx = int(st.number_input("Transaction index (position in test set)", 0, len(test) - 1, 0))
    row = X_test.iloc[[idx]]
    prob = float(model.predict_proba(row)[0, 1])
    shap_row = get_test_shap(model, X_test)[idx]

    left, right = st.columns([1, 2])
    with left:
        st.plotly_chart(gauge(prob, threshold), width="stretch")
        st.caption(f"Decision threshold {threshold:.4f}: flagged as **{'fraud' if prob >= threshold else 'legitimate'}**")
        st.caption(f"True label: **{'fraud' if test[TARGET].iloc[idx] == 1 else 'legitimate'}**")
        st.dataframe(row.T.rename(columns={row.index[0]: "value"}), height=320)
    with right:
        data = local_waterfall_data(shap_row, get_base_value(model), features, top_k=12)
        st.plotly_chart(waterfall(data), width="stretch")

    st.subheader("Counterfactual: what would need to change?")
    cf = get_counterfactual(model, splits["train"], idx, row, tuple(features), threshold)
    if "error" in cf:
        st.warning(f"DiCE: {cf['error']}")
    else:
        target = "legitimate" if prob >= threshold else "fraud"
        st.caption(f"Minimal changes found by DiCE to flip the prediction to **{target}**.")
        st.dataframe(pd.DataFrame(cf, index=["original", "counterfactual"]).T)

    if st.button("Log this explanation"):
        log_explanation(
            transaction_id=f"test_{test.index[idx]}",
            shap_values=dict(zip(features, map(float, shap_row))),
            counterfactual=cf,
            fraud_prob=prob,
        )
        st.success("Explanation appended to logs/explanation_log.jsonl")




def drift_monitor(model, splits: dict) -> None:
    if st.button("Run drift analysis on all cohorts"):
        with st.spinner("Computing SHAP drift for each cohort..."):
            st.session_state["drift"] = run_all_cohorts(model, splits, DRIFT_THRESHOLD)

    results = st.session_state.get("drift")
    if results:
        for name, signal in results.items():
            if signal["alerts"]:
                flagged = ", ".join(f"{a['feature']} ({a['shift_pct']:.0%})" for a in signal["alerts"])
                st.error(f"**{name}**: {len(signal['alerts'])} feature(s) shifted > {DRIFT_THRESHOLD:.0%}: {flagged}")
            else:
                st.success(f"**{name}**: no feature shifted more than {DRIFT_THRESHOLD:.0%}")
        cols = st.columns(len(results))
        for col, (name, signal) in zip(cols, results.items()):
            col.plotly_chart(drift_chart(name, signal), width="stretch")

    st.subheader("Drift log")
    try:
        history = json.loads(DRIFT_LOG_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        history = []
    if not history:
        st.caption("No drift runs logged yet.")
        return
    rows = [
        {
            "timestamp_utc": h["timestamp_utc"],
            "cohort": h["cohort"],
            "cohort_size": h["cohort_size"],
            "n_alerts": len(h["alerts"]),
            "alerted_features": ", ".join(a["feature"] for a in h["alerts"]),
        }
        for h in reversed(history)
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def main() -> None:
    st.title("XCCFDS — Explainable Credit Card Fraud Detection")
    try:
        model, features = get_model()
        splits = get_splits()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    threshold = (load_report() or {}).get("threshold", 0.5)
    tabs = st.tabs(["Live Demo", "Custom Scenario", "Global View", "Transaction Explainer", "Drift Monitor"])
    with tabs[0]:
        demo_tab(model, splits, features, threshold, get_test_shap(model, splits["test"][features]))
    with tabs[1]:
        scenario_tab(model, splits, features, threshold)
    with tabs[2]:
        global_view(model, splits["test"][features], features)
    with tabs[3]:
        transaction_explainer(model, splits, features, threshold)
    with tabs[4]:
        drift_monitor(model, splits)

main()

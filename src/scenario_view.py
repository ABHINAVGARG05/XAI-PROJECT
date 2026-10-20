"""Custom Scenario tab: build a hypothetical transaction and get an explained prediction."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import streamlit as st

from src.demo_view import get_reference, render_decision
from src.paths import TARGET

# name: (amount €, hour of day, how fraud-like the hidden behaviour pattern is 0..1)
STORIES = {
    "☕ Morning coffee": (4.50, 8.25, 0.0),
    "🛒 Big weekend shop": (850.0, 14.0, 0.2),
    "🤔 Slightly odd evening order": (120.0, 23.0, 0.3),
    "🌙 3 a.m. card test": (1.00, 3.17, 0.6),
    "💳 Account takeover": (1999.0, 2.5, 1.0),
}
KEYS = ("sc_amount", "sc_hour", "sc_pattern")


@st.cache_data
def behaviour_profiles(train: pd.DataFrame, features: tuple[str, ...]) -> tuple[pd.Series, pd.Series]:
    """Median of each hidden V-feature for legitimate and for fraudulent training rows."""
    hidden = [f for f in features if f.startswith("V")]
    return (
        train.loc[train[TARGET] == 0, hidden].median(),
        train.loc[train[TARGET] == 1, hidden].median(),
    )


def build_row(
    features: list[str], legit: pd.Series, fraud: pd.Series, amount: float, hour: float, pattern: float
) -> pd.DataFrame:
    """Encode a scenario exactly as preprocessing encodes a real transaction."""
    values = ((1 - pattern) * legit + pattern * fraud).to_dict()
    values["Amount_log"] = float(np.log1p(amount))
    values["Time_sin"] = float(np.sin(2 * np.pi * hour / 24))
    values["Time_cos"] = float(np.cos(2 * np.pi * hour / 24))
    return pd.DataFrame([values])[features]


def scenario_id(row: pd.DataFrame) -> str:
    digest = hashlib.sha1(np.round(row.to_numpy(), 6).tobytes()).hexdigest()[:10]
    return f"scenario_{digest}"


def scenario_tab(model, splits: dict, features: list[str], threshold: float) -> None:
    train = splits["train"]
    legit, fraud = behaviour_profiles(train, tuple(features))
    reference = get_reference(train, tuple(features))

    st.markdown(
        "Describe a transaction and the model predicts and explains it. Amount and time are set "
        "directly. V1–V28 are anonymised, so their values come from the **behaviour pattern** "
        "slider, which moves them from a typical legitimate profile towards a typical fraud "
        "profile (medians of the training data)."
    )
    st.caption("Start from a story:")
    cols = st.columns(len(STORIES))
    for col, (name, preset) in zip(cols, STORIES.items()):
        if col.button(name, width="stretch", key=f"story_{name}"):
            st.session_state.update(dict(zip(KEYS, preset)))
            # A preset is a complete scenario: drop manual V-feature overrides and their values.
            st.session_state["sc_override"] = []
            for k in [k for k in st.session_state if k.startswith("sc_V")]:
                del st.session_state[k]
    for k, default in zip(KEYS, next(iter(STORIES.values()))):
        st.session_state.setdefault(k, default)

    c1, c2, c3 = st.columns(3)
    amount = c1.number_input("Amount (€)", 0.0, 25_000.0, step=1.0, key="sc_amount")
    hour = c2.slider("Time of day (hour)", 0.0, 23.99, step=0.25, key="sc_hour")
    pattern = c3.slider(
        "Behaviour pattern: typical customer → fraud-like", 0.0, 1.0, step=0.05, key="sc_pattern"
    )
    st.caption(f"Scenario: €{amount:,.2f} at {int(hour):02d}:{int(hour % 1 * 60):02d}, pattern {pattern:.0%} fraud-like")

    row = build_row(features, legit, fraud, amount, hour, pattern)
    hidden = [f for f in features if f.startswith("V")]
    with st.expander("🔬 Advanced: override individual anonymised features"):
        chosen = st.multiselect("Features to set by hand", hidden, key="sc_override")
        ov_cols = st.columns(max(len(chosen), 1))
        for col, feat in zip(ov_cols, chosen):
            row[feat] = col.number_input(feat, value=float(row[feat].iloc[0]), step=0.5, key=f"sc_{feat}")

    st.info(
        "What this model learned: the hidden behaviour pattern drives the decision; "
        "amount and time of day barely move it. Try the sliders to see this for yourself."
    )
    render_decision(
        model,
        row,
        features,
        threshold,
        reference,
        train,
        transaction_id=scenario_id(row),
        key="scenario",
    )

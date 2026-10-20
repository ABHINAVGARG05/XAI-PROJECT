"""Plotly figure builders shared by the dashboard tabs."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

FRAUD_RED = "#d62728"
LEGIT_BLUE = "#1f77b4"


def gauge(prob: float, threshold: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "valueformat": ".2f"},
            title={"text": "Fraud probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": FRAUD_RED if prob >= threshold else LEGIT_BLUE},
                "steps": [{"range": [0, threshold * 100], "color": "#eef4fb"}, {"range": [threshold * 100, 100], "color": "#fdecec"}],
                "threshold": {"line": {"color": "#333", "width": 3}, "value": threshold * 100},
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=10))
    return fig


def waterfall(data: dict) -> go.Figure:
    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            base=data["base_value"],
            x=data["shap_vals"][::-1],
            y=data["features"][::-1],
            increasing={"marker": {"color": FRAUD_RED}},
            decreasing={"marker": {"color": LEGIT_BLUE}},
            connector={"line": {"color": "#999", "width": 1}},
        )
    )
    fig.update_layout(
        title=f"SHAP contributions: base {data['base_value']:.2f} → f(x) {data['prediction']:.2f} (log-odds)",
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def drift_chart(name: str, signal: dict) -> go.Figure:
    shifts = pd.DataFrame(signal["feature_shifts"]).T.nlargest(5, "shift_pct")
    fig = go.Figure(
        [
            go.Bar(name="baseline (train)", x=shifts.index, y=shifts["baseline"], marker_color=LEGIT_BLUE),
            go.Bar(name=name, x=shifts.index, y=shifts["cohort"], marker_color=FRAUD_RED),
        ]
    )
    fig.update_layout(
        barmode="group",
        title=f"{name}: top-5 shifted features (n={signal['cohort_size']:,})",
        yaxis_title="mean |SHAP|",
        height=360,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def reasons_chart(reasons: dict[str, list[dict]]) -> go.Figure:
    items = sorted(reasons["for_fraud"] + reasons["against_fraud"], key=lambda r: r["impact"])
    fig = go.Figure(
        go.Bar(
            x=[r["impact"] for r in items],
            y=[r["feature"] for r in items],
            orientation="h",
            marker_color=[FRAUD_RED if r["impact"] > 0 else LEGIT_BLUE for r in items],
        )
    )
    fig.update_layout(
        title="Impact on the decision (→ fraud, ← legitimate)",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig

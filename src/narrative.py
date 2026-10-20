"""Turn SHAP values and counterfactuals into plain-English explanations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.paths import SEED, TARGET

TIME_FEATURES = ("Time_sin", "Time_cos")
TIME_LABEL = "Time of day"
UNUSUAL_PCT = 0.95
REFERENCE_SAMPLE = 50_000
MIN_IMPACT = 0.01  # |SHAP| in log-odds below which a feature is not worth mentioning


def build_reference(train: pd.DataFrame, feature_names: list[str]) -> dict[str, np.ndarray]:
    """Sorted per-feature values of legitimate training transactions."""
    legit = train[train[TARGET] == 0]
    legit = legit.sample(min(REFERENCE_SAMPLE, len(legit)), random_state=SEED)
    return {f: np.sort(legit[f].to_numpy()) for f in feature_names}


def percentile(reference: dict[str, np.ndarray], feature: str, value: float) -> float:
    """Fraction of legitimate transactions with a value below ``value``."""
    ref = reference[feature]
    return float(np.searchsorted(ref, value) / len(ref))


def hour_of_day(time_sin: float, time_cos: float) -> float:
    """Invert the cyclic encoding back to an hour in [0, 24)."""
    return float(np.arctan2(time_sin, time_cos) / (2 * np.pi) * 24) % 24


def format_value(feature: str, value: float) -> str:
    """Human-readable value: amounts in currency, other features as numbers."""
    if feature == "Amount_log":
        return f"€{np.expm1(value):,.2f}"
    return f"{value:.2f}"


def display_name(feature: str) -> str:
    return "Amount" if feature == "Amount_log" else feature


def _share_of(frac: float) -> str:
    return "virtually all (>99.9%)" if frac > 0.999 else f"{frac:.1%} of"


def describe_rarity(pct: float) -> str:
    """Where a value sits relative to legitimate transactions."""
    if pct >= UNUSUAL_PCT:
        return f"higher than {_share_of(pct)} normal transactions"
    if pct <= 1 - UNUSUAL_PCT:
        return f"lower than {_share_of(1 - pct)} normal transactions"
    return "within the normal range"


def explain_reasons(
    shap_row: np.ndarray,
    row: pd.Series,
    feature_names: list[str],
    reference: dict[str, np.ndarray],
    top_k: int = 3,
) -> dict[str, list[dict]]:
    """Top reasons raising and lowering fraud risk, as sentences.

    The two cyclic time features are merged into a single "Time of day" reason.
    """
    contributions = dict(zip(feature_names, map(float, shap_row)))
    reasons = []
    if all(t in contributions for t in TIME_FEATURES):
        impact = contributions.pop(TIME_FEATURES[0]) + contributions.pop(TIME_FEATURES[1])
        hour = hour_of_day(row[TIME_FEATURES[0]], row[TIME_FEATURES[1]])
        reasons.append(
            {"feature": TIME_LABEL, "impact": impact, "text": f"made at {int(hour):02d}:{int(hour % 1 * 60):02d}"}
        )
    for feat, impact in contributions.items():
        value = float(row[feat])
        rarity = describe_rarity(percentile(reference, feat, value))
        reasons.append(
            {"feature": display_name(feat), "impact": impact, "text": f"is {format_value(feat, value)}, {rarity}"}
        )

    for r in reasons:
        effect = "raises" if r["impact"] > 0 else "lowers"
        r["sentence"] = f"**{r['feature']}** {r['text']} → {effect} fraud risk ({r['impact']:+.2f})"

    ranked = sorted(reasons, key=lambda r: abs(r["impact"]), reverse=True)
    return {
        "for_fraud": [r for r in ranked if r["impact"] >= MIN_IMPACT][:top_k],
        "against_fraud": [r for r in ranked if r["impact"] <= -MIN_IMPACT][:top_k],
    }


def decision_label(prob: float, threshold: float) -> str:
    return "BLOCKED" if prob >= threshold else "APPROVED"


def counterfactual_sentences(cf: dict, prob: float, threshold: float) -> list[str]:
    """Describe a DiCE counterfactual as a list of changes plus the new outcome."""
    if "error" in cf:
        return []
    lines = []
    time_changed = False
    for feat, (orig, new) in cf.items():
        if feat in TIME_FEATURES:
            time_changed = True
            continue
        lines.append(
            f"**{display_name(feat)}** were {format_value(feat, new)} instead of {format_value(feat, orig)}"
        )
    if time_changed:
        lines.append(f"**{TIME_LABEL}** were different")
    flipped = "APPROVED" if decision_label(prob, threshold) == "BLOCKED" else "BLOCKED"
    return [f"The transaction would have been **{flipped}** if:"] + [f"- {line}" for line in lines]

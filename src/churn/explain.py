"""Reason codes: roll per-feature SHAP values up into business-readable churn drivers.

A retention agent can't act on "rch_max_gap_days_90d = +0.41". They can act on
"Recharge slowdown -> recharge bonus". Features are grouped into families, and summed SHAP per
family ranks the reasons for each subscriber.

Families are either *symptoms* (the subscriber is going quiet, spending less) or *root causes*
(bad network, competitor offer, new SIM, credit stress, multi-SIM). Symptoms dominate the score,
but the offer should address the cause. So the next-best action comes from the strongest
root-cause driver, and falls back to the top symptom when no cause stands out.
"""

from __future__ import annotations

import numpy as np

FAMILY_LABELS = {
    "inactivity": "Going quiet: fewer active days, long time since last activity",
    "recharge_slowdown": "Recharge slowdown: longer gaps, fewer or smaller top-ups",
    "usage_decline": "Falling voice / data / revenue",
    "network_experience": "Poor network experience: call drops, slow data, complaints",
    "competitor_pressure": "Competitor pull: calls shifting to a rival network, local promotions",
    "social_contagion": "Close contacts recently left the network",
    "early_life": "New SIM in the early-life churn window",
    "credit_stress": "Frequent emergency balance loans",
    "multi_sim_low_loyalty": "Multi-SIM user with a weak on-net community",
    "profile": "Customer profile / value segment",
}

FAMILY_ACTIONS = {
    "inactivity": "Win-back nudge: personalised bonus on next recharge (SMS + app push)",
    "recharge_slowdown": "Recharge bonus on the usual denomination (e.g. recharge 99 Tk, get 1.5 GB)",
    "usage_decline": "Usage-stimulation bundle sized to the last 3 months' consumption",
    "network_experience": "Proactive care call, network ticket for the serving site, goodwill minutes",
    "competitor_pressure": "Price-matched counter-offer against the competitor the subscriber calls most",
    "social_contagion": "Friends & family bundle: on-net group minutes to keep the circle together",
    "early_life": "Onboarding journey: bonus on 2nd/3rd recharge + self-care app sign-up reward",
    "credit_stress": "Higher emergency balance limit / low-denomination micro packs",
    "multi_sim_low_loyalty": "'Primary SIM' loyalty offer: on-net minutes + FnF bonus",
    "profile": "Standard retention offer",
}

SYMPTOMS = frozenset({"inactivity", "usage_decline", "recharge_slowdown"})

_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("inactivity", ("days_since_last_activity", "days_active", "months_active")),
    ("recharge_slowdown", ("rch_", "days_since_last_recharge")),
    ("network_experience", ("drop_call", "data_speed", "complaint", "div_drop", "div_outage")),
    ("competitor_pressure", ("competitor_", "top_competitor")),
    ("social_contagion", ("contacts_",)),
    ("early_life", ("tenure_months",)),
    ("credit_stress", ("emergency_loan",)),
    ("multi_sim_low_loyalty", ("dual_sim", "onnet_share")),
    ("usage_decline", ("revenue", "voice_min", "onnet_min", "data_mb", "sms_cnt", "data_pack")),
]


def feature_family(feature: str) -> str:
    for family, prefixes in _RULES:
        if feature.startswith(prefixes):
            return family
    return "profile"


def family_matrix(features: list[str]) -> tuple[list[str], np.ndarray]:
    """(families, F) where F[i, j] = 1 if feature i belongs to family j."""
    families = list(FAMILY_LABELS)
    F = np.zeros((len(features), len(families)))
    for i, f in enumerate(features):
        F[i, families.index(feature_family(f))] = 1.0
    return families, F


def primary_driver(fam_scores: np.ndarray, families: list[str], min_score: float = 0.05) -> np.ndarray:
    """Strongest root-cause family per row (SHAP > min_score); falls back to the strongest family overall."""
    cause_cols = np.array([f not in SYMPTOMS for f in families])
    causes = np.where(cause_cols, fam_scores, -np.inf)
    best_cause = causes.argmax(axis=1)
    has_cause = causes[np.arange(len(causes)), best_cause] > min_score
    names = np.array(families, dtype=object)
    return np.where(has_cause, names[best_cause], names[fam_scores.argmax(axis=1)])


def global_family_importance(contrib: np.ndarray, features: list[str]) -> dict[str, float]:
    families, F = family_matrix(features)
    imp = np.abs(contrib @ F).mean(axis=0)
    return dict(sorted(zip(families, imp.tolist(), strict=True), key=lambda kv: -kv[1]))

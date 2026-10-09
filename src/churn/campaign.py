"""Retention campaign economics: who is worth contacting, and what the model earns.

A churn score alone doesn't decide who to target. A 90%-risk subscriber paying 30 Tk/month
is worth less than a 40%-risk subscriber paying 800 Tk/month. Each subscriber gets:

    value_if_saved = save_rate * margin * ARPU * clv_months
    cost           = contact_cost + offer_redeem_rate * offer_cost
    expected_net   = p(churn) * value_if_saved - cost

Target everyone with expected_net > 0, ranked by expected_net and optionally capped by budget.
"""

from __future__ import annotations

import numpy as np


def contact_cost(cfg: dict) -> float:
    return cfg["contact_cost"] + cfg["offer_redeem_rate"] * cfg["offer_cost"]


def value_if_saved(arpu: np.ndarray, cfg: dict) -> np.ndarray:
    return cfg["save_rate"] * cfg["margin"] * np.asarray(arpu, dtype=float) * cfg["clv_months"]


def expected_net(p: np.ndarray, arpu: np.ndarray, cfg: dict) -> np.ndarray:
    return np.asarray(p) * value_if_saved(arpu, cfg) - contact_cost(cfg)


def profit_curve(y: np.ndarray, rank_score: np.ndarray, arpu: np.ndarray, cfg: dict) -> np.ndarray:
    """Realised campaign profit when targeting the top-k subscribers by `rank_score`, for every k."""
    order = np.argsort(-np.asarray(rank_score))
    gain = np.asarray(y)[order] * value_if_saved(np.asarray(arpu)[order], cfg) - contact_cost(cfg)
    return np.concatenate([[0.0], np.cumsum(gain)])


def campaign_summary(y: np.ndarray, p: np.ndarray, arpu: np.ndarray, cfg: dict) -> dict:
    """Compare model-driven targeting (by expected_net) against untargeted baselines on labelled data."""
    y, p, arpu = np.asarray(y), np.asarray(p), np.asarray(arpu, dtype=float)
    n = len(y)
    ev = expected_net(p, arpu, cfg)
    curve = profit_curve(y, ev, arpu, cfg)
    k_policy = int((ev > 0).sum())
    k_best = int(np.argmax(curve))
    random_per_contact = (y * value_if_saved(arpu, cfg)).mean() - contact_cost(cfg)
    return {
        "base_size": n,
        "cost_per_contact": contact_cost(cfg),
        "policy_targets": k_policy,
        "policy_target_share": k_policy / n,
        "policy_profit": float(curve[k_policy]),
        "policy_churners_reached": int(y[np.argsort(-ev)[:k_policy]].sum()),
        "oracle_best_k": k_best,
        "oracle_best_profit": float(curve[k_best]),
        "random_targeting_profit_same_k": float(random_per_contact * k_policy),
        "blanket_campaign_profit": float(random_per_contact * n),
    }

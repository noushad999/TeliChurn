import numpy as np
import pandas as pd
import pytest

from churn.campaign import campaign_summary, contact_cost, expected_net, profit_curve
from churn.evaluate import classification_metrics, gains_table
from churn.explain import SYMPTOMS, family_matrix, feature_family, primary_driver
from churn.monitor import build_reference, drift_report

CAMPAIGN = {
    "contact_cost": 2.0,
    "offer_cost": 40.0,
    "offer_redeem_rate": 0.6,
    "save_rate": 0.3,
    "margin": 0.55,
    "clv_months": 6,
}


def test_expected_net_breakeven():
    assert contact_cost(CAMPAIGN) == pytest.approx(26.0)
    arpu = np.array([200.0])
    breakeven_p = contact_cost(CAMPAIGN) / (0.3 * 0.55 * 200 * 6)
    assert expected_net(np.array([breakeven_p]), arpu, CAMPAIGN)[0] == pytest.approx(0.0)
    assert (
        expected_net(np.array([0.9]), arpu, CAMPAIGN)[0]
        > 0
        > expected_net(np.array([0.01]), arpu, CAMPAIGN)[0]
    )


def test_perfect_ranking_beats_random():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.05).astype(int)
    arpu = rng.lognormal(5, 0.5, 5000)
    perfect = profit_curve(y, y + rng.random(5000) * 0.1, arpu, CAMPAIGN).max()
    random = profit_curve(y, rng.random(5000), arpu, CAMPAIGN).max()
    assert perfect > random
    summary = campaign_summary(y, y * 0.9 + 0.01, arpu, CAMPAIGN)
    assert summary["policy_profit"] > summary["random_targeting_profit_same_k"]


def test_metrics_on_perfect_scores():
    y = np.array([0] * 90 + [1] * 10)
    m = classification_metrics(y, y.astype(float))
    assert m["roc_auc"] == 1.0 and m["capture_top10"] == 1.0 and m["lift_top10"] == pytest.approx(10.0)
    g = gains_table(y, y.astype(float))
    assert g["cum_capture"].iloc[-1] == pytest.approx(1.0) and len(g) == 10


def test_feature_families():
    assert feature_family("rch_max_gap_days_90d") == "recharge_slowdown"
    assert feature_family("days_since_last_recharge") == "recharge_slowdown"
    assert feature_family("days_since_last_activity") == "inactivity"
    assert feature_family("drop_call_rate_m0") == "network_experience"
    assert feature_family("tenure_months") == "early_life"
    assert feature_family("age") == "profile"
    fams, F = family_matrix(["tenure_months", "age", "data_mb_m0"])
    assert F.sum(axis=1).tolist() == [1, 1, 1] and len(fams) == F.shape[1]


def test_primary_driver_prefers_root_cause_over_symptom():
    fams = ["inactivity", "network_experience", "early_life"]
    scores = np.array([[2.0, 0.4, 0.1], [2.0, 0.01, 0.0]])
    drivers = primary_driver(scores, fams)
    assert drivers[0] == "network_experience"  # cause wins even though the symptom is larger
    assert drivers[1] == "inactivity"  # no meaningful cause -> fall back to the symptom
    assert "inactivity" in SYMPTOMS


def test_psi_detects_shift():
    rng = np.random.default_rng(1)
    ref_X = pd.DataFrame({"x": rng.normal(0, 1, 20000), "c": pd.Categorical(rng.choice(["a", "b"], 20000))})
    ref = build_reference(ref_X, rng.random(20000))
    same = drift_report(ref, ref_X, rng.random(20000))
    assert same["score_status"] == "stable" and same["n_major"] == 0
    shifted_X = pd.DataFrame({"x": rng.normal(1.5, 1, 20000), "c": pd.Categorical(["a"] * 20000)})
    shifted = drift_report(ref, shifted_X, rng.random(20000) ** 3)
    assert shifted["n_major"] == 2 and shifted["score_status"] == "major"


def test_qini_rewards_true_uplift_ranking():
    from churn.uplift import evaluate as evaluate_uplift
    from churn.uplift import qini_curve

    rng = np.random.default_rng(3)
    n = 40_000
    true_uplift = rng.uniform(-0.05, 0.15, n)
    t = (rng.random(n) < 0.5).astype(int)
    base = 0.2
    y = (rng.random(n) < base - t * true_uplift).astype(int)
    res = evaluate_uplift(y, t, {"oracle": true_uplift, "random": rng.random(n)})
    r = res["rankings"]
    assert r["oracle"]["qini_coefficient"] > r["random"]["qini_coefficient"]
    assert r["oracle"]["prevented_per_1000_top10"] > 100  # top decile has uplift ~0.13
    x, q = qini_curve(y, t, true_uplift)
    assert x[0] == 0 and x[-1] == pytest.approx(1.0) and q.argmax() < len(q) - 1  # sleeping dogs at the end


def test_value_curve_counts_contact_cost():
    from churn.uplift import value_curve

    y = np.zeros(1000, int)
    t = np.tile([0, 1], 500)
    x, v = value_curve(y, t, np.arange(1000.0), np.full(1000, 100.0), cost=10.0)
    assert v[-1] == pytest.approx(-10.0 * 1000)  # nobody churns: pure cost


def test_segment_report_flags_miscalibrated_group():
    import polars as pl

    from churn.fairness import segment_report

    rng = np.random.default_rng(5)
    n = 20_000
    div = np.where(rng.random(n) < 0.5, "Dhaka", "Sylhet")
    p = np.where(rng.random(n) < 0.1, 0.6, 0.02)  # a sharp, well-ranking model
    y = (rng.random(n) < p).astype(int)
    p_biased = np.where(div == "Sylhet", p + 0.08, p)  # ...that over-predicts Sylhet
    test = pl.DataFrame({"churn": y, "division": div, "tenure_months": np.full(n, 24), "urban": [None] * n})
    rep = segment_report(test, p_biased, p > 0.3)
    flagged = {(f["dimension"], f["segment"]) for f in rep["flags"]}
    assert ("division", "Sylhet") in flagged and ("division", "Dhaka") not in flagged
    assert not any(r["dimension"] == "urban" for r in rep["segments"])  # unknown -> not reported as rural

from datetime import date, datetime

import polars as pl
import pytest

from churn.features import build_snapshot, feature_columns
from churn.simulate import simulate


def test_simulation_is_deterministic():
    a = simulate(n_subscribers=500, seed=7)
    b = simulate(n_subscribers=500, seed=7)
    for name in a:
        assert a[name].equals(b[name]), name


def test_churned_subscribers_never_come_back(tables):
    # activity months per subscriber must be one contiguous run: silence means gone
    runs = (
        tables["usage_monthly"]
        .group_by("sub_id")
        .agg(n=pl.len(), first=pl.col("month").min(), last=pl.col("month").max())
        .with_columns(
            span=(pl.col("last").dt.year() * 12 + pl.col("last").dt.month())
            - (pl.col("first").dt.year() * 12 + pl.col("first").dt.month())
            + 1
        )
    )
    assert (runs["n"] == runs["span"]).all()


def test_recharges_only_while_active(tables):
    usage_keys = tables["usage_monthly"].select("sub_id", "month")
    rch = tables["recharges"].select("sub_id", month=pl.col("ts").dt.truncate("1mo").cast(pl.Date))
    orphans = rch.join(usage_keys, on=["sub_id", "month"], how="anti")
    assert orphans.height == 0


def test_realistic_churn_rate(tables):
    snap = build_snapshot(tables, "2025-06")
    assert 0.015 < snap["churn"].mean() < 0.10


def test_features_do_not_leak_future_data(tables):
    """Deleting everything after the cutoff must not change a single feature value."""
    cutoff = "2025-06"
    full = build_snapshot(tables, cutoff, with_label=False)
    cut_end = datetime(2025, 6, 30, 23, 59, 59, 999999)
    truncated = {
        **tables,
        "usage_monthly": tables["usage_monthly"].filter(pl.col("month") <= date(2025, 6, 1)),
        "recharges": tables["recharges"].filter(pl.col("ts") <= cut_end),
        "network_market": tables["network_market"].filter(pl.col("month") <= date(2025, 6, 1)),
        "campaigns": tables["campaigns"].filter(pl.col("campaign_month") <= date(2025, 6, 1)),
    }
    past_only = build_snapshot(truncated, cutoff, with_label=False)
    assert full.equals(past_only)


def _tiny_tables() -> dict[str, pl.DataFrame]:
    months = [date(2025, m, 1) for m in (1, 2, 3, 4, 5, 6)]
    subs = pl.DataFrame(
        {
            "sub_id": [1, 2, 3, 4],
            "activation_month": [date(2024, 1, 1)] * 4,
            "division": ["Dhaka"] * 4,
            "urban": [True] * 4,
            "gender": ["M"] * 4,
            "age": [30] * 4,
            "plan_type": ["prepaid"] * 4,
            "handset": ["smartphone_4g"] * 4,
            "dual_sim": [False] * 4,
            "app_user": [False] * 4,
            "mfs_user": [True] * 4,
        }
    )
    # sub 1: active through June -> stays. sub 2: last activity in March -> churn.
    # sub 3: silent Apr, recharges in May -> not churn. sub 4: gone before March -> not in snapshot.
    active = {1: months, 2: months[:3], 3: months[:3], 4: months[:1]}
    rows = [
        {
            "sub_id": s,
            "month": m,
            "voice_min": 50.0,
            "onnet_min": 25.0,
            "data_mb": 900.0,
            "sms_cnt": 3,
            "data_pack_cnt": 1,
            "revenue": 100.0,
            "days_active": 25,
            "last_active_day": 28,
            "emergency_loan_cnt": 0,
            "complaint_cnt": 0,
            "drop_call_rate": 0.01,
            "data_speed_mbps": 15.0,
        }
        for s, ms in active.items()
        for m in ms
    ]
    usage = pl.DataFrame(rows).with_columns(
        pl.col(
            "sms_cnt",
            "data_pack_cnt",
            "days_active",
            "last_active_day",
            "emergency_loan_cnt",
            "complaint_cnt",
        ).cast(pl.Int16)
    )
    rch = pl.DataFrame(
        {
            "sub_id": [1, 2, 3, 3],
            "ts": [datetime(2025, 3, 5), datetime(2025, 3, 10), datetime(2025, 3, 12), datetime(2025, 5, 20)],
            "amount": [99.0, 49.0, 49.0, 20.0],
            "channel": ["mfs", "retailer", "retailer", "mfs"],
        }
    )
    market = pl.DataFrame(
        {
            "division": ["Dhaka"] * 6,
            "month": months,
            "avg_drop_call_rate": [0.01] * 6,
            "network_outage": [False] * 6,
            "competitor_promo": [False] * 6,
        }
    )
    return {"subscribers": subs, "usage_monthly": usage, "recharges": rch, "network_market": market}


def test_label_definition():
    snap = build_snapshot(_tiny_tables(), "2025-03", lookback_months=3, inactivity_months=2)
    labels = dict(zip(snap["sub_id"].to_list(), snap["churn"].to_list(), strict=True))
    assert labels == {1: 0, 2: 1, 3: 0}
    row = snap.filter(pl.col("sub_id") == 2).row(0, named=True)
    assert row["days_since_last_recharge"] == pytest.approx(21, abs=1)
    assert row["rch_cnt_30d"] == 1


def test_label_needs_future_data():
    with pytest.raises(ValueError, match="Cannot label"):
        build_snapshot(_tiny_tables(), "2025-05", inactivity_months=2)


def test_feature_set_is_stable(tables):
    a = feature_columns(build_snapshot(tables, "2025-04"))
    b = feature_columns(build_snapshot(tables, "2025-09"))
    assert a == b and len(a) > 60


def test_graph_and_competitor_features_present_and_sane(tables):
    snap = build_snapshot(tables, "2025-07")
    for col in (
        "contacts_gone_share",
        "contacts_n",
        "competitor_share_m0",
        "competitor_share_trend",
        "top_competitor",
    ):
        assert col in snap.columns
    assert snap["contacts_gone_share"].drop_nulls().is_between(0, 1).all()
    assert snap["competitor_share_m0"].drop_nulls().is_between(0, 1).all()
    assert set(snap["top_competitor"].drop_nulls().unique()) <= {"robi", "banglalink", "teletalk"}
    # churners call competitors more and have lost more contacts, on average
    g = snap.group_by("churn").agg(pl.col("competitor_share_m0").mean(), pl.col("contacts_gone_share").mean())
    g = {r["churn"]: r for r in g.iter_rows(named=True)}
    assert g[1]["competitor_share_m0"] > g[0]["competitor_share_m0"]
    assert g[1]["contacts_gone_share"] > g[0]["contacts_gone_share"]


def test_optional_sources_can_be_absent(tables):
    minimal = {k: v for k, v in tables.items() if k not in ("contacts", "campaigns")}
    minimal["usage_monthly"] = tables["usage_monthly"].drop(
        "offnet_robi_min", "offnet_banglalink_min", "offnet_teletalk_min"
    )
    snap = build_snapshot(minimal, "2025-05")
    assert "contacts_n" not in snap.columns and "competitor_share_m0" not in snap.columns
    assert snap.height > 1000


def test_campaigns_are_randomised(tables):
    c = tables["campaigns"]
    share = c.group_by("campaign_month").agg(pl.col("treatment").mean())["treatment"]
    assert share.is_between(0.45, 0.55).all()

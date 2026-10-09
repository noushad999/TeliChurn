"""CVM dashboard for retention, marketing and call-centre teams (Streamlit).

    streamlit run src/churn/dashboard.py        # or: docker compose up dashboard

Reads the latest batch scores, the champion model's metadata, the registry, and the drift,
backtest and data-quality reports. It never recomputes features, so it opens instantly on the full base.
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import polars as pl
import streamlit as st

from churn.campaign import contact_cost, expected_net
from churn.config import load_config
from churn.explain import FAMILY_ACTIONS, FAMILY_LABELS
from churn.registry import Registry, resolve_model_dir
from churn.style import LIGHT

st.set_page_config(page_title="Churn · CVM dashboard", page_icon="📉", layout="wide")

ACCENT, INK, INK_2, MUTED, CARD = LIGHT["accent"], LIGHT["ink"], LIGHT["ink_2"], LIGHT["muted"], LIGHT["card"]
BAND_ORDER = ["critical", "high", "medium", "low"]

st.markdown(
    f"""
<link href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  html, body, [class*="css"], .stMarkdown, .stText {{ font-family: 'Source Sans 3', sans-serif; }}
  h1, h2, h3 {{ font-family: 'Source Sans 3', sans-serif !important; font-weight: 700 !important; color: {INK}; letter-spacing: -.01em; }}
  .kpi {{ background: {CARD}; border: 1px solid {LIGHT["edge"]}; border-radius: 10px; padding: 16px 18px; }}
  .kpi .v {{ font-family: 'Source Sans 3', sans-serif; font-weight: 700; font-size: 30px; color: {INK}; line-height: 1.1; }}
  .kpi .v.accent {{ color: {ACCENT}; }}
  .kpi .l {{ color: {INK_2}; font-size: 14px; margin-top: 6px; }}
  .eyebrow {{ color: {ACCENT}; font-weight: 600; letter-spacing: .04em; font-size: 13px; }}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_state() -> dict:
    cfg = load_config()
    paths = cfg["paths"]
    scores_files = sorted(Path(paths["scores_dir"]).glob("scores_20*.parquet"))
    if not scores_files:
        return {"cfg": cfg, "scores": None}
    latest = scores_files[-1]
    model_dir = resolve_model_dir(cfg)
    reports = Path(paths["reports_dir"])
    month = latest.stem.removeprefix("scores_")

    def read_json(p: Path):
        return json.loads(p.read_text()) if p.exists() else None

    return {
        "cfg": cfg,
        "month": month,
        "scores": pl.read_parquet(latest),
        "metadata": read_json(model_dir / "metadata.json"),
        "metrics": read_json(reports / "metrics.json"),
        "drift": read_json(reports / f"drift_{month}.json"),
        "dq": read_json(reports / f"data_quality_{month}.json") or read_json(reports / "data_quality.json"),
        "backtests": [read_json(p) for p in sorted(reports.glob("backtest_*.json"))],
        "versions": Registry(paths["registry_dir"]).versions() if paths.get("registry_dir") else [],
    }


def kpi(col, value: str, label: str, accent: bool = False) -> None:
    col.markdown(
        f'<div class="kpi"><div class="v{" accent" if accent else ""}">{value}</div><div class="l">{label}</div></div>',
        unsafe_allow_html=True,
    )


def bars(df: pl.DataFrame, x: str, y: str, title: str, order: list[str] | None = None, fmt: str = ",.0f"):
    """Horizontal bar chart: sorted by value, or by an explicit category `order`."""
    if order:
        rank = {v: i for i, v in enumerate(order)}
        df = df.with_columns(_order=pl.col(y).replace_strict(rank, default=len(order), return_dtype=pl.Int32))
    else:
        df = df.sort(x, descending=True).with_row_index("_order")
    title_params = alt.TitleParams(
        title, anchor="start", font="Source Sans 3", fontSize=16, fontWeight=600, color=INK
    )
    chart = (
        alt.Chart(df.to_pandas(), title=title_params)
        .mark_bar(color=ACCENT, cornerRadiusEnd=3)
        .encode(
            x=alt.X(
                f"{x}:Q",
                title=None,
                axis=alt.Axis(
                    format=fmt, grid=True, gridColor=LIGHT["grid"], domain=False, tickSize=0, labelColor=MUTED
                ),
            ),
            y=alt.Y(
                f"{y}:N",
                title=None,
                sort=alt.EncodingSortField(field="_order", order="ascending"),
                axis=alt.Axis(domain=False, tickSize=0, labelColor=INK, labelLimit=320, labelOverlap=False),
            ),
            tooltip=[y, alt.Tooltip(x, format=fmt)],
        )
        .properties(height=max(180, 36 * df.height + 30))
        .configure_view(stroke=None)
    )
    st.altair_chart(chart, use_container_width=True)


state = load_state()
cfg = state["cfg"]
st.markdown('<div class="eyebrow">PREPAID CHURN · CVM DASHBOARD</div>', unsafe_allow_html=True)
if state["scores"] is None:
    st.title("No scores yet")
    st.write("Run `churn pipeline` (or `churn score`) to produce the first scored month.")
    st.stop()

scores: pl.DataFrame = state["scores"]
meta = state["metadata"] or {}
camp_cfg = cfg["campaign"]
cur = camp_cfg["currency"]
st.title(f"Churn risk · {state['month']}")
st.caption(
    f"Model {meta.get('model_version', 'unknown')} · trained {meta.get('trained_at', '?')} · "
    f"{scores.height:,} active subscribers scored"
)

overview, campaign_tab, health, lookup = st.tabs(
    ["Overview", "Campaign", "Model health", "Subscriber lookup"]
)

# ------------------------------------------------------------------------------ overview
with overview:
    value_at_risk = (scores["churn_probability"] * scores["arpu_3m"] * camp_cfg["clv_months"]).sum()
    targets = scores.filter(pl.col("target"))
    c = st.columns(5)
    kpi(c[0], f"{scores.height:,}", "active subscribers")
    kpi(c[1], f"{(scores['risk_band'] == 'critical').sum():,}", "critical risk", accent=True)
    kpi(c[2], f"{value_at_risk / 1e6:,.1f}M {cur}", f"revenue at risk, next {camp_cfg['clv_months']} months")
    kpi(c[3], f"{targets.height:,}", "worth a retention offer")
    kpi(c[4], f"{targets['expected_net_value'].sum() / 1e3:,.0f}k {cur}", "expected campaign net value")
    st.write("")
    left, right = st.columns(2)
    with left:
        bands = scores.group_by("risk_band").agg(pl.len().alias("subscribers"))
        bars(bands, "subscribers", "risk_band", "Subscribers by risk band", order=BAND_ORDER)
        by_div = scores.group_by("division").agg(
            (pl.col("churn_probability") * pl.col("arpu_3m") * camp_cfg["clv_months"]).sum().alias("at_risk")
        )
        bars(by_div, "at_risk", "division", f"Revenue at risk by division ({cur})")
    with right:
        drivers = (
            targets.group_by("primary_driver")
            .len()
            .with_columns(
                driver=pl.col("primary_driver").replace_strict(
                    {k: v.split(":")[0] for k, v in FAMILY_LABELS.items()}, default=pl.col("primary_driver")
                )
            )
        )
        bars(drivers, "len", "driver", "Why campaign targets are at risk (primary driver)")
        tenure = (
            scores.with_columns(
                tenure=pl.when(pl.col("tenure_months") < 3)
                .then(pl.lit("0–2 months"))
                .when(pl.col("tenure_months") < 12)
                .then(pl.lit("3–11 months"))
                .when(pl.col("tenure_months") < 36)
                .then(pl.lit("1–3 years"))
                .otherwise(pl.lit("3+ years"))
            )
            .group_by("tenure")
            .agg(pl.col("churn_probability").mean().alias("avg_risk"))
        )
        bars(
            tenure,
            "avg_risk",
            "tenure",
            "Average churn risk by tenure",
            fmt=".1%",
            order=["0–2 months", "3–11 months", "1–3 years", "3+ years"],
        )

# ------------------------------------------------------------------------------ campaign
with campaign_tab:
    st.subheader("Build a campaign")
    f1, f2, f3, f4 = st.columns(4)
    offer_cost = f1.slider(f"Offer cost ({cur})", 0.0, 100.0, float(camp_cfg["offer_cost"]), 1.0)
    save_rate = f2.slider("Save rate (measured by backtest)", 0.0, 0.6, float(camp_cfg["save_rate"]), 0.01)
    divisions = f3.multiselect("Divisions", sorted(scores["division"].unique().to_list()))
    drivers_sel = f4.multiselect("Primary driver", sorted(FAMILY_ACTIONS))
    sim_cfg = dict(camp_cfg, offer_cost=offer_cost, save_rate=save_rate)

    ev = expected_net(scores["churn_probability"].to_numpy(), scores["arpu_3m"].to_numpy(), sim_cfg)
    sim = scores.with_columns(sim_value=pl.Series(ev)).filter(
        (pl.col("sim_value") > 0) & ~pl.col("sleeping_dog")
    )
    if divisions:
        sim = sim.filter(pl.col("division").is_in(divisions))
    if drivers_sel:
        sim = sim.filter(pl.col("primary_driver").is_in(drivers_sel))
    sim = sim.sort("sim_value", descending=True)
    budget = st.slider(f"Budget cap ({cur}, 0 = none)", 0, 2_000_000, 0, 10_000)
    if budget:
        sim = sim.head(int(budget // max(contact_cost(sim_cfg), 1e-9)))

    c = st.columns(4)
    kpi(c[0], f"{sim.height:,}", "subscribers to contact")
    kpi(c[1], f"{sim.height * contact_cost(sim_cfg) / 1e3:,.0f}k {cur}", "campaign cost")
    kpi(c[2], f"{sim['sim_value'].sum() / 1e3:,.0f}k {cur}", "expected net value", accent=True)
    kpi(c[3], f"{int(camp_cfg.get('holdout_share', 0.1) * sim.height):,}", "held out to measure ROI")
    st.write("")
    cols = [
        "sub_id",
        "division",
        "risk_band",
        "churn_probability",
        "uplift",
        "arpu_3m",
        "primary_driver",
        "recommended_action",
        "sim_value",
        "arm",
    ]
    st.dataframe(
        sim.select([c for c in cols if c in sim.columns]).head(1000).to_pandas(),
        use_container_width=True,
        hide_index=True,
        column_config={
            "churn_probability": st.column_config.ProgressColumn(
                "churn risk", min_value=0, max_value=1, format="%.2f"
            ),
            "sim_value": st.column_config.NumberColumn(f"expected net ({cur})", format="%.0f"),
            "arpu_3m": st.column_config.NumberColumn(f"ARPU ({cur})", format="%.0f"),
            "uplift": st.column_config.NumberColumn("uplift", format="%.3f"),
        },
    )
    st.download_button(
        "Download campaign list (CSV)",
        sim.select([c for c in cols if c in sim.columns]).write_csv(),
        file_name=f"campaign_{state['month']}.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------- model health
with health:
    tm = meta.get("test_metrics") or {}
    c = st.columns(5)
    kpi(c[0], f"{tm.get('roc_auc', float('nan')):.3f}", "ROC-AUC, out-of-time test")
    kpi(c[1], f"{tm.get('pr_auc', float('nan')):.3f}", "PR-AUC")
    kpi(c[2], f"{tm.get('capture_top10', float('nan')):.0%}", "churners in top 10%")
    drift = state["drift"] or {}
    kpi(
        c[3],
        f"{drift.get('score_psi', float('nan')):.3f}",
        f"score PSI ({drift.get('score_status', 'n/a')})",
        accent=drift.get("score_status") not in (None, "stable"),
    )
    dq = state["dq"] or {}
    kpi(c[4], str(dq.get("status", "n/a")).upper(), "data contract")
    st.write("")
    left, right = st.columns(2)
    with left:
        if drift.get("features"):
            top = pl.DataFrame(drift["features"][:12]).select("feature", "psi")
            bars(top, "psi", "feature", "Feature drift vs training (PSI)", fmt=".2f")
            st.caption("PSI < 0.10 stable · 0.10–0.25 investigate · > 0.25 retrain")
    with right:
        st.subheader("Registry")
        if state["versions"]:
            st.dataframe(
                pl.DataFrame(state["versions"])
                .select(
                    [
                        c
                        for c in (
                            "version",
                            "status",
                            "test_cutoff",
                            "roc_auc",
                            "pr_auc",
                            "capture_top10",
                            "decision",
                        )
                        if c in pl.DataFrame(state["versions"]).columns
                    ]
                )
                .reverse()
                .to_pandas(),
                use_container_width=True,
                hide_index=True,
            )
        st.subheader("Realised performance (backtests)")
        rows = [
            {
                "month": b["month"],
                "status": b["status"],
                "ROC-AUC": b["realised"]["roc_auc"],
                "capture@10%": b["realised"]["capture_top10"],
                "prevented / 1,000": (b.get("campaign") or {}).get("prevented_per_1000"),
                "measured save rate": (b.get("campaign") or {}).get("measured_save_rate"),
            }
            for b in state["backtests"]
            if b
        ]
        if rows:
            st.dataframe(pl.DataFrame(rows).to_pandas(), use_container_width=True, hide_index=True)
        else:
            st.caption("Run `churn backtest --month <scored month>` once its outcome window has passed.")

# ------------------------------------------------------------------------- subscriber lookup
with lookup:
    st.subheader("Subscriber lookup")
    sid = st.text_input("Subscriber id", placeholder="pseudonymised id from the CRM")
    if sid:
        try:
            row = scores.filter(pl.col("sub_id") == int(sid))
        except ValueError:
            row = pl.DataFrame()
        if row.height == 0:
            st.warning("Not in the active scored base for this month.")
        else:
            r = row.row(0, named=True)
            c = st.columns(4)
            kpi(
                c[0],
                f"{r['churn_probability']:.0%}",
                f"churn risk · {r['risk_band']}",
                accent=r["risk_band"] == "critical",
            )
            kpi(c[1], f"{r['arpu_3m']:,.0f} {cur}", "ARPU (3-month average)")
            up = r.get("uplift")
            kpi(c[2], "—" if up is None or np.isnan(up) else f"{up * 100:+.1f} pts", "offer effect (uplift)")
            kpi(c[3], "Yes" if r["target"] else "No", "in this month's campaign")
            st.write("")
            reasons = [x for x in (r.get("reason_1"), r.get("reason_2")) if x]
            if reasons:
                st.markdown("**Why:** " + " · ".join(FAMILY_LABELS.get(x, x) for x in reasons))
            if r.get("recommended_action"):
                st.markdown(f"**Next best action:** {r['recommended_action']}")
            if r.get("sleeping_dog"):
                st.error("Do not contact: the uplift model predicts an offer would push this subscriber out.")

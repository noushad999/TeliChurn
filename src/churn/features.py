"""Point-in-time feature snapshots and churn labels (Polars).

A snapshot at cutoff month T contains every subscriber with activity in T. Features only use
data up to the last day of T. The label looks strictly forward:

    churn = 1  if the subscriber has NO usage and NO recharge in months T+1 .. T+inactivity_months

This is the standard prepaid definition (60-90 days of zero revenue-generating activity). There
is no "churn flag" column to cheat with: churn is observed as silence.

Optional sources enrich the snapshot when present: per-operator off-net minutes (competitor
pressure) and the CDR contact graph (social contagion: contacts who recently went silent).
Optional usage KPIs (active days, SMS, complaints, call drops ...) and the network/market table
are used when an operator provides them. Features for missing inputs are simply not built.
"""

from __future__ import annotations

from datetime import date, datetime, time

import polars as pl

from churn.config import add_months, month_end, month_label, parse_month

SUM_METRICS = [
    "revenue",
    "voice_min",
    "onnet_min",
    "data_mb",
    "sms_cnt",
    "days_active",
    "data_pack_cnt",
    "emergency_loan_cnt",
    "complaint_cnt",
]
MEAN_METRICS = ["drop_call_rate", "data_speed_mbps"]
TREND_METRICS = ["revenue", "voice_min", "data_mb", "days_active", "data_pack_cnt"]

COMPETITOR_COLS = ["offnet_robi_min", "offnet_banglalink_min", "offnet_teletalk_min"]
CATEGORICAL = ["division", "gender", "plan_type", "handset", "top_competitor"]
KEY = ["sub_id", "snapshot"]
LABEL = "churn"


def build_snapshot(
    tables: dict[str, pl.DataFrame],
    cutoff: str | date,
    lookback_months: int = 3,
    inactivity_months: int = 2,
    with_label: bool = True,
    label_events: tuple[str, ...] = ("usage", "recharge"),
) -> pl.DataFrame:
    cut = parse_month(cutoff)
    cut_end = datetime.combine(month_end(cut), time.max)
    first = add_months(cut, -(lookback_months - 1))
    usage = tables["usage_monthly"]
    subs = tables["subscribers"]

    # months back from cutoff: 0 = cutoff month, 1 = previous month, ...
    lb = usage.filter(pl.col("month").is_between(first, cut)).with_columns(
        mb=((cut.year * 12 + cut.month) - (pl.col("month").dt.year() * 12 + pl.col("month").dt.month())).cast(
            pl.Int8
        )
    )
    active = lb.filter(pl.col("mb") == 0).select("sub_id")
    present = set(usage.columns)
    sum_metrics = [m for m in SUM_METRICS if m in present]
    trend_metrics = [m for m in TREND_METRICS if m in present]
    has_activity_days = "last_active_day" in present

    aggs = [pl.len().alias("months_active_lb")]
    for m in sum_metrics:
        for k in range(lookback_months):
            aggs.append(pl.col(m).filter(pl.col("mb") == k).sum().alias(f"{m}_m{k}"))
        aggs.append(pl.col(m).sum().truediv(lookback_months).alias(f"{m}_avg{lookback_months}m"))
    for m in (m for m in MEAN_METRICS if m in present):
        aggs.append(pl.col(m).filter(pl.col("mb") == 0).mean().alias(f"{m}_m0"))
        aggs.append(pl.col(m).mean().alias(f"{m}_avg{lookback_months}m"))
    if has_activity_days:
        aggs.append(pl.col("last_active_day").filter(pl.col("mb") == 0).first().alias("last_active_day_m0"))
    aggs.append((pl.col("onnet_min").sum() / (pl.col("voice_min").sum() + 1e-6)).alias("onnet_share"))
    has_competitor = all(c in usage.columns for c in COMPETITOR_COLS)
    if has_competitor:
        lb = lb.with_columns(competitor_min=pl.sum_horizontal(COMPETITOR_COLS))
        for k in range(lookback_months):
            in_k = pl.col("mb") == k
            aggs.append(
                (
                    pl.col("competitor_min").filter(in_k).sum() / (pl.col("voice_min").filter(in_k).sum() + 1)
                ).alias(f"competitor_share_m{k}")
            )
        for c in COMPETITOR_COLS:
            aggs.append(pl.col(c).sum().alias(f"_{c}"))
    usage_feats = lb.join(active, on="sub_id", how="semi").group_by("sub_id").agg(aggs)
    if has_competitor:
        usage_feats = _competitor_features(usage_feats, lookback_months)

    df = (
        active.join(subs, on="sub_id", how="left")
        .join(usage_feats, on="sub_id", how="left")
        .with_columns(
            tenure_months=(
                (cut.year * 12 + cut.month)
                - (pl.col("activation_month").dt.year() * 12 + pl.col("activation_month").dt.month())
            ).cast(pl.Int16),
        )
    )
    if has_activity_days:
        df = df.with_columns(
            days_since_last_activity=(month_end(cut).day - pl.col("last_active_day_m0")).cast(pl.Int16)
        ).drop("last_active_day_m0")

    # months before activation are unknown, not zero
    null_out = []
    for m in sum_metrics:
        for k in range(1, lookback_months):
            null_out.append(
                pl.when(pl.col("tenure_months") >= k)
                .then(pl.col(f"{m}_m{k}"))
                .otherwise(None)
                .alias(f"{m}_m{k}")
            )
    df = df.with_columns(null_out)
    trends = []
    for m in trend_metrics:
        prev = pl.mean_horizontal([pl.col(f"{m}_m{k}") for k in range(1, lookback_months)])
        trends.append(((pl.col(f"{m}_m0") + 1) / (prev + 1)).alias(f"{m}_trend"))
        trends.append((pl.col(f"{m}_m0") - pl.col(f"{m}_m1")).alias(f"{m}_delta_1m"))
    df = df.with_columns(trends)
    if "days_active" in present:
        df = df.with_columns(
            data_mb_per_active_day=pl.col("data_mb_m0") / pl.col("days_active_m0"),
            revenue_per_active_day=pl.col("revenue_m0") / pl.col("days_active_m0"),
        )

    df = df.join(_recharge_features(tables["recharges"], cut_end), on="sub_id", how="left")
    if tables.get("network_market") is not None:
        df = df.join(
            _market_features(tables["network_market"], cut, lookback_months), on="division", how="left"
        )
    if tables.get("contacts") is not None:
        df = df.join(_graph_features(tables["contacts"], usage, cut), on="sub_id", how="left")

    if with_label:
        df = df.join(build_labels(tables, cut, inactivity_months, label_events), on="sub_id", how="left")

    df = df.with_columns(
        snapshot=pl.lit(month_label(cut)),
        urban=pl.col("urban").cast(pl.Int8),
        dual_sim=pl.col("dual_sim").cast(pl.Int8),
        app_user=pl.col("app_user").cast(pl.Int8),
        mfs_user=pl.col("mfs_user").cast(pl.Int8),
    ).drop("activation_month")
    front = KEY + ([LABEL] if with_label else [])
    return df.select(front + [c for c in df.columns if c not in front]).sort("sub_id")


def build_labels(
    tables: dict[str, pl.DataFrame],
    cutoff: str | date,
    inactivity_months: int = 2,
    label_events: tuple[str, ...] = ("usage", "recharge"),
) -> pl.DataFrame:
    """Churn label for every subscriber active in the cutoff month: silent for the next N months.

    `label_events` sets what counts as "not silent". The default (usage or a recharge) matches the
    revenue-generating-event definition most finance teams use; ("usage",) matches definitions
    that ignore top-ups, as in the public Indian telecom case study.
    """
    cut = parse_month(cutoff)
    usage = tables["usage_monthly"]
    horizon_end = add_months(cut, inactivity_months)
    last_month = usage.select(pl.col("month").max()).item()
    if horizon_end > last_month:
        raise ValueError(
            f"Cannot label cutoff {month_label(cut)}: need data through {month_label(horizon_end)}, "
            f"have {month_label(last_month)}."
        )
    future = add_months(cut, 1)
    active = usage.filter(pl.col("month") == cut).select("sub_id").unique()
    parts = []
    if "usage" in label_events:
        parts.append(usage.filter(pl.col("month").is_between(future, horizon_end)).select("sub_id"))
    if "recharge" in label_events:
        parts.append(_recharge_ids(tables["recharges"], future, horizon_end))
    if not parts:
        raise ValueError("label_events must include 'usage' and/or 'recharge'")
    alive = pl.concat(parts).unique().with_columns(_alive=pl.lit(1))
    return (
        active.join(alive, on="sub_id", how="left")
        .with_columns(pl.col("_alive").is_null().cast(pl.Int8).alias(LABEL))
        .drop("_alive")
    )


def _recharge_ids(recharges: pl.DataFrame, future: date, horizon_end: date) -> pl.DataFrame:
    return recharges.filter(
        pl.col("ts").is_between(
            datetime.combine(future, time.min), datetime.combine(month_end(horizon_end), time.max)
        )
    ).select("sub_id")


def _recharge_features(recharges: pl.DataFrame, cut_end: datetime) -> pl.DataFrame:
    hist = recharges.filter(pl.col("ts") <= cut_end)
    last = hist.group_by("sub_id").agg(
        days_since_last_recharge=((pl.lit(cut_end) - pl.col("ts").max()).dt.total_hours() / 24).cast(
            pl.Float32
        )
    )
    w90 = (
        hist.filter(pl.col("ts") > cut_end - pl.duration(days=90))
        .sort("sub_id", "ts")
        .with_columns(
            age_days=(pl.lit(cut_end) - pl.col("ts")).dt.total_hours() / 24,
            gap_days=pl.col("ts").diff().over("sub_id").dt.total_hours() / 24,
        )
    )
    in30 = pl.col("age_days") <= 30
    agg = w90.group_by("sub_id").agg(
        rch_cnt_30d=in30.sum().cast(pl.Int16),
        rch_amt_30d=pl.col("amount").filter(in30).sum(),
        rch_cnt_90d=pl.len().cast(pl.Int16),
        rch_amt_90d=pl.col("amount").sum(),
        rch_avg_ticket_90d=pl.col("amount").mean(),
        rch_small_ticket_share_90d=(pl.col("amount") <= 20).mean(),
        rch_mean_gap_days_90d=pl.col("gap_days").mean(),
        rch_max_gap_days_90d=pl.col("gap_days").max(),
        rch_mfs_share_90d=(pl.col("channel") == "mfs").mean(),
        rch_app_share_90d=(pl.col("channel") == "app").mean(),
    )
    return last.join(agg, on="sub_id", how="left").with_columns(
        rch_amt_trend=(pl.col("rch_amt_30d").fill_null(0) + 1)
        / ((pl.col("rch_amt_90d").fill_null(0) - pl.col("rch_amt_30d").fill_null(0)) / 2 + 1),
        rch_cnt_30d=pl.col("rch_cnt_30d").fill_null(0),
        rch_amt_30d=pl.col("rch_amt_30d").fill_null(0),
        rch_cnt_90d=pl.col("rch_cnt_90d").fill_null(0),
        rch_amt_90d=pl.col("rch_amt_90d").fill_null(0),
    )


def _competitor_features(usage_feats: pl.DataFrame, lookback_months: int) -> pl.DataFrame:
    op_cols = [f"_{c}" for c in COMPETITOR_COLS]
    names = [c.removeprefix("offnet_").removesuffix("_min") for c in COMPETITOR_COLS]
    total = pl.sum_horizontal(op_cols)
    prev = pl.mean_horizontal([pl.col(f"competitor_share_m{k}") for k in range(1, lookback_months)])
    return usage_feats.with_columns(
        competitor_share_trend=pl.col("competitor_share_m0") - prev,
        top_competitor_share=pl.max_horizontal(op_cols) / (total + 1e-6),
        top_competitor=pl.when(total <= 0)
        .then(None)
        .otherwise(
            pl.concat_list(op_cols)
            .list.arg_max()
            .replace_strict(dict(enumerate(names)), return_dtype=pl.Utf8)
        ),
    ).drop(op_cols)


def _graph_features(contacts: pl.DataFrame, usage: pl.DataFrame, cut: date) -> pl.DataFrame:
    """Social contagion from the CDR contact graph, using only activity observed up to the cutoff.

    A contact counts as *gone* when they were active 2-3 months ago but silent in both of the
    last two months (T-1 and T): the same silence a churn label would later confirm.
    """
    months = [add_months(cut, -k) for k in range(4)]
    seen = usage.filter(pl.col("month").is_in(months)).select("sub_id", "month")
    status = seen.group_by("sub_id").agg(
        recent=pl.col("month").is_in(months[:2]).any(),
        earlier=pl.col("month").is_in(months[2:]).any(),
        active_m0=(pl.col("month") == months[0]).any(),
    )
    edges = contacts.join(status.rename({"sub_id": "contact_id"}), on="contact_id", how="left").with_columns(
        gone=(pl.col("earlier").fill_null(False) & ~pl.col("recent").fill_null(False)),
        active_m0=pl.col("active_m0").fill_null(False),
    )
    return edges.group_by("sub_id").agg(
        contacts_n=pl.len().cast(pl.Int16),
        contacts_active_share=pl.col("active_m0").mean(),
        contacts_gone_n=pl.col("gone").sum().cast(pl.Int16),
        contacts_gone_share=(pl.col("minutes") * pl.col("gone")).sum() / pl.col("minutes").sum(),
    )


def _market_features(market: pl.DataFrame, cut: date, lookback_months: int) -> pl.DataFrame:
    first = add_months(cut, -(lookback_months - 1))
    w = market.filter(pl.col("month").is_between(first, cut))
    return w.group_by("division").agg(
        div_drop_rate_m0=pl.col("avg_drop_call_rate").filter(pl.col("month") == cut).first(),
        div_drop_rate_change=pl.col("avg_drop_call_rate").filter(pl.col("month") == cut).first()
        - pl.col("avg_drop_call_rate").filter(pl.col("month") != cut).mean(),
        div_outage_m0=pl.col("network_outage").filter(pl.col("month") == cut).first().cast(pl.Int8),
        competitor_promo_m0=pl.col("competitor_promo").filter(pl.col("month") == cut).first().cast(pl.Int8),
        competitor_promo_lb=pl.col("competitor_promo").sum().cast(pl.Int8),
    )


def arpu_column(df: pl.DataFrame) -> str:
    """Name of the lookback-average revenue column (`revenue_avg<N>m`), whatever the lookback."""
    return next(c for c in df.columns if c.startswith("revenue_avg") and c.endswith("m"))


def feature_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in (*KEY, LABEL)]


def build_many(tables: dict[str, pl.DataFrame], cutoffs: list[str], **kwargs) -> pl.DataFrame:
    return pl.concat([build_snapshot(tables, c, **kwargs) for c in cutoffs], how="vertical_relaxed")

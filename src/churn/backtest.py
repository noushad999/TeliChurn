"""Realised-performance monitoring: once a scored month's outcome window has passed, measure it.

PSI tells you the inputs moved; only labels tell you the model got worse. Two months after a scoring
run, `churn backtest --month <scored month>` joins the stored scores with the churn that actually
happened and reports:

* realised ROC-AUC, PR-AUC, lift and capture against what the model promised on its test month,
  with a DEGRADED alert when capture@10% or PR-AUC fall past the configured thresholds
* the campaign's measured effect: churn among treated vs held-out targets (the holdout arm that
  `churn score` reserves), i.e. a measured save rate instead of an assumed one
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from churn import ingest
from churn.evaluate import classification_metrics
from churn.features import build_labels
from churn.registry import Registry


def run(cfg: dict, month: str) -> dict:
    paths = cfg["paths"]
    scores_path = Path(paths["scores_dir"]) / f"scores_{month}.parquet"
    if not scores_path.exists():
        raise FileNotFoundError(f"No stored scores for {month}. Run `churn score --month {month}` first.")
    scores = pl.read_parquet(scores_path)
    tables, _ = ingest.load(cfg)
    labelled = scores.join(
        build_labels(
            tables,
            month,
            cfg["churn"]["inactivity_months"],
            tuple(cfg["churn"].get("label_events", ("usage", "recharge"))),
        ),
        on="sub_id",
        how="inner",
    )
    y = labelled["churn"].to_numpy()
    p = labelled["churn_probability"].to_numpy()
    realised = classification_metrics(y, p)

    version = scores["model_version"][0] if "model_version" in scores.columns else None
    promised = _promised_metrics(paths, version)
    thresholds = cfg.get("monitoring", {"max_capture_drop": 0.05, "max_pr_auc_drop_rel": 0.2})
    alerts = []
    if promised:
        if realised["capture_top10"] < promised["capture_top10"] - thresholds["max_capture_drop"]:
            alerts.append(
                f"capture@10% {realised['capture_top10']:.1%} vs {promised['capture_top10']:.1%} promised"
            )
        if realised["pr_auc"] < promised["pr_auc"] * (1 - thresholds["max_pr_auc_drop_rel"]):
            alerts.append(f"PR-AUC {realised['pr_auc']:.3f} vs {promised['pr_auc']:.3f} promised")

    campaign = _campaign_effect(labelled, tables, month)
    report = {
        "month": month,
        "model_version": version,
        "subscribers": int(len(y)),
        "realised": realised,
        "promised_on_test_month": promised,
        "status": "DEGRADED" if alerts else "HEALTHY",
        "alerts": alerts,
        "campaign": campaign,
    }
    out = Path(paths["reports_dir"]) / f"backtest_{month}.json"
    out.write_text(json.dumps(report, indent=2))
    _print(report, out)
    return report


def _promised_metrics(paths: dict, version: str | None) -> dict | None:
    if not version:
        return None
    reg = Registry(paths["registry_dir"])
    meta = reg.path(version) / "metadata.json"
    if not meta.exists():
        return None
    return json.loads(meta.read_text()).get("test_metrics")


def _campaign_effect(labelled: pl.DataFrame, tables: dict, month: str) -> dict | None:
    """Measured effect from the executed campaign's randomised arms (treatment vs holdout)."""
    arms = None
    camp = tables.get("campaigns")
    if camp is not None:
        executed = camp.filter(pl.col("campaign_month") == pl.lit(f"{month}-01").str.to_date())
        if executed.height:
            arms = labelled.join(executed.select("sub_id", "treatment"), on="sub_id", how="inner")
    if arms is None and "arm" in labelled.columns:
        arms = labelled.filter(pl.col("arm").is_not_null()).with_columns(
            treatment=pl.col("arm") == "treatment"
        )
    if arms is None or arms.height == 0:
        return None
    g = {
        r["treatment"]: r
        for r in arms.group_by("treatment").agg(pl.len(), pl.col("churn").mean()).iter_rows(named=True)
    }
    if True not in g or False not in g:
        return None
    treated, holdout = g[True]["churn"], g[False]["churn"]
    n_t, n_h = g[True]["len"], g[False]["len"]
    se = np.sqrt(treated * (1 - treated) / n_t + holdout * (1 - holdout) / n_h)
    effect = holdout - treated
    return {
        "treated": int(n_t),
        "holdout": int(n_h),
        "churn_treated": float(treated),
        "churn_holdout": float(holdout),
        "prevented_per_1000": float(effect * 1000),
        "ci95_per_1000": [float((effect - 1.96 * se) * 1000), float((effect + 1.96 * se) * 1000)],
        "measured_save_rate": float(effect / holdout) if holdout > 0 else None,
    }


def _print(r: dict, out: Path) -> None:
    m = r["realised"]
    print(
        f"Backtest {r['month']} (model {r['model_version']}, {r['subscribers']:,} subscribers): {r['status']}"
    )
    print(
        f"  realised ROC-AUC {m['roc_auc']:.3f} · PR-AUC {m['pr_auc']:.3f} · "
        f"lift@10% {m['lift_top10']:.2f} · capture@10% {m['capture_top10']:.1%}"
    )
    if p := r["promised_on_test_month"]:
        print(
            f"  promised ROC-AUC {p['roc_auc']:.3f} · PR-AUC {p['pr_auc']:.3f} · "
            f"lift@10% {p['lift_top10']:.2f} · capture@10% {p['capture_top10']:.1%}"
        )
    for a in r["alerts"]:
        print(f"  ALERT: {a}")
    if c := r["campaign"]:
        lo, hi = c["ci95_per_1000"]
        print(
            f"  campaign: churn {c['churn_treated']:.2%} treated vs {c['churn_holdout']:.2%} holdout → "
            f"{c['prevented_per_1000']:.1f} prevented per 1,000 (95% CI {lo:.1f} to {hi:.1f}); "
            f"measured save rate {c['measured_save_rate']:.0%}"
        )
    print(f"  -> {out}")

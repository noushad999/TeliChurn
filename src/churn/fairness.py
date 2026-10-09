"""Segment and fairness report: does the model work, and treat people consistently, across groups?

For each segment (division, urban/rural, gender, plan, handset, tenure band) on the out-of-time test
month we report:

* calibration   - mean predicted churn vs observed churn (a gap means the probabilities that drive
                  targeting are wrong for that group)
* ranking       - ROC-AUC within the segment
* selection     - share of the segment the campaign policy would contact
* reach         - share of the segment's churners the policy reaches (equal-opportunity view)

Flags: calibration gap above `max_calibration_gap` (absolute), within-segment AUC below
`min_auc`, or reach below `min_reach_ratio` × the overall reach. Contact rates are expected to
differ (they follow risk and value); reach and calibration gaps are what need a human look.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

SEGMENTS = {
    "division": pl.col("division"),
    "urban": pl.when(pl.col("urban") == 1)
    .then(pl.lit("urban"))
    .when(pl.col("urban") == 0)
    .then(pl.lit("rural")),
    "gender": pl.col("gender"),
    "plan_type": pl.col("plan_type"),
    "handset": pl.col("handset"),
    "tenure": pl.when(pl.col("tenure_months") < 3)
    .then(pl.lit("0-2 months"))
    .when(pl.col("tenure_months") < 12)
    .then(pl.lit("3-11 months"))
    .when(pl.col("tenure_months") < 36)
    .then(pl.lit("1-3 years"))
    .otherwise(pl.lit("3+ years")),
}
DEFAULTS = {"max_calibration_gap": 0.02, "min_auc": 0.75, "min_reach_ratio": 0.5, "min_segment_size": 500}


def segment_report(test: pl.DataFrame, p: np.ndarray, selected: np.ndarray, cfg: dict | None = None) -> dict:
    th = DEFAULTS | (cfg or {})
    y = test["churn"].to_numpy()
    overall_reach = float(selected[y == 1].mean()) if y.sum() else 0.0
    base = test.with_columns(_p=pl.Series(p), _sel=pl.Series(selected), _y=pl.Series(y))
    rows, flags = [], []
    for name, expr in SEGMENTS.items():
        source_cols = expr.meta.root_names()
        if any(c not in test.columns or test[c].null_count() == test.height for c in source_cols):
            continue
        g = base.with_columns(_seg=expr.cast(pl.Utf8)).filter(pl.col("_seg").is_not_null())
        for seg, part in g.group_by("_seg"):
            if part.height < th["min_segment_size"]:
                continue
            yy, pp, ss = part["_y"].to_numpy(), part["_p"].to_numpy(), part["_sel"].to_numpy()
            auc = float(roc_auc_score(yy, pp)) if 0 < yy.sum() < len(yy) else None
            reach = float(ss[yy == 1].mean()) if yy.sum() else None
            row = {
                "dimension": name,
                "segment": seg[0],
                "n": int(part.height),
                "churn_rate": float(yy.mean()),
                "mean_predicted": float(pp.mean()),
                "calibration_gap": float(pp.mean() - yy.mean()),
                "roc_auc": auc,
                "contact_rate": float(ss.mean()),
                "reach": reach,
            }
            rows.append(row)
            issues = []
            if abs(row["calibration_gap"]) > th["max_calibration_gap"]:
                issues.append(f"calibration gap {row['calibration_gap']:+.3f}")
            if auc is not None and auc < th["min_auc"]:
                issues.append(f"AUC {auc:.3f}")
            if reach is not None and overall_reach and reach < th["min_reach_ratio"] * overall_reach:
                issues.append(f"reach {reach:.1%} vs {overall_reach:.1%} overall")
            if issues:
                flags.append({"dimension": name, "segment": seg[0], "issues": issues})
    rows.sort(key=lambda r: (r["dimension"], r["segment"]))
    return {"overall_reach": overall_reach, "thresholds": th, "segments": rows, "flags": flags}


def markdown_table(report: dict) -> str:
    lines = [
        "| Dimension | Segment | Subscribers | Churn | Predicted | ROC-AUC | Contacted | Churners reached |",
        "|:--|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for r in report["segments"]:
        auc = "–" if r["roc_auc"] is None else f"{r['roc_auc']:.3f}"
        reach = "–" if r["reach"] is None else f"{r['reach']:.1%}"
        lines.append(
            f"| {r['dimension']} | {r['segment']} | {r['n']:,} | {r['churn_rate']:.1%} | {r['mean_predicted']:.1%} "
            f"| {auc} | {r['contact_rate']:.1%} | {reach} |"
        )
    return "\n".join(lines)

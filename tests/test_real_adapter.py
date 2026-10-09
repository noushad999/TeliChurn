"""The public-dataset adapter maps the Indian telecom schema onto the data contract."""

import importlib.util
from pathlib import Path

import polars as pl

from churn.contract import conform, validate
from churn.features import build_snapshot

SPEC = importlib.util.spec_from_file_location(
    "prepare", Path(__file__).parents[1] / "examples" / "real_data" / "prepare_indian_telco.py"
)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def _fake_raw(n: int = 300) -> pl.DataFrame:
    cols = {
        "mobile_number": list(range(7000000000, 7000000000 + n)),
        "circle_id": [109] * n,
        "aon": [400] * n,
    }
    for m in (6, 7, 8, 9):
        silent = m == 9
        churner = [i % 10 == 0 for i in range(n)]  # 10% go silent in September
        use = [0.0 if (silent and c) else 100.0 + i for i, c in enumerate(churner)]
        for c in ("total_og_mou", "total_ic_mou", "onnet_mou", "vol_2g_mb", "vol_3g_mb", "arpu"):
            cols[f"{c}_{m}"] = use
        for c in ("monthly_2g", "sachet_2g", "monthly_3g", "sachet_3g"):
            cols[f"{c}_{m}"] = [1] * n
        cols[f"total_rech_num_{m}"] = [3] * n
        cols[f"total_rech_amt_{m}"] = [300] * n
        cols[f"last_day_rch_amt_{m}"] = [100] * n
        cols[f"date_of_last_rech_{m}"] = [f"{m}/25/2014"] * n
    return pl.DataFrame(cols)


def test_adapter_output_passes_contract_and_labels():
    tables = prepare.convert(_fake_raw())
    t, report = conform(tables)
    report = validate(t, report)
    assert report.ok, report.summary()
    # 3 recharges a month: 1 exact (last of month) + 2 undated, amounts preserved
    aug = t["recharges"].filter(pl.col("ts").dt.month() == 8)
    assert aug.height == 3 * 300 and abs(aug["amount"].sum() - 300 * 300) < 1e-6
    snap = build_snapshot(t, "2014-08", lookback_months=2, inactivity_months=1, label_events=("usage",))
    assert snap.height == 300
    assert abs(snap["churn"].mean() - 0.10) < 1e-9  # exactly the silent 10%

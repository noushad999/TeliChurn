import polars as pl
import pytest

from churn import ingest
from churn.config import load_config
from churn.contract import DataContractError, conform, validate
from churn.features import build_snapshot


def _checks(report) -> set[tuple[str, str, str]]:
    return {(i.table, i.check, i.severity) for i in report.issues}


def test_clean_simulated_data_passes(tables):
    t, report = conform(tables)
    report = validate(t, report)
    assert report.ok, report.summary()


def test_contract_catches_broken_data(tables):
    broken = dict(tables)
    usage = tables["usage_monthly"]
    broken["usage_monthly"] = pl.concat(
        [usage, usage.head(50)]  # duplicate keys
    ).with_columns(voice_min=pl.when(pl.col("sub_id") % 50 == 0).then(-5.0).otherwise(pl.col("voice_min")))
    broken["subscribers"] = tables["subscribers"].with_columns(
        plan_type=pl.when(pl.col("sub_id") % 10 == 0).then(pl.lit("hybrid")).otherwise(pl.col("plan_type"))
    )
    broken["recharges"] = tables["recharges"].drop("amount")
    broken["contacts"] = tables["contacts"].with_columns(contact_id=pl.col("contact_id") + 10_000_000)
    t, report = conform(broken)
    report = validate(t, report)
    checks = _checks(report)
    assert not report.ok
    assert ("usage_monthly", "duplicate_key", "error") in checks
    assert ("usage_monthly", "out_of_range", "error") in checks
    assert ("subscribers", "unknown_value", "error") in checks
    assert ("recharges", "missing_column", "error") in checks
    assert ("contacts", "orphan_ids", "error") in checks


def test_optional_sources_only_warn(tables):
    t, report = conform({k: v for k, v in tables.items() if k not in ("contacts", "campaigns")})
    report = validate(t, report)
    assert report.ok
    assert ("contacts", "missing_table", "warning") in _checks(report)


def test_operator_csv_with_mapping_and_pseudonymisation(tables, tmp_path, monkeypatch):
    """Operator-style extracts: own column names, codes, string MSISDNs -> canonical, hashed ids."""
    msisdn = (
        pl.col("sub_id").cast(pl.Utf8).str.zfill(8).map_elements(lambda s: "88017" + s, return_dtype=pl.Utf8)
    )
    subs = tables["subscribers"].with_columns(
        msisdn, plan_type=pl.col("plan_type").replace({"prepaid": "PRE", "postpaid": "POST"})
    )
    subs.rename({"sub_id": "MSISDN", "plan_type": "PRICE_PLAN"}).write_csv(tmp_path / "subscribers.csv")
    tables["usage_monthly"].with_columns(msisdn).rename(
        {"sub_id": "MSISDN", "voice_min": "TOTAL_MOU"}
    ).write_csv(tmp_path / "usage_monthly.csv")
    tables["recharges"].with_columns(msisdn).rename({"sub_id": "MSISDN"}).write_csv(
        tmp_path / "recharges.csv"
    )
    tables["network_market"].write_csv(tmp_path / "network_market.csv")

    monkeypatch.setenv("CHURN_PII_SALT", "test-salt")
    cfg = load_config(
        overrides={
            "source": {
                "type": "csv",
                "path": str(tmp_path),
                "pseudonymize": {"enabled": True, "salt_env": "CHURN_PII_SALT", "columns": ["sub_id"]},
                "tables": {
                    "subscribers": {
                        "columns": {"MSISDN": "sub_id", "PRICE_PLAN": "plan_type"},
                        "values": {"plan_type": {"PRE": "prepaid", "POST": "postpaid"}},
                    },
                    "usage_monthly": {"columns": {"MSISDN": "sub_id", "TOTAL_MOU": "voice_min"}},
                    "recharges": {"columns": {"MSISDN": "sub_id"}},
                },
            }
        }
    )
    loaded, report = ingest.load(cfg)
    assert report.ok, report.summary()
    assert loaded["subscribers"]["sub_id"].dtype == pl.Int64
    assert set(loaded["subscribers"]["plan_type"].unique()) <= {"prepaid", "postpaid"}
    # same MSISDN -> same pseudonymous id in every table (joins still work)
    usage_ids = loaded["usage_monthly"].select("sub_id").unique()
    assert usage_ids.join(loaded["subscribers"].select("sub_id"), on="sub_id", how="anti").height == 0
    snap = build_snapshot(loaded, "2025-06")
    assert snap.height > 1000 and snap["churn"].mean() > 0


def test_strict_load_raises_on_contract_failure(tmp_path):
    pl.DataFrame({"sub_id": [1]}).write_csv(tmp_path / "subscribers.csv")
    cfg = load_config(overrides={"source": {"type": "csv", "path": str(tmp_path)}})
    with pytest.raises(DataContractError):
        ingest.load(cfg)


def test_config_extends(tmp_path):
    (tmp_path / "base.yaml").write_text("seed: 1\npaths: {raw_dir: a, model_dir: m}\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\npaths: {raw_dir: b}\n")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg == {"seed": 1, "paths": {"raw_dir": "b", "model_dir": "m"}}

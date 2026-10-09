"""Ingestion: read operator extracts, map them onto the data contract, pseudonymise, validate.

Configured under `source:` in the config file (see `configs/source_example.yaml`):

    source:
      type: parquet | csv | sql | simulated
      path: /data/exports              # parquet / csv: directory with one file per table
      uri_env: CHURN_DB_URI            # sql: connection URI read from this environment variable
      tables:
        usage_monthly:
          file: USAGE_SUMMARY.csv      # or `query: SELECT ...` for sql
          columns: {MSISDN: sub_id, MONTH_KEY: month, TOTAL_MOU: voice_min}
          values: {plan_type: {PRE: prepaid, POST: postpaid}}
      pseudonymize:
        enabled: true
        salt_env: CHURN_PII_SALT       # secret salt; never stored with the data
        columns: [sub_id, contact_id]

Raw MSISDNs never reach the feature store: identifier columns are replaced by a salted BLAKE2b
hash (64-bit), so joins still work across tables but numbers cannot be read back.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import polars as pl

from churn.contract import CONTRACT, DataContractError, Report, conform, validate
from churn.simulate import load_tables


def pseudonymize(series: pl.Series, salt: str) -> pl.Series:
    """Salted BLAKE2b hash of identifiers to signed 64-bit ints (hashes each unique value once)."""
    uniques = series.cast(pl.Utf8).unique().drop_nulls()
    key = salt.encode()
    hashed = np.array(
        [
            int.from_bytes(hashlib.blake2b(v.encode(), digest_size=8, key=key).digest(), "big", signed=True)
            for v in uniques.to_list()
        ],
        dtype=np.int64,
    )
    lookup = pl.DataFrame({"_raw": uniques, "_id": hashed})
    return (
        pl.DataFrame({"_raw": series.cast(pl.Utf8)})
        .join(lookup, on="_raw", how="left", maintain_order="left")["_id"]
        .alias(series.name)
    )


def _read(name: str, spec: dict, src: dict) -> pl.DataFrame | None:
    kind = src["type"]
    if kind == "sql":
        query = spec.get("query") or f"SELECT * FROM {spec.get('table', name)}"
        uri = os.environ.get(src.get("uri_env", "CHURN_DB_URI"))
        if not uri:
            raise RuntimeError(f"Set {src.get('uri_env', 'CHURN_DB_URI')} to the database connection URI")
        try:
            return pl.read_database_uri(query, uri)
        except ImportError as exc:  # connectorx is an optional dependency
            raise RuntimeError("SQL sources need `pip install telco-churn[sql]`") from exc
    base = Path(src["path"])
    path = base / spec.get("file", f"{name}.{'parquet' if kind == 'parquet' else 'csv'}")
    if not path.exists():
        return None
    if kind == "parquet":
        return pl.read_parquet(path)
    return pl.read_csv(path, try_parse_dates=False, infer_schema_length=10_000)


def _map(df: pl.DataFrame, spec: dict) -> pl.DataFrame:
    if cols := spec.get("columns"):
        df = df.rename({k: v for k, v in cols.items() if k in df.columns})
    for col, mapping in (spec.get("values") or {}).items():
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Utf8).replace(mapping))
    return df


def load(cfg: dict, strict: bool = True) -> tuple[dict[str, pl.DataFrame], Report]:
    """Load, map, pseudonymise, conform and validate all tables. Raises on contract errors if `strict`."""
    src = cfg.get("source") or {"type": "simulated"}
    if src["type"] == "simulated":
        raw = load_tables(cfg["paths"]["raw_dir"])
    else:
        raw = {}
        for name in CONTRACT:
            spec = (src.get("tables") or {}).get(name, {})
            df = _read(name, spec, src)
            if df is not None:
                raw[name] = _map(df, spec)
        pz = src.get("pseudonymize") or {}
        if pz.get("enabled"):
            salt = os.environ.get(pz.get("salt_env", "CHURN_PII_SALT"))
            if not salt:
                raise RuntimeError(
                    f"Pseudonymisation enabled but {pz.get('salt_env', 'CHURN_PII_SALT')} is not set"
                )
            id_cols = set(pz.get("columns", ["sub_id", "contact_id"]))
            for name, df in raw.items():
                for col in id_cols & set(df.columns):
                    df = df.with_columns(pseudonymize(df[col], salt))
                raw[name] = df

    tables, report = conform(raw)
    report = validate(tables, report)
    if strict and not report.ok:
        raise DataContractError(report)
    return tables, report

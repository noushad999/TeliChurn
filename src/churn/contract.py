"""Data contract: the canonical schema every source must satisfy, and the checks that enforce it.

Operators export data from different warehouses with different names, types and codes. The
pipeline only ever sees the canonical tables below. `conform` casts a source into this shape, and
`validate` refuses to train or score on data that breaks the contract:

* errors (block the run):  missing required tables/columns, uncastable types, nulls in required
  fields, duplicate keys, out-of-range or unknown values above tolerance, orphan subscriber ids
* warnings (logged):       missing optional sources, gaps in the monthly series, sudden volume
  jumps month over month, small shares of out-of-range values
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import polars as pl

POLARS_TYPES = {
    "id": pl.Int64,
    "int": pl.Int32,
    "float": pl.Float64,
    "str": pl.Utf8,
    "bool": pl.Boolean,
    "date": pl.Date,
    "datetime": pl.Datetime("ms"),
}
TOLERANCE = 0.001  # share of rows allowed to break a value rule before it becomes an error


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str
    required: bool = True
    nullable: bool = False
    min: float | None = None
    max: float | None = None
    allowed: tuple[str, ...] | None = None
    description: str = ""


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]
    key: tuple[str, ...]
    required: bool = True
    time_column: str | None = None
    description: str = ""

    def column(self, name: str) -> Column:
        return next(c for c in self.columns if c.name == name)


def _usage_min(name: str, required: bool = True, description: str = "") -> Column:
    return Column(name, "float", required=required, min=0, description=description)


CONTRACT: dict[str, Table] = {
    t.name: t
    for t in (
        Table(
            "subscribers",
            (
                Column("sub_id", "id", description="pseudonymised subscriber id (hash of MSISDN)"),
                Column("activation_month", "date", description="first day of the SIM activation month"),
                Column("division", "str"),
                Column("urban", "bool", nullable=True),
                Column("gender", "str", nullable=True, allowed=("M", "F")),
                Column("age", "int", nullable=True, min=10, max=100),
                Column("plan_type", "str", allowed=("prepaid", "postpaid")),
                Column(
                    "handset",
                    "str",
                    nullable=True,
                    allowed=("feature_phone", "smartphone_3g", "smartphone_4g"),
                ),
                Column("dual_sim", "bool", nullable=True),
                Column("app_user", "bool", nullable=True),
                Column("mfs_user", "bool", nullable=True),
            ),
            key=("sub_id",),
            description="one row per SIM",
        ),
        Table(
            "usage_monthly",
            (
                Column("sub_id", "id"),
                Column("month", "date", description="first day of the month"),
                _usage_min("voice_min"),
                _usage_min("onnet_min"),
                _usage_min("offnet_robi_min", required=False),
                _usage_min("offnet_banglalink_min", required=False),
                _usage_min("offnet_teletalk_min", required=False),
                Column("data_mb", "float", min=0),
                Column("sms_cnt", "int", required=False, min=0),
                Column("data_pack_cnt", "int", required=False, min=0),
                Column("revenue", "float", min=0),
                Column("days_active", "int", required=False, min=1, max=31),
                Column("last_active_day", "int", required=False, min=1, max=31),
                Column("emergency_loan_cnt", "int", required=False, min=0),
                Column("complaint_cnt", "int", required=False, min=0),
                Column("drop_call_rate", "float", required=False, nullable=True, min=0, max=1),
                Column("data_speed_mbps", "float", required=False, nullable=True, min=0),
            ),
            key=("sub_id", "month"),
            time_column="month",
            description="one row per subscriber-month with any revenue-generating activity",
        ),
        Table(
            "recharges",
            (
                Column("sub_id", "id"),
                Column("ts", "datetime"),
                Column("amount", "float", min=0.01),
                Column("channel", "str", nullable=True),
            ),
            key=(),
            time_column="ts",
            description="one row per top-up transaction",
        ),
        Table(
            "network_market",
            (
                Column("division", "str"),
                Column("month", "date"),
                Column("avg_drop_call_rate", "float", min=0, max=1),
                Column("network_outage", "bool"),
                Column("competitor_promo", "bool"),
            ),
            key=("division", "month"),
            required=False,
            time_column="month",
            description="division-month network KPIs and market intelligence (optional)",
        ),
        Table(
            "contacts",
            (
                Column("sub_id", "id"),
                Column("contact_id", "id"),
                Column("minutes", "float", min=0),
            ),
            key=("sub_id", "contact_id"),
            required=False,
            description="top on-net contacts per subscriber from CDRs (optional)",
        ),
        Table(
            "campaigns",
            (
                Column("campaign_month", "date"),
                Column("sub_id", "id"),
                Column("treatment", "bool"),
                Column("offer", "str", nullable=True),
            ),
            key=("campaign_month", "sub_id"),
            required=False,
            description="randomised retention campaigns: treatment vs holdout (optional)",
        ),
    )
}


@dataclass
class Issue:
    table: str
    check: str
    severity: str  # "error" | "warning"
    detail: str
    column: str | None = None
    rows: int | None = None


@dataclass
class Report:
    issues: list[Issue] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, table: str, check: str, severity: str, detail: str, column: str | None = None, rows=None):
        self.issues.append(Issue(table, check, severity, detail, column, None if rows is None else int(rows)))

    def to_dict(self) -> dict:
        return {
            "status": "pass" if self.ok else "fail",
            "stats": self.stats,
            "issues": [asdict(i) for i in self.issues],
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str))

    def summary(self) -> str:
        lines = [
            f"Data contract: {'PASS' if self.ok else 'FAIL'} ({len(self.errors)} errors, "
            f"{len(self.issues) - len(self.errors)} warnings)"
        ]
        for name, st in self.stats.items():
            span = f"  {st['first']} → {st['last']}" if st.get("first") else ""
            lines.append(f"  {name:<16}{st['rows']:>12,} rows{span}")
        for i in self.issues:
            where = f"{i.table}.{i.column}" if i.column else i.table
            rows = f" ({i.rows:,} rows)" if i.rows else ""
            lines.append(f"  [{i.severity.upper():<7}] {where}: {i.check} — {i.detail}{rows}")
        return "\n".join(lines)


class DataContractError(ValueError):
    def __init__(self, report: Report):
        super().__init__(report.summary())
        self.report = report


def conform(tables: dict[str, pl.DataFrame]) -> tuple[dict[str, pl.DataFrame], Report]:
    """Cast known columns to canonical types (non-strict) and record anything that failed to cast."""
    report = Report()
    out: dict[str, pl.DataFrame] = {}
    for name, spec in CONTRACT.items():
        df = tables.get(name)
        if df is None:
            if spec.required:
                report.add(name, "missing_table", "error", "required table not provided")
            else:
                report.add(
                    name, "missing_table", "warning", "optional source not provided; related features skipped"
                )
            continue
        casts = []
        for col in spec.columns:
            if col.name not in df.columns:
                sev = "error" if col.required else "warning"
                report.add(name, "missing_column", sev, "column not provided", col.name)
                continue
            target = POLARS_TYPES[col.dtype]
            if df.schema[col.name] != target:
                casts.append(_cast(pl.col(col.name), df.schema[col.name], target).alias(col.name))
        if casts:
            before = {c.name: df[c.name].null_count() for c in spec.columns if c.name in df.columns}
            df = df.with_columns(casts)
            for c, n0 in before.items():
                added = df[c].null_count() - n0
                if added > 0:
                    report.add(
                        name,
                        "type_cast",
                        "error",
                        f"{added:,} values could not be cast to {spec.column(c).dtype}",
                        c,
                        added,
                    )
        out[name] = df
    return out, report


def _cast(expr: pl.Expr, source: pl.DataType, target: pl.DataType) -> pl.Expr:
    if source == pl.Utf8 and target == pl.Date:
        return expr.str.to_date(strict=False)
    if source == pl.Utf8 and target == pl.Datetime("ms"):
        return expr.str.to_datetime(strict=False).cast(target)
    if source == pl.Utf8 and target == pl.Boolean:
        return expr.str.to_lowercase().replace_strict(
            {
                "true": True,
                "1": True,
                "y": True,
                "yes": True,
                "false": False,
                "0": False,
                "n": False,
                "no": False,
            },
            default=None,
            return_dtype=pl.Boolean,
        )
    return expr.cast(target, strict=False)


def validate(tables: dict[str, pl.DataFrame], report: Report | None = None) -> Report:
    report = report or Report()
    for name, spec in CONTRACT.items():
        df = tables.get(name)
        if df is None:
            continue
        stats = {"rows": df.height}
        if spec.time_column and spec.time_column in df.columns and df.height:
            stats["first"] = str(df[spec.time_column].min())
            stats["last"] = str(df[spec.time_column].max())
        report.stats[name] = stats
        if df.height == 0:
            report.add(name, "empty", "error" if spec.required else "warning", "table has no rows")
            continue
        _check_columns(df, spec, report)
        if spec.key and all(k in df.columns for k in spec.key):
            dup = df.height - df.select(spec.key).unique().height
            if dup:
                report.add(name, "duplicate_key", "error", f"key {spec.key} is not unique", rows=dup)
        if spec.time_column == "month" and "month" in df.columns:
            _check_months(df, name, report)
    _check_references(tables, report)
    return report


def _check_columns(df: pl.DataFrame, spec: Table, report: Report) -> None:
    n = df.height
    for col in spec.columns:
        if col.name not in df.columns:
            continue
        s = df[col.name]
        if not col.nullable and (nulls := s.null_count()):
            report.add(spec.name, "null", "error", "nulls in a required field", col.name, nulls)
        if col.min is not None or col.max is not None:
            bad = 0
            if col.min is not None:
                bad += int((s < col.min).sum() or 0)
            if col.max is not None:
                bad += int((s > col.max).sum() or 0)
            if bad:
                sev = "error" if bad / n > TOLERANCE else "warning"
                report.add(spec.name, "out_of_range", sev, f"outside [{col.min}, {col.max}]", col.name, bad)
        if col.allowed is not None:
            unknown = s.drop_nulls().filter(~s.drop_nulls().is_in(list(col.allowed)))
            if unknown.len():
                sev = "error" if unknown.len() / n > TOLERANCE else "warning"
                sample = ", ".join(map(str, unknown.unique().head(5).to_list()))
                report.add(
                    spec.name,
                    "unknown_value",
                    sev,
                    f"not in {col.allowed}: {sample}",
                    col.name,
                    unknown.len(),
                )


def _check_months(df: pl.DataFrame, name: str, report: Report) -> None:
    counts = df.group_by("month").len().sort("month")
    months = counts["month"].to_list()
    idx = [m.year * 12 + m.month for m in months]
    missing = [i for i in range(idx[0], idx[-1] + 1) if i not in set(idx)]
    if missing:
        report.add(name, "month_gap", "warning", f"{len(missing)} month(s) missing inside the observed range")
    if name == "usage_monthly" and len(counts) >= 3:
        rows = counts["len"].to_numpy().astype(float)
        change = abs(rows[1:] / rows[:-1] - 1)
        for m, c in zip(months[1:], change, strict=True):
            if c > 0.25:
                report.add(
                    name, "volume_jump", "warning", f"row count changed {c:.0%} in {m} vs previous month"
                )


def _check_references(tables: dict[str, pl.DataFrame], report: Report) -> None:
    subs = tables.get("subscribers")
    if subs is None or "sub_id" not in subs.columns:
        return
    known = subs.select("sub_id").unique()
    for name, col in (
        ("usage_monthly", "sub_id"),
        ("recharges", "sub_id"),
        ("contacts", "sub_id"),
        ("contacts", "contact_id"),
        ("campaigns", "sub_id"),
    ):
        df = tables.get(name)
        if df is None or col not in df.columns:
            continue
        orphans = df.select(pl.col(col).alias("sub_id")).unique().join(known, on="sub_id", how="anti").height
        if orphans:
            share = orphans / max(df.select(col).n_unique(), 1)
            sev = "error" if share > TOLERANCE else "warning"
            report.add(
                name, "orphan_ids", sev, f"{col} values not found in subscribers ({share:.2%})", col, orphans
            )


def describe() -> str:
    """Markdown documentation of the contract (used to generate docs/data_contract.md)."""
    lines = []
    for t in CONTRACT.values():
        req = "required" if t.required else "optional"
        lines += [
            f"### `{t.name}` ({req})",
            "",
            t.description,
            "",
            f"Key: `{', '.join(t.key) or '—'}`",
            "",
            "| Column | Type | Required | Nullable | Rule | Notes |",
            "|:--|:--|:--:|:--:|:--|:--|",
        ]
        for c in t.columns:
            rule = []
            if c.min is not None or c.max is not None:
                rule.append(f"[{'' if c.min is None else c.min}, {'' if c.max is None else c.max}]")
            if c.allowed:
                rule.append(" / ".join(c.allowed))
            lines.append(
                f"| `{c.name}` | {c.dtype} | {'✓' if c.required else ''} | {'✓' if c.nullable else ''} "
                f"| {' '.join(rule)} | {c.description} |"
            )
        lines.append("")
    return "\n".join(lines)

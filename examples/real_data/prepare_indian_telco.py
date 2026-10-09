"""Adapter: the public Indian prepaid telecom churn dataset → this project's data contract.

Dataset: 99,999 prepaid subscribers of an Indian (South-East Asian market) operator, four months
of usage and recharge aggregates (June-September 2014, 226 columns), widely used as the
"Telecom Churn Case Study". It is not redistributed here. The script downloads it from a public
mirror (or reads a local copy) and writes contract tables as Parquet:

    python examples/real_data/prepare_indian_telco.py            # -> data/real/*.parquet
    churn --config examples/real_data/config.yaml validate
    churn --config examples/real_data/config.yaml train

Mapping choices (documented because a real operator's extract never matches a schema 1:1):

* usage_monthly: one row per subscriber-month with any usage (calls or data)
    voice_min = total outgoing + incoming minutes · onnet_min = on-net minutes
    data_mb = 2G + 3G volume · revenue = ARPU (negatives, i.e. adjustments, floored at 0)
    data_pack_cnt = monthly + sachet 2G/3G packs
  Not in the dataset, so left out (the contract marks them optional): active days, SMS,
  complaints, call-drop rate, data speed, per-competitor minutes, network/market table.
* recharges: the dataset has monthly aggregates, not transactions. The last recharge of each
  month is an exact event (its date and amount are given); the month's other top-ups become
  events on the 1st of the month with the average remaining amount. Counts, amounts and
  "days since last recharge" are therefore exact; within-month gap features are approximate.
* subscribers: activation month from age-on-network (`aon`, days) at the end of September;
  the single telecom circle becomes `division`; everything else is unknown (nullable).
* churn: no calls and no data in the following month (the dataset's standard definition,
  `label_events: [usage]`). Top-ups alone don't count as activity here: 81% of subscribers
  with zero September usage still recharged. 30-day window, because only one month follows the
  last feature month.
"""

from __future__ import annotations

import sys
import urllib.request
from datetime import date
from pathlib import Path

import polars as pl

URL = "https://raw.githubusercontent.com/avineet123/Telecom-Churn-Case-Study/master/telecom_churn_data.csv"
MONTHS = {6: date(2014, 6, 1), 7: date(2014, 7, 1), 8: date(2014, 8, 1), 9: date(2014, 9, 1)}
END_OF_DATA = date(2014, 9, 30)


def load_raw(path: Path) -> pl.DataFrame:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, path)
    return pl.read_csv(path, infer_schema_length=20_000)


def _num(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Float64).fill_null(0.0)


def convert(raw: pl.DataFrame) -> dict[str, pl.DataFrame]:
    raw = raw.with_columns(sub_id=pl.col("mobile_number").cast(pl.Utf8))

    subscribers = raw.select(
        "sub_id",
        activation_month=(pl.lit(END_OF_DATA) - pl.duration(days=pl.col("aon").cast(pl.Int64))).dt.truncate(
            "1mo"
        ),
        division=pl.format("circle_{}", pl.col("circle_id")),
        urban=pl.lit(None, dtype=pl.Boolean),
        gender=pl.lit(None, dtype=pl.Utf8),
        age=pl.lit(None, dtype=pl.Int32),
        plan_type=pl.lit("prepaid"),
        handset=pl.lit(None, dtype=pl.Utf8),
        dual_sim=pl.lit(None, dtype=pl.Boolean),
        app_user=pl.lit(None, dtype=pl.Boolean),
        mfs_user=pl.lit(None, dtype=pl.Boolean),
    )

    usage_parts, recharge_parts = [], []
    for m, start in MONTHS.items():
        u = raw.select(
            "sub_id",
            month=pl.lit(start),
            voice_min=_num(f"total_og_mou_{m}") + _num(f"total_ic_mou_{m}"),
            onnet_min=_num(f"onnet_mou_{m}"),
            data_mb=_num(f"vol_2g_mb_{m}") + _num(f"vol_3g_mb_{m}"),
            data_pack_cnt=(
                _num(f"monthly_2g_{m}")
                + _num(f"sachet_2g_{m}")
                + _num(f"monthly_3g_{m}")
                + _num(f"sachet_3g_{m}")
            ).cast(pl.Int32),
            revenue=_num(f"arpu_{m}").clip(lower_bound=0),
            _rech_n=_num(f"total_rech_num_{m}"),
            _rech_amt=_num(f"total_rech_amt_{m}"),
            _last_date=pl.col(f"date_of_last_rech_{m}").str.to_date("%m/%d/%Y", strict=False),
            _last_amt=_num(f"last_day_rch_amt_{m}"),
        )
        active = u.filter((pl.col("voice_min") > 0) | (pl.col("data_mb") > 0))
        # onnet can exceed total in the source (different counting of forwarded calls): cap it
        usage_parts.append(
            active.select(
                "sub_id",
                "month",
                "voice_min",
                pl.min_horizontal("onnet_min", "voice_min").alias("onnet_min"),
                "data_mb",
                "data_pack_cnt",
                "revenue",
            )
        )

        r = u.filter((pl.col("_rech_n") > 0) & (pl.col("_rech_amt") > 0))
        last = r.filter(pl.col("_last_date").is_not_null() & (pl.col("_last_amt") > 0)).select(
            "sub_id",
            ts=pl.col("_last_date").cast(pl.Datetime("ms")) + pl.duration(hours=12),
            amount=pl.col("_last_amt"),
            channel=pl.lit("last_of_month"),
        )
        rest = (
            r.with_columns(
                n_rest=(pl.col("_rech_n") - (pl.col("_last_amt") > 0).cast(pl.Float64))
                .clip(lower_bound=0)
                .cast(pl.Int64),
                amt_rest=(pl.col("_rech_amt") - pl.col("_last_amt")).clip(lower_bound=0),
            )
            .filter((pl.col("n_rest") > 0) & (pl.col("amt_rest") > 0))
            .with_columns(
                amount=pl.col("amt_rest") / pl.col("n_rest"), rep=pl.int_ranges(0, pl.col("n_rest"))
            )
            .explode("rep", empty_as_null=True)
            .select(
                "sub_id", ts=pl.lit(start).cast(pl.Datetime("ms")), amount="amount", channel=pl.lit("undated")
            )
        )
        recharge_parts += [last, rest]

    usage = pl.concat(usage_parts)
    recharges = pl.concat(recharge_parts).sort("ts")
    return {"subscribers": subscribers, "usage_monthly": usage, "recharges": recharges}


def main(out_dir: str = "data/real", raw_path: str = "data/real/telecom_churn_data.csv") -> None:
    raw = load_raw(Path(raw_path))
    tables = convert(raw)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.write_parquet(out / f"{name}.parquet")
        print(f"  {name:<14}{df.height:>12,} rows")
    aug = tables["usage_monthly"].filter(pl.col("month") == MONTHS[8]).select("sub_id")
    sep = tables["usage_monthly"].filter(pl.col("month") == MONTHS[9]).select("sub_id")
    print(
        f"Wrote contract tables to {out}; Aug-active subscribers with no Sept usage: "
        f"{aug.join(sep, on='sub_id', how='anti').height / aug.height:.1%}"
    )


if __name__ == "__main__":
    main(*sys.argv[1:])

"""Render the README figures from the reference run.

Run after `churn pipeline`:  python docs/assets/make_figures.py
Writes to docs/assets/: hero, stats and pipeline (light + dark SVG), story.png, reason_waterfall.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from churn.explain import FAMILY_LABELS, primary_driver
from churn.model import ChurnModel
from churn.style import DARK, LIGHT, SANS

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
THEMES = {"light": LIGHT, "dark": DARK}


def _canvas(w: float, h: float, t: dict):
    fig = plt.figure(figsize=(w, h), facecolor=t["surface"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def _save_svg(fig, name: str, t: dict) -> None:
    fig.savefig(OUT / name, facecolor=t["surface"])
    plt.close(fig)


# ----------------------------------------------------------------------------- hero
def hero(t: dict, name: str) -> None:
    W, H = 16, 5.4
    fig, ax = _canvas(W, H, t)
    ax.text(
        0.9, 4.62, "PREPAID CHURN PREDICTION", fontsize=13, color=t["accent"], family=SANS, weight="semibold"
    )
    ax.text(
        0.9,
        4.3,
        "Find the subscribers\ngoing silent, 60 days\nbefore they do.",
        fontsize=38,
        color=t["ink"],
        family=SANS,
        weight="semibold",
        linespacing=1.08,
        va="top",
    )
    ax.text(
        0.9,
        0.45,
        "An end-to-end churn system for prepaid mobile operators:\n"
        "behavioural labels, leakage-safe features, explained\n"
        "scores and profit-aware targeting.",
        fontsize=15,
        color=t["ink_2"],
        family=SANS,
        linespacing=1.45,
        va="bottom",
    )

    # one subscriber's weekly activity, winding down into silence
    rng = np.random.default_rng(3)
    weeks = 26
    act = np.clip(np.round(6.2 + rng.normal(0, 0.6, weeks)), 4, 7)
    act[13:21] = [6, 5, 5, 4, 3, 3, 2, 1]
    act[21:] = 0
    flag = 13
    x0, y0, bw, gap, scale = 8.7, 1.05, 0.2, 0.065, 0.36
    ax.add_patch(
        Rectangle(
            (x0 + 21 * (bw + gap) - gap / 2, y0 - 0.1),
            5 * (bw + gap),
            7 * scale + 0.25,
            fc=t["accent_soft"],
            ec="none",
            alpha=0.55,
        )
    )
    for i, a in enumerate(act):
        x = x0 + i * (bw + gap)
        if a == 0:
            ax.add_patch(plt.Circle((x + bw / 2, y0 + 0.05), 0.035, color=t["muted"]))
            continue
        color = t["accent"] if i >= flag else t["ink_2"]
        ax.add_patch(
            FancyBboxPatch(
                (x, y0),
                bw,
                a * scale,
                boxstyle="round,pad=0,rounding_size=0.05",
                fc=color,
                ec="none",
                alpha=0.95 if i >= flag else 0.35,
            )
        )
    fx = x0 + flag * (bw + gap) - gap / 2
    ax.plot([fx, fx], [y0 - 0.15, y0 + 7 * scale + 0.55], color=t["accent"], lw=1.4, ls=(0, (2, 2)))
    ax.text(
        fx + 0.12,
        y0 + 7 * scale + 0.62,
        "Model flags risk",
        fontsize=13.5,
        color=t["ink"],
        family=SANS,
        weight="semibold",
        ha="left",
        va="bottom",
    )
    ax.text(
        fx + 0.12,
        y0 + 7 * scale + 0.34,
        "p(churn) = 0.82",
        fontsize=12.5,
        color=t["ink_2"],
        family=SANS,
        ha="left",
        va="bottom",
    )
    sx = x0 + 21 * (bw + gap)
    ax.text(
        sx + 0.05,
        y0 + 7 * scale - 0.1,
        "Silent: no usage,\nno recharge",
        fontsize=12.5,
        color=t["ink_2"],
        family=SANS,
        va="top",
        linespacing=1.35,
    )
    ax.annotate(
        "",
        xy=(sx - 0.05, y0 - 0.42),
        xytext=(fx, y0 - 0.42),
        arrowprops=dict(arrowstyle="-|>", color=t["muted"], lw=1, mutation_scale=10),
    )
    ax.text(
        (fx + sx) / 2,
        y0 - 0.62,
        "≈ 2 months of warning",
        fontsize=12.5,
        color=t["ink_2"],
        family=SANS,
        ha="center",
        va="top",
    )
    ax.text(
        x0,
        y0 + 7 * scale + 0.62,
        "One subscriber, weekly active days",
        fontsize=12,
        color=t["muted"],
        family=SANS,
        va="bottom",
    )
    _save_svg(fig, name, t)


# ---------------------------------------------------------------------------- stats
def stats(t: dict, name: str, m: dict) -> None:
    lgb = m["metrics_test"]["LightGBM"]
    guard = m["uplift_test"]["profit_best"]["Risk × value + sleeping-dog guard"]["best_net_value"]
    risk = m["uplift_test"]["profit_best"]["Churn risk × value"]["best_net_value"]
    cards = [
        (f"{lgb['roc_auc']:.3f}", "ROC-AUC on a future month\nthe model never saw"),
        (f"{lgb['capture_top10']:.0%}", "of churners reached by\ncontacting only 10%"),
        (f"+{guard / risk - 1:.0%}", "campaign profit from the\nuplift sleeping-dog guard"),
        ("84 s", "monthly run for 1M subscribers,\nvalidated, scored, explained"),
    ]
    W, H = 16, 2.35
    fig, ax = _canvas(W, H, t)
    cw, gap = 3.55, 0.25
    x0 = (W - (4 * cw + 3 * gap)) / 2
    for i, (num, cap) in enumerate(cards):
        x = x0 + i * (cw + gap)
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.2),
                cw,
                1.95,
                boxstyle="round,pad=0,rounding_size=0.12",
                fc=t["card"],
                ec=t["edge"],
                lw=0.8,
            )
        )
        ax.text(
            x + 0.32,
            1.52,
            num,
            fontsize=38,
            color=t["accent"] if i == 0 else t["ink"],
            family=SANS,
            weight="semibold",
            va="center",
        )
        ax.text(x + 0.34, 0.7, cap, fontsize=14, color=t["ink_2"], family=SANS, va="center", linespacing=1.4)
    _save_svg(fig, name, t)


# ------------------------------------------------------------------------- pipeline
STAGES = [
    ("01", "Operator data", ["data contract + checks", "CSV · Parquet · SQL", "pseudonymised MSISDNs"]),
    ("02", "Snapshots", ["92 point-in-time features", "call graph · competitors", "60-day silence label"]),
    ("03", "Models", ["LightGBM churn risk", "X-learner uplift", "out-of-time validated"]),
    ("04", "Decide", ["SHAP root-cause driver", "value-based targeting", "sleeping-dog guard"]),
    ("05", "Operate", ["registry · gate · rollback", "API · dashboard", "holdouts · backtests"]),
]


def pipeline(t: dict, name: str) -> None:
    W, H = 16, 3.5
    fig, ax = _canvas(W, H, t)
    w, h, gap, y0 = 2.72, 2.8, 0.42, 0.35
    x0 = (W - (5 * w + 4 * gap)) / 2
    for i, (num, title, lines) in enumerate(STAGES):
        x = x0 + i * (w + gap)
        hi = i == 2
        ax.add_patch(
            FancyBboxPatch(
                (x, y0),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=0.12",
                fc=t["card"],
                ec=t["accent"] if hi else t["edge"],
                lw=1.5 if hi else 0.8,
            )
        )
        ax.text(
            x + 0.26,
            y0 + h - 0.36,
            num,
            fontsize=13,
            color=t["accent"],
            family=SANS,
            weight="semibold",
            va="center",
        )
        ax.text(
            x + 0.26,
            y0 + h - 0.8,
            title,
            fontsize=20,
            color=t["ink"],
            family=SANS,
            weight="semibold",
            va="center",
        )
        for j, line in enumerate(lines):
            ax.text(
                x + 0.26,
                y0 + h - 1.33 - j * 0.36,
                line,
                fontsize=14,
                color=t["ink_2"],
                family=SANS,
                va="center",
            )
        if i < 4:
            ax.add_patch(
                FancyArrowPatch(
                    (x + w + 0.07, y0 + h / 2),
                    (x + w + gap - 0.07, y0 + h / 2),
                    arrowstyle="-|>",
                    mutation_scale=11,
                    color=t["muted"],
                    lw=1.1,
                )
            )
    _save_svg(fig, name, t)


# ------------------------------------------------------------------------ story figure
def _pick_examples(usage: pl.DataFrame) -> dict[str, np.ndarray]:
    months = sorted(usage["month"].unique().to_list())
    wide = (
        usage.select("sub_id", "month", "days_active")
        .pivot(on="month", index="sub_id", values="days_active")
        .select(["sub_id", *[str(m) for m in months]])
    )
    arr = wide.drop("sub_id").to_numpy().astype(float)  # NaN = no activity that month
    present = ~np.isnan(arr)
    n_active = present.sum(axis=1)
    first_gap = np.where(present.all(axis=1), 12, np.argmin(present, axis=1))
    contiguous_from_start = present[:, 0] & (n_active == first_gap)
    base = np.nanmedian(arr[:, :5], axis=1)

    def first(mask):
        idx = np.flatnonzero(mask)
        return arr[idx[0]] if idx.size else None

    k = 9
    winddown = first(
        contiguous_from_start
        & (first_gap == k)
        & (base >= 26)
        & (arr[:, k - 1] <= 0.35 * base)
        & (arr[:, k - 2] <= 0.75 * base)
        & (arr[:, k - 3] >= 0.9 * base)
    )
    steady = np.nanmin(np.where(present, arr, np.inf), axis=1) >= 0.85 * base
    abrupt = first(contiguous_from_start & (first_gap == 8) & (base >= 26) & steady)
    dip = np.nanmin(arr[:, 3:9], axis=1) if arr.shape[1] > 9 else None
    dormant = first(
        present.all(axis=1)
        & (base >= 26)
        & (dip <= 0.25 * base)
        & (np.sort(arr, axis=1)[:, 1] >= 0.85 * base)
    )
    return {"winddown": winddown, "dormant": dormant, "abrupt": abrupt}


def story(t: dict, path: Path, usage: pl.DataFrame) -> None:
    ex = _pick_examples(usage)
    panels = [
        ("Winds down, then goes silent", "Churn — the pattern the model learns", ex["winddown"]),
        ("Goes quiet for a month, returns", "Not churn — a village trip or the other SIM", ex["dormant"]),
        ("Leaves without warning", "Churn — no signal in the data", ex["abrupt"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9), facecolor=t["surface"], sharey=True)
    fig.subplots_adjust(wspace=0.12)
    labels = list("JFMAMJJASOND")
    for ax, (title, sub, series) in zip(axes, panels, strict=True):
        ax.set_facecolor(t["surface"])
        med = np.nanmedian(series)
        for i, v in enumerate(series):
            if np.isnan(v):
                ax.plot(i, 0.6, "o", ms=4, color=t["muted"])
                continue
            low = v < 0.8 * med
            ax.bar(i, v, width=0.72, color=t["accent"] if low else t["ink_2"], alpha=1 if low else 0.3)
        ax.annotate(
            title,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(0, 32),
            textcoords="offset points",
            fontsize=15,
            family=SANS,
            weight="semibold",
            color=t["ink"],
            va="bottom",
        )
        ax.annotate(
            sub,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(0, 13),
            textcoords="offset points",
            fontsize=10.5,
            family=SANS,
            color=t["ink_2"],
            va="bottom",
        )
        ax.set_xticks(range(12), labels)
        ax.set_ylim(0, 33)
        ax.tick_params(colors=t["muted"], length=0, labelsize=9.5)
        ax.grid(axis="y", color=t["grid"], lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(t["grid"])
    axes[0].set_ylabel("Active days per month", color=t["ink_2"], fontsize=10, labelpad=8)
    fig.savefig(path, dpi=150, facecolor=t["surface"], bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


# -------------------------------------------------------------------- reason waterfall
def waterfall(t: dict, path: Path, model: ChurnModel, test: pl.DataFrame) -> None:
    X = model.matrix(test)
    p = model.predict(X)
    contrib_full = model.booster.predict(X, pred_contrib=True)
    fam = contrib_full[:, :-1] @ model._fam
    cause_cols = [
        i
        for i, f in enumerate(model.families)
        if f in ("network_experience", "early_life", "competitor_pressure", "credit_stress")
    ]
    y = test["churn"].to_numpy()
    cause = fam[:, cause_cols].max(axis=1)
    cand = np.flatnonzero((y == 1) & (p > 0.6) & (p < 0.95))
    i = int(cand[np.argmax(cause[cand])]) if cand.size else int(np.argmax(p))
    base = contrib_full[i, -1]
    items = sorted(zip(model.families, fam[i], strict=True), key=lambda kv: -abs(kv[1]))
    items = [kv for kv in items if abs(kv[1]) >= 0.02]

    def prob(z):
        return 1 / (1 + np.exp(-z))

    fig, ax = plt.subplots(figsize=(9.5, 0.52 * len(items) + 2.0), facecolor=t["surface"])
    ax.set_facecolor(t["surface"])
    ends = base + np.cumsum([0] + [v for _, v in items])
    x_col = ends.max() + 0.75
    run = base
    for row, (_f, v) in enumerate(items):
        ax.barh(row, v, left=run, height=0.58, color=t["accent"] if v > 0 else t["violet"])
        ax.plot([max(run, run + v), x_col - 0.42], [row, row], color=t["grid"], lw=0.8, zorder=0)
        ax.text(x_col, row, f"{v:+.2f}", va="center", ha="right", fontsize=10, color=t["ink_2"])
        run += v
    ax.set_xlim(base - 0.25, x_col + 0.05)
    ax.set_yticks(range(len(items)), [FAMILY_LABELS[f].split(":")[0] for f, _ in items])
    ax.invert_yaxis()
    ax.axvline(base, color=t["muted"], lw=1, ls=(0, (3, 3)))
    ax.axvline(run, color=t["ink"], lw=1)
    top = -0.9
    ax.text(
        base, top, f"Baseline\n{prob(base):.1%}", ha="center", va="bottom", fontsize=9.5, color=t["ink_2"]
    )
    ax.text(
        run,
        top,
        f"This subscriber\n{prob(run):.0%}",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=t["ink"],
        weight="semibold",
    )
    ax.annotate(
        "Why one subscriber scored high",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 58),
        textcoords="offset points",
        fontsize=15,
        family=SANS,
        weight="semibold",
        color=t["ink"],
        va="bottom",
    )
    driver = primary_driver(fam[i : i + 1], model.families)[0]
    ax.annotate(
        f"TreeSHAP contributions by driver family · root cause: {driver.replace('_', ' ')} · churned in the test window",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 40),
        textcoords="offset points",
        fontsize=10.5,
        family=SANS,
        color=t["ink_2"],
        va="bottom",
    )
    ax.tick_params(axis="y", colors=t["ink"], labelsize=10.5, length=0)
    ax.tick_params(axis="x", colors=t["muted"], labelsize=9, length=0)
    ax.grid(axis="x", color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.set_xlabel("Log-odds of churn", color=t["ink_2"], fontsize=10, labelpad=8)
    fig.savefig(path, dpi=150, facecolor=t["surface"], bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def main() -> None:
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
    for mode, t in THEMES.items():
        hero(t, f"hero-{mode}.svg")
        stats(t, f"stats-{mode}.svg", metrics)
        pipeline(t, f"pipeline-{mode}.svg")
    usage = pl.read_parquet(ROOT / "data" / "raw" / "usage_monthly.parquet")
    story(LIGHT, OUT / "story.png", usage)
    model = ChurnModel.load(ROOT / "artifacts" / "model")
    test = pl.read_parquet(ROOT / "data" / "features" / "test.parquet")
    waterfall(LIGHT, OUT / "reason_waterfall.png", model, test)
    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()

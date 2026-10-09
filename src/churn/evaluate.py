"""Metrics that retention teams actually use, plus report charts.

Accuracy is meaningless at ~4% churn (predicting "nobody churns" scores 96%). What matters:
ranking quality (ROC-AUC, KS), precision on the rare class (PR-AUC), and how many churners a
campaign of fixed size reaches (lift and capture rate in the top deciles).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from churn.style import LIGHT, SANS, SERIES  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID = (
    LIGHT["surface"],
    LIGHT["ink"],
    LIGHT["ink_2"],
    LIGHT["muted"],
    LIGHT["grid"],
)


def top_share_stats(y: np.ndarray, p: np.ndarray, share: float) -> tuple[float, float, float]:
    """(precision, capture rate, lift) when targeting the top `share` of the base."""
    k = max(1, int(round(share * len(y))))
    top = y[np.argsort(-p, kind="stable")[:k]]
    precision = top.mean()
    return precision, top.sum() / max(y.sum(), 1), precision / y.mean()


def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    fpr, tpr, _ = roc_curve(y, p)
    out = {
        "n": int(len(y)),
        "churn_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "ks": float(np.max(tpr - fpr)),
    }
    if p.min() >= 0 and p.max() <= 1:
        out["brier"] = float(brier_score_loss(y, p))
        out["log_loss"] = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
    for share in (0.05, 0.10, 0.20):
        prec, cap, lift = top_share_stats(y, p, share)
        pct = int(share * 100)
        out[f"precision_top{pct}"] = float(prec)
        out[f"capture_top{pct}"] = float(cap)
        out[f"lift_top{pct}"] = float(lift)
    return out


def gains_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    df = (
        pd.DataFrame({"y": y, "p": p}).sort_values("p", ascending=False, kind="stable").reset_index(drop=True)
    )
    df["decile"] = np.arange(len(df)) * bins // len(df) + 1
    g = df.groupby("decile").agg(subscribers=("y", "size"), churners=("y", "sum"), avg_score=("p", "mean"))
    g["churn_rate"] = g["churners"] / g["subscribers"]
    g["lift"] = g["churn_rate"] / df["y"].mean()
    g["cum_capture"] = g["churners"].cumsum() / df["y"].sum()
    return g.reset_index()


# ---------------------------------------------------------------------------- charts
def _style(ax, title: str, xlabel: str, ylabel: str, subtitle: str | None = None) -> None:
    """Chart frame: bold title, subtitle stating the takeaway, recessive axes."""
    ax.set_facecolor(SURFACE)
    ax.annotate(
        title,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 34 if subtitle else 14),
        textcoords="offset points",
        fontsize=15,
        color=INK,
        family=SANS,
        weight="semibold",
        va="bottom",
    )
    if subtitle:
        ax.annotate(
            subtitle,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(0, 14),
            textcoords="offset points",
            fontsize=10.5,
            color=INK_2,
            family=SANS,
            va="bottom",
        )
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10, labelpad=8)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10, labelpad=8)
    ax.tick_params(colors=MUTED, labelsize=9, length=0, pad=6)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def _legend(ax, **kw) -> None:
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK_2, handlelength=1.6, **kw)


def _figure(ncols: int = 1, width: float = 6.4, height: float = 4.2):
    fig, axes = plt.subplots(1, ncols, figsize=(width * ncols, height), facecolor=SURFACE)
    return fig, np.atleast_1d(axes)


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def plot_roc_pr(y: np.ndarray, scores: dict[str, np.ndarray], path: Path) -> None:
    fig, (a1, a2) = _figure(2)
    fig.subplots_adjust(wspace=0.28)
    for (name, p), color in zip(scores.items(), SERIES, strict=False):
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(fpr, tpr, color=color, lw=2, label=f"{name}  ·  {roc_auc_score(y, p):.3f}")
        prec, rec, _ = precision_recall_curve(y, p)
        a2.plot(rec, prec, color=color, lw=2, label=f"{name}  ·  {average_precision_score(y, p):.3f}")
    a1.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (3, 3)))
    a2.axhline(y.mean(), color=MUTED, lw=1, ls=(0, (3, 3)))
    _style(
        a1,
        "ROC curve",
        "False positive rate",
        "True positive rate",
        "Out-of-time test month · legend shows ROC-AUC",
    )
    _style(
        a2,
        "Precision–recall",
        "Recall (share of churners caught)",
        "Precision",
        "Legend shows average precision",
    )
    _legend(a1, loc="lower right")
    _legend(a2, loc="upper right")
    _save(fig, path)


def plot_gains(y: np.ndarray, scores: dict[str, np.ndarray], path: Path) -> None:
    fig, (ax,) = _figure()
    for (name, p), color in zip(scores.items(), SERIES, strict=False):
        cum = np.concatenate([[0], np.cumsum(y[np.argsort(-p, kind="stable")])]) / y.sum()
        idx = np.linspace(0, len(y), 1000).astype(int)
        ax.plot(idx / len(y) * 100, cum[idx] * 100, color=color, lw=2, label=name)
    ax.plot([0, 100], [0, 100], color=MUTED, lw=1, ls=(0, (3, 3)), label="Random")
    first = next(iter(scores.values()))
    cap10 = top_share_stats(y, first, 0.10)[1] * 100
    ax.scatter([10], [cap10], s=46, color=SERIES[0], edgecolor=SURFACE, lw=2, zorder=5)
    ax.annotate(
        f"{cap10:.0f}% of churners\nin the top 10%",
        (10, cap10),
        xytext=(21, 15),
        va="top",
        fontsize=9.5,
        color=INK,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8),
    )
    _style(
        ax,
        "Cumulative gains",
        "% of base contacted, highest risk first",
        "% of churners reached",
        "How many churners a campaign of a given size reaches",
    )
    _legend(ax, loc="lower right")
    _save(fig, path)


def plot_calibration(y: np.ndarray, p: np.ndarray, path: Path, bins: int = 20) -> None:
    fig, (ax,) = _figure(width=5.4)
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"], bins, labels=False, duplicates="drop")
    g = df.groupby("bin").agg(pred=("p", "mean"), obs=("y", "mean"))
    lim = max(g["pred"].max(), g["obs"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls=(0, (3, 3)), label="Perfect calibration")
    ax.plot(
        g["pred"], g["obs"], color=SERIES[0], lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, label="LightGBM"
    )
    _style(
        ax,
        "Calibration",
        "Predicted churn probability",
        "Observed churn rate",
        "Predicted risk matches observed churn (20 bins)",
    )
    _legend(ax, loc="upper left")
    _save(fig, path)


def plot_driver_importance(importance: dict[str, float], labels: dict[str, str], path: Path) -> None:
    # families the data cannot express (no such inputs) are omitted
    items = [(k, v) for k, v in importance.items() if v >= 0.005][::-1]
    fig, (ax,) = _figure(width=7.6, height=4.2)
    names = [labels.get(k, k).split(":")[0] for k, _ in items]
    ax.barh(names, [v for _, v in items], color=SERIES[0], height=0.62)
    for i, (_, v) in enumerate(items):
        ax.text(v, i, f"  {v:.2f}", va="center", fontsize=9.5, color=INK_2)
    _style(
        ax,
        "What drives churn risk",
        "Mean |SHAP| contribution (log-odds)",
        "",
        "Per-subscriber TreeSHAP values summed into business driver families",
    )
    ax.tick_params(axis="y", colors=INK, labelsize=10)
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def plot_profit(
    curves: dict[str, np.ndarray], policy_k: int, n: int, currency: str, path: Path, max_share: float = 0.15
) -> None:
    fig, (ax,) = _figure(width=7.0)
    k_max = int(max_share * n)
    for (name, c), color in zip(curves.items(), SERIES, strict=False):
        ax.plot(np.arange(k_max + 1) / n * 100, c[: k_max + 1] / 1e3, color=color, lw=2, label=name)
    model_curve = next(iter(curves.values()))
    px, py = policy_k / n * 100, model_curve[policy_k] / 1e3
    ax.scatter([px], [py], s=56, color=SERIES[0], edgecolor=SURFACE, lw=2, zorder=5)
    ax.annotate(
        f"Contact {policy_k:,} subscribers ({policy_k / n:.1%})\nnet +{py:,.0f}k {currency}",
        (px, py),
        xytext=(0.55, 0.93),
        textcoords="axes fraction",
        va="top",
        fontsize=9.5,
        color=INK,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8),
    )
    ax.axhline(0, color=INK_2, lw=1)
    ax.set_xlim(0, max_share * 100)
    _style(
        ax,
        "Campaign profit",
        "% of base contacted",
        f"Net profit (thousand {currency})",
        "Expected-value targeting vs the recency rule and random selection",
    )
    _legend(ax, loc="lower left")
    _save(fig, path)


def plot_uplift(
    qini: dict[str, tuple[np.ndarray, np.ndarray]],
    profit: dict[str, tuple[np.ndarray, np.ndarray]],
    currency: str,
    path: Path,
) -> None:
    fig, (a1, a2) = _figure(2)
    fig.subplots_adjust(wspace=0.3)
    for (name, (x, q)), color in zip(qini.items(), SERIES + [MUTED], strict=False):
        dashed = name.startswith("Random")
        a1.plot(
            x * 100,
            q,
            color=MUTED if dashed else color,
            lw=1 if dashed else 2,
            ls=(0, (3, 3)) if dashed else "-",
            label=name,
        )
    for (name, (x, v)), color in zip(profit.items(), SERIES + [MUTED], strict=False):
        dashed = name.startswith("Random")
        a2.plot(
            x * 100,
            v / 1e3,
            color=MUTED if dashed else color,
            lw=1 if dashed else 2,
            ls=(0, (3, 3)) if dashed else "-",
            label=name,
        )
    a2.axhline(0, color=INK_2, lw=1)
    _style(
        a1,
        "Qini curve",
        "% of campaign population treated, best first",
        "Churners prevented",
        "Randomised campaign, out-of-time · higher is better",
    )
    _style(
        a2,
        "Campaign value by targeting rule",
        "% of campaign population treated",
        f"Net value (thousand {currency})",
        "Realised incremental value minus contact and offer cost",
    )
    visible = [v[x <= 0.2] / 1e3 for x, v in profit.values()]
    lo, hi = min(v.min() for v in visible), max(v.max() for v in visible)
    a2.set_xlim(0, 20)
    a2.set_ylim(lo - 0.08 * (hi - lo), hi + 0.25 * (hi - lo))
    _legend(a1, loc="lower right")
    _legend(a2, loc="lower left")
    _save(fig, path)

"""Drift monitoring with the Population Stability Index (PSI), the standard check in telco and banking.

At training time we freeze a reference (validation-month) distribution for every feature and for
the score. Each scoring run is compared against it:

    PSI < 0.10  stable  |  0.10-0.25  moderate shift, investigate  |  > 0.25  major shift, retrain
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-4
# division-level network / market context changes month to month by design and has only 8 distinct
# values per snapshot; it is tracked on market dashboards, not via PSI.
CONTEXT_PREFIXES = ("div_", "competitor_")


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(expected, EPS, None)
    a = np.clip(actual, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _numeric_bins(values: np.ndarray, bins: int) -> list[float]:
    v = values[~np.isnan(values)]
    if v.size == 0:
        return []
    return np.unique(np.quantile(v, np.linspace(0, 1, bins + 1)[1:-1])).tolist()


def _numeric_dist(values: np.ndarray, edges: list[float]) -> list[float]:
    """Distribution over [NaN] + len(edges)+1 quantile buckets."""
    nan = np.isnan(values)
    idx = np.searchsorted(np.asarray(edges), values[~nan], side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1)
    return (np.concatenate([[nan.sum()], counts]) / max(len(values), 1)).tolist()


def build_reference(X: pd.DataFrame, scores: np.ndarray, bins: int = 10) -> dict:
    ref: dict = {"features": {}}
    for col in X.columns:
        if col.startswith(CONTEXT_PREFIXES):
            continue
        if isinstance(X[col].dtype, pd.CategoricalDtype):
            dist = X[col].astype(object).fillna("__missing__").value_counts(normalize=True)
            ref["features"][col] = {"type": "categorical", "dist": dist.to_dict()}
        else:
            vals = X[col].to_numpy(dtype=float)
            edges = _numeric_bins(vals, bins)
            ref["features"][col] = {"type": "numeric", "edges": edges, "dist": _numeric_dist(vals, edges)}
    edges = _numeric_bins(scores, bins)
    ref["score"] = {"type": "numeric", "edges": edges, "dist": _numeric_dist(scores, edges)}
    return ref


def _column_psi(spec: dict, values: pd.Series | np.ndarray) -> float:
    if spec["type"] == "categorical":
        cur = pd.Series(values).astype(object).fillna("__missing__").value_counts(normalize=True)
        keys = set(spec["dist"]) | set(cur.index)
        return _psi(
            np.array([spec["dist"].get(k, 0.0) for k in keys]), np.array([cur.get(k, 0.0) for k in keys])
        )
    return _psi(
        np.array(spec["dist"]), np.array(_numeric_dist(np.asarray(values, dtype=float), spec["edges"]))
    )


def drift_report(reference: dict, X: pd.DataFrame, scores: np.ndarray) -> dict:
    rows = [
        {"feature": col, "psi": _column_psi(spec, X[col])}
        for col, spec in reference["features"].items()
        if col in X.columns
    ]
    rows.sort(key=lambda r: -r["psi"])
    for r in rows:
        r["status"] = "major" if r["psi"] > 0.25 else "moderate" if r["psi"] > 0.10 else "stable"
    score_psi = _column_psi(reference["score"], scores)
    return {
        "score_psi": score_psi,
        "score_status": "major" if score_psi > 0.25 else "moderate" if score_psi > 0.10 else "stable",
        "n_major": sum(r["status"] == "major" for r in rows),
        "n_moderate": sum(r["status"] == "moderate" for r in rows),
        "features": rows,
    }


def save_reference(ref: dict, model_dir: str | Path) -> None:
    Path(model_dir, "drift_reference.json").write_text(json.dumps(ref))


def load_reference(model_dir: str | Path) -> dict:
    return json.loads(Path(model_dir, "drift_reference.json").read_text())

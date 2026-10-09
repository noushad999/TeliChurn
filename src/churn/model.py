"""Model artifact: LightGBM booster + the metadata needed to score consistently in production.

Everything that inference depends on is frozen at training time: feature order, categorical
levels and risk-band thresholds. Nothing is re-fitted on inference data, and unknown
categories or missing features become NaN, which LightGBM handles natively.

Inference runs on a dense float matrix (categoricals encoded as their training level index,
which is exactly what LightGBM does internally for pandas categoricals). That skips pandas
entirely: ~0.04 ms per prediction instead of ~9 ms.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from churn.explain import FAMILY_ACTIONS, FAMILY_LABELS, family_matrix, primary_driver


class ChurnModel:
    def __init__(self, booster: lgb.Booster, metadata: dict[str, Any]):
        self.booster = booster
        self.metadata = metadata
        self.features: list[str] = metadata["features"]
        self.categories: dict[str, list[str]] = metadata["categories"]
        self.families, self._fam = family_matrix(self.features)
        self._codes = {c: {lvl: float(i) for i, lvl in enumerate(lv)} for c, lv in self.categories.items()}

    # ------------------------------------------------------------------ persistence
    def save(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(model_dir / "model.txt"))
        (model_dir / "metadata.json").write_text(json.dumps(self.metadata, indent=2, default=str))

    @classmethod
    def load(cls, model_dir: str | Path) -> ChurnModel:
        model_dir = Path(model_dir)
        booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
        metadata = json.loads((model_dir / "metadata.json").read_text())
        return cls(booster, metadata)

    # ------------------------------------------------------------------ inference
    def matrix(self, data: pl.DataFrame | list[dict[str, Any]]) -> np.ndarray:
        """Dense float matrix in training feature order: the fast path for batch and API scoring."""
        if isinstance(data, pl.DataFrame):
            cols = []
            for f in self.features:
                if f not in data.columns:
                    cols.append(pl.lit(None, dtype=pl.Float64).alias(f))
                elif f in self._codes:
                    cols.append(
                        pl.col(f).replace_strict(self._codes[f], default=None, return_dtype=pl.Float64)
                    )
                else:
                    cols.append(pl.col(f).cast(pl.Float64, strict=False))
            return data.select(cols).to_numpy().astype(np.float64, copy=False)
        X = np.full((len(data), len(self.features)), np.nan)
        for j, f in enumerate(self.features):
            codes = self._codes.get(f)
            for i, rec in enumerate(data):
                v = rec.get(f)
                if v is None:
                    continue
                if codes is not None:
                    X[i, j] = codes.get(v, np.nan)
                else:
                    try:
                        X[i, j] = float(v)
                    except (TypeError, ValueError):
                        pass
        return X

    def frame(self, data: pl.DataFrame | pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
        """Pandas view with categorical dtypes (used for drift monitoring and training-side analysis)."""
        if isinstance(data, pl.DataFrame):
            present = [c for c in self.features if c in data.columns]
            df = data.select(present).to_pandas()
        elif isinstance(data, list):
            df = pd.DataFrame.from_records(data)
        else:
            df = data
        df = df.reindex(columns=self.features)
        for col in self.features:
            if col in self.categories:
                levels = self.categories[col]
                df[col] = pd.Categorical(df[col].where(df[col].isin(levels)), categories=levels)
            elif not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def predict(self, X: np.ndarray | pd.DataFrame, num_threads: int = 0) -> np.ndarray:
        return self.booster.predict(X, num_threads=num_threads)

    def contributions(self, X: np.ndarray | pd.DataFrame, num_threads: int = 0) -> np.ndarray:
        """Per-feature SHAP values in log-odds space (LightGBM native TreeSHAP), bias column dropped."""
        return self.booster.predict(X, pred_contrib=True, num_threads=num_threads)[:, :-1]

    def risk_band(self, p: np.ndarray) -> np.ndarray:
        th = self.metadata["risk_thresholds"]
        return np.select(
            [p >= th["critical"], p >= th["high"], p >= th["medium"]], ["critical", "high", "medium"], "low"
        )

    def reasons(self, contrib: np.ndarray, top_k: int = 2) -> tuple[np.ndarray, np.ndarray]:
        """(top-k risk-raising families, primary root-cause driver) per subscriber."""
        fam_scores = contrib @ self._fam
        order = np.argsort(-fam_scores, axis=1)[:, :top_k]
        top_scores = np.take_along_axis(fam_scores, order, axis=1)
        names = np.array(self.families, dtype=object)[order]
        names[top_scores <= 0] = None
        return names, primary_driver(fam_scores, self.families)

    def score_frame(
        self,
        X: np.ndarray | pd.DataFrame,
        explain: np.ndarray | None = None,
        p: np.ndarray | None = None,
        top_k: int = 2,
    ) -> dict[str, np.ndarray]:
        """Scores for every row; SHAP reason codes + action for rows where `explain` is True (default: all).

        TreeSHAP costs ~1000x a plain prediction, so batch scoring explains only the subscribers who will
        actually be actioned (high risk / campaign targets).
        """
        p = self.predict(X) if p is None else p
        n = len(p)
        mask = np.ones(n, bool) if explain is None else np.asarray(explain, bool)
        names = np.full((n, top_k), None, dtype=object)
        driver = np.full(n, None, dtype=object)
        if mask.any():
            names[mask], driver[mask] = self.reasons(self.contributions(X[mask]), top_k=top_k)
        actions = np.array([FAMILY_ACTIONS[d] if d else None for d in driver], dtype=object)
        return {
            "churn_probability": p,
            "risk_band": self.risk_band(p),
            **{f"reason_{i + 1}": names[:, i] for i in range(top_k)},
            "primary_driver": driver,
            "recommended_action": actions,
        }

    @staticmethod
    def describe_reason(code: str | None) -> str | None:
        return FAMILY_LABELS.get(code) if code else None

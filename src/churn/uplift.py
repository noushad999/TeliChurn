"""Uplift modelling: who should receive a retention offer, not just who is at risk.

A churn score ranks subscribers by risk, but a campaign only earns money on subscribers whose
behaviour the offer *changes*. Randomised test-and-learn campaigns (treatment vs holdout) give
the data to estimate that directly:

    uplift(x) = P(churn | x, no offer) - P(churn | x, offer)

We fit a seed-ensembled X-learner (Künzel et al., 2019) with LightGBM base learners on earlier campaigns
and evaluate out-of-time on a later campaign with the Qini curve, against a T-learner and against
plain churn-risk ranking. The Qini curve is the incremental churners prevented when
treating the top-k subscribers by score, estimated from the randomised holdout. Four kinds of
subscriber show up in the data:

* persuadables:  stay only if treated (positive uplift)  -> the whole point of the campaign
* sure things:   stay either way                          -> offer is wasted money
* lost causes:   leave either way (often the highest risk) -> offer is wasted money
* sleeping dogs: leave *because* they were contacted      -> offer destroys value
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from churn.config import parse_month

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 5,
    "min_child_samples": 400,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "verbose": -1,
}


def campaign_frame(snapshot: pl.DataFrame, campaigns: pl.DataFrame, month: str) -> pl.DataFrame:
    """Snapshot rows for subscribers enrolled in the campaign sent after `month`, with their arm."""
    enrolled = campaigns.filter(pl.col("campaign_month") == parse_month(month)).select("sub_id", "treatment")
    return snapshot.join(enrolled, on="sub_id", how="inner")


class UpliftModel:
    """Seed-ensembled two-arm meta-learner over LightGBM; `learner` is "x" (X-learner) or "t" (T-learner).

    X-learner: fit outcome models per arm, impute individual effects on each arm with the other
    arm's model, regress those effects on features, and blend with the propensity (0.5 in a
    randomised 50/50 campaign). It borrows strength across arms, which matters when effects are
    small and noisy, as retention effects are. Because single fits are still noisy, `n_models`
    learners with different seeds are averaged, so the targeting decision does not depend on luck.
    """

    NAMES = ("mu_treated", "mu_control", "tau_treated", "tau_control")

    def __init__(self, members: list[dict[str, lgb.Booster]], metadata: dict):
        self.members, self.metadata = members, metadata

    @staticmethod
    def _fit_one(X, t, y, learner: str, rounds: int, seed: int) -> dict[str, lgb.Booster]:
        params = PARAMS | {"seed": seed}
        tr, co = t == 1, t == 0
        b = {
            "mu_treated": lgb.train(params, lgb.Dataset(X[tr], y[tr]), num_boost_round=rounds),
            "mu_control": lgb.train(params, lgb.Dataset(X[co], y[co]), num_boost_round=rounds),
        }
        if learner == "x":
            reg = params | {"objective": "regression"}
            d_treated = b["mu_control"].predict(X[tr]) - y[tr]  # churn avoided by treating
            d_control = y[co] - b["mu_treated"].predict(X[co])
            b["tau_treated"] = lgb.train(reg, lgb.Dataset(X[tr], d_treated), num_boost_round=rounds)
            b["tau_control"] = lgb.train(reg, lgb.Dataset(X[co], d_control), num_boost_round=rounds)
        return b

    @classmethod
    def fit(
        cls,
        X: pd.DataFrame,
        t: np.ndarray,
        y: np.ndarray,
        learner: str = "x",
        rounds: int = 200,
        seed: int = 42,
        n_models: int = 5,
    ) -> UpliftModel:
        members = [cls._fit_one(X, t, y, learner, rounds, seed + k) for k in range(n_models)]
        meta = {
            "learner": {"x": "X-learner", "t": "T-learner"}[learner] + " (LightGBM)",
            "rounds": rounds,
            "n_models": n_models,
        }
        return cls(members, meta)

    @staticmethod
    def _predict_one(b: dict[str, lgb.Booster], X) -> np.ndarray:
        if "tau_treated" in b:
            propensity = 0.5
            return propensity * b["tau_control"].predict(X) + (1 - propensity) * b["tau_treated"].predict(X)
        return b["mu_control"].predict(X) - b["mu_treated"].predict(X)

    def predict(self, X) -> np.ndarray:
        """Estimated reduction in churn probability if the subscriber receives the offer."""
        return np.mean([self._predict_one(b, X) for b in self.members], axis=0)

    def save(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        for k, b in enumerate(self.members):
            for name, booster in b.items():
                booster.save_model(str(model_dir / f"uplift_{k}_{name}.txt"))
        (model_dir / "uplift_metadata.json").write_text(json.dumps(self.metadata, indent=2))

    @classmethod
    def load(cls, model_dir: str | Path) -> UpliftModel | None:
        model_dir = Path(model_dir)
        if not (model_dir / "uplift_metadata.json").exists():
            return None
        metadata = json.loads((model_dir / "uplift_metadata.json").read_text())
        members = [
            {
                n: lgb.Booster(model_file=str(model_dir / f"uplift_{k}_{n}.txt"))
                for n in cls.NAMES
                if (model_dir / f"uplift_{k}_{n}.txt").exists()
            }
            for k in range(metadata.get("n_models", 1))
        ]
        return cls(members, metadata)


def qini_curve(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Incremental churners prevented when treating the top-k by `score` (randomised data).

    Q(k) = C_control(k) * N_treated(k) / N_control(k) - C_treated(k), where C are churners among the
    top-k in each arm. Returns (share of population, Q) sampled at 200 points.
    """
    order = np.argsort(-score, kind="stable")
    y, t = y[order], t[order]
    n_t, n_c = np.cumsum(t), np.cumsum(1 - t)
    c_t, c_c = np.cumsum(y * t), np.cumsum(y * (1 - t))
    q = c_c * np.divide(n_t, n_c, out=np.zeros(len(y)), where=n_c > 0) - c_t
    idx = np.linspace(0, len(y) - 1, 200).astype(int)
    return np.concatenate([[0.0], (idx + 1) / len(y)]), np.concatenate([[0.0], q[idx]])


def qini_coefficient(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> float:
    """Area between the Qini curve and the random-targeting line, normalised by population size."""
    x, q = qini_curve(y, t, score)
    random_line = x * q[-1]
    gap = q - random_line
    area = np.sum((gap[1:] + gap[:-1]) / 2 * np.diff(x))
    return float(area / max(len(y), 1) * 1000)


def value_curve(
    y: np.ndarray, t: np.ndarray, score: np.ndarray, value: np.ndarray, cost: float
) -> tuple[np.ndarray, np.ndarray]:
    """Realised net campaign value when treating the top-k by `score`, estimated from randomised data.

    Incremental value(k) = (value lost among holdout churners, rescaled to the treated count) minus
    (value lost among treated churners), then minus the cost of contacting the treated share of k.
    """
    order = np.argsort(-score, kind="stable")
    y, t, value = y[order], t[order], value[order]
    n_t, n_c = np.cumsum(t), np.cumsum(1 - t)
    lost_t, lost_c = np.cumsum(y * t * value), np.cumsum(y * (1 - t) * value)
    k = np.arange(1, len(y) + 1)
    scale = np.divide(k, np.maximum(n_c, 1)) * (n_c > 0)
    saved = lost_c * scale - lost_t * np.divide(k, np.maximum(n_t, 1)) * (n_t > 0)
    net = saved - cost * k
    idx = np.linspace(0, len(y) - 1, 200).astype(int)
    return np.concatenate([[0.0], (idx + 1) / len(y)]), np.concatenate([[0.0], net[idx]])


def incremental_at(y: np.ndarray, t: np.ndarray, score: np.ndarray, share: float) -> float:
    """Churners prevented per 1,000 subscribers treated, when treating the top `share`."""
    k = max(1, int(share * len(y)))
    top = np.argsort(-score, kind="stable")[:k]
    yt, tt = y[top], t[top]
    if tt.sum() == 0 or (1 - tt).sum() == 0:
        return 0.0
    return float((yt[tt == 0].mean() - yt[tt == 1].mean()) * 1000)


def evaluate(y: np.ndarray, t: np.ndarray, scores: dict[str, np.ndarray]) -> dict:
    ate = float(y[t == 0].mean() - y[t == 1].mean())
    out = {
        "n": int(len(y)),
        "treated_share": float(t.mean()),
        "average_effect_per_1000": ate * 1000,
        "rankings": {},
    }
    for name, s in scores.items():
        out["rankings"][name] = {
            "qini_coefficient": qini_coefficient(y, t, s),
            "prevented_per_1000_top10": incremental_at(y, t, s, 0.10),
            "prevented_per_1000_top30": incremental_at(y, t, s, 0.30),
        }
    return out

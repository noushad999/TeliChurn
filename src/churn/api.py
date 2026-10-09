"""Real-time scoring API (FastAPI).

* ``POST /v1/predict``: score feature vectors on demand (e.g. from a feature store or CRM event)
* ``GET  /v1/subscribers/{sub_id}``: latest batch score + reasons + action (call-centre / CRM lookup)
* ``GET  /v1/model``: model provenance and test metrics
* ``POST /admin/reload``: hot-swap to the newly promoted champion without a restart
* ``GET  /health`` (liveness/readiness) and ``GET /metrics`` (Prometheus), both unauthenticated

Security and operations:

* API keys: ``CHURN_API_KEYS`` holds comma-separated keys. Only their SHA-256 digests are kept in
  memory and compared in constant time. Without it the API runs in open dev mode and logs a warning.
* Structured JSON access logs with a request id (``X-Request-ID`` is honoured and echoed). Feature
  values and identifiers are never logged.

Run: ``churn serve`` or ``uvicorn churn.api:app``. Configure with CHURN_MODEL_DIR and CHURN_SCORES.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.security import APIKeyHeader
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from churn.model import ChurnModel
from churn.uplift import UpliftModel

log = logging.getLogger("churn.api")


class Subscriber(BaseModel):
    sub_id: int | None = None
    features: dict[str, Any] = Field(
        ..., description="Feature name -> value. Missing features are treated as unknown."
    )


class PredictRequest(BaseModel):
    subscribers: list[Subscriber] = Field(..., min_length=1, max_length=10_000)


class Reason(BaseModel):
    code: str
    description: str


class Prediction(BaseModel):
    sub_id: int | None
    churn_probability: float
    risk_band: str
    reasons: list[Reason]
    primary_driver: str
    recommended_action: str
    uplift: float | None = Field(None, description="Estimated churn-probability reduction if contacted")
    sleeping_dog: bool | None = Field(None, description="True if an offer is predicted to backfire")


class PredictResponse(BaseModel):
    model_version: str
    latency_ms: float
    predictions: list[Prediction]


def _reasons(*codes: str | None) -> list[Reason]:
    return [Reason(code=c, description=ChurnModel.describe_reason(c)) for c in codes if c]


class _State:
    """Everything the API serves, swapped atomically on reload."""

    def __init__(self, model_dir: Path, scores_path: Path | None):
        self.model = ChurnModel.load(model_dir)
        self.uplift = UpliftModel.load(model_dir)
        self.version = str(self.model.metadata.get("model_version", "unknown"))
        self.scores: pl.DataFrame | None = None
        self.sub_index = None
        if scores_path and Path(scores_path).exists():
            self.scores = pl.read_parquet(scores_path).sort("sub_id")
            self.sub_index = self.scores["sub_id"].to_numpy()


def _json_log(**fields) -> None:
    log.info(json.dumps(fields, default=str))


def create_app(model_dir: str | Path | None = None, scores_path: str | Path | None = None) -> FastAPI:
    model_dir = Path(model_dir or os.getenv("CHURN_MODEL_DIR", "artifacts/model"))
    scores_path = scores_path or os.getenv("CHURN_SCORES")
    state = {"current": _State(model_dir, Path(scores_path) if scores_path else None)}
    lock = threading.Lock()

    key_digests = [
        hashlib.sha256(k.strip().encode()).digest()
        for k in os.getenv("CHURN_API_KEYS", "").split(",")
        if k.strip()
    ]
    if not key_digests:
        log.warning("CHURN_API_KEYS not set: API is running WITHOUT authentication (dev mode)")
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def require_key(key: str | None = Depends(api_key_header)) -> None:
        if not key_digests:
            return
        digest = hashlib.sha256((key or "").encode()).digest()
        if not any(hmac.compare_digest(digest, d) for d in key_digests):
            raise HTTPException(401, "Missing or invalid API key", headers={"WWW-Authenticate": "API-Key"})

    registry = CollectorRegistry()
    requests_total = Counter(
        "churn_api_requests_total", "Requests", ["endpoint", "status"], registry=registry
    )
    latency = Histogram(
        "churn_api_request_seconds",
        "Request latency",
        ["endpoint"],
        buckets=(0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
        registry=registry,
    )
    predictions_total = Counter(
        "churn_predictions_total", "Subscribers scored", ["risk_band"], registry=registry
    )

    app = FastAPI(title="Telco Churn Scoring API", version=state["current"].version)

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            route = request.scope.get("route")
            endpoint = getattr(route, "path", "unmatched")
            elapsed = time.perf_counter() - t0
            if endpoint != "/metrics":
                requests_total.labels(endpoint, str(status)).inc()
                latency.labels(endpoint).observe(elapsed)
            _json_log(
                event="request",
                request_id=rid,
                method=request.method,
                endpoint=endpoint,
                status=status,
                latency_ms=round(elapsed * 1e3, 2),
                model_version=state["current"].version,
            )

    @app.get("/health")
    def health() -> dict[str, Any]:
        s = state["current"]
        return {
            "status": "ok",
            "model_version": s.version,
            "uplift_model": s.uplift is not None,
            "batch_scores_loaded": s.scores is not None,
            "auth": bool(key_digests),
        }

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/model", dependencies=[Depends(require_key)])
    def model_info() -> dict[str, Any]:
        m = state["current"].model
        keys = [
            "model_version",
            "trained_at",
            "label_definition",
            "train_cutoffs",
            "valid_cutoff",
            "test_cutoff",
            "best_iteration",
            "train_rows",
            "test_metrics",
            "risk_thresholds",
            "data_quality",
        ]
        return {k: m.metadata.get(k) for k in keys} | {"n_features": len(m.features), "features": m.features}

    @app.post("/v1/predict", response_model=PredictResponse, dependencies=[Depends(require_key)])
    def predict(req: PredictRequest) -> PredictResponse:
        t0 = time.perf_counter()
        s = state["current"]
        X = s.model.matrix([sub.features for sub in req.subscribers])
        out = s.model.score_frame(X)
        uplift = s.uplift.predict(X) if s.uplift is not None else None
        preds = []
        for i, sub in enumerate(req.subscribers):
            band = str(out["risk_band"][i])
            predictions_total.labels(band).inc()
            preds.append(
                Prediction(
                    sub_id=sub.sub_id,
                    churn_probability=round(float(out["churn_probability"][i]), 6),
                    risk_band=band,
                    reasons=_reasons(out["reason_1"][i], out["reason_2"][i]),
                    primary_driver=str(out["primary_driver"][i]),
                    recommended_action=str(out["recommended_action"][i]),
                    uplift=None if uplift is None else round(float(uplift[i]), 6),
                    sleeping_dog=None if uplift is None else bool(uplift[i] <= 0),
                )
            )
        return PredictResponse(
            model_version=s.version, latency_ms=round((time.perf_counter() - t0) * 1e3, 2), predictions=preds
        )

    @app.get("/v1/subscribers/{sub_id}", dependencies=[Depends(require_key)])
    def subscriber(sub_id: int) -> dict[str, Any]:
        s = state["current"]
        if s.scores is None:
            raise HTTPException(503, "No batch scores loaded. Run `churn score` and set CHURN_SCORES.")
        i = int(np.searchsorted(s.sub_index, sub_id))
        if i >= len(s.sub_index) or s.sub_index[i] != sub_id:
            raise HTTPException(404, f"Subscriber {sub_id} not in the active scored base")
        row = {
            k: None if isinstance(v, float) and v != v else v for k, v in s.scores.row(i, named=True).items()
        }
        row["reasons"] = [r.model_dump() for r in _reasons(row.pop("reason_1"), row.pop("reason_2"))]
        return row

    @app.post("/admin/reload", dependencies=[Depends(require_key)])
    def reload() -> dict[str, Any]:
        """Load the current champion export (and scores) and swap it in atomically."""
        with lock:
            previous = state["current"].version
            state["current"] = _State(model_dir, Path(scores_path) if scores_path else None)
        _json_log(event="reload", previous=previous, current=state["current"].version)
        return {"previous": previous, "current": state["current"].version}

    return app


def __getattr__(name: str):
    # lazily build `app` so importing the module (e.g. in tests) doesn't require a trained model
    if name == "app":
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        return create_app()
    raise AttributeError(name)

# One image for every role: API (default), batch pipeline (CronJob / Airflow) and dashboard.
FROM python:3.11-slim

# libgomp: OpenMP runtime for LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 churn

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[dashboard]"
COPY configs ./configs
COPY docs/assets/fonts ./docs/assets/fonts
RUN mkdir -p data artifacts outputs reports && chown -R churn:churn /app

USER churn
ENV PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    CHURN_CONFIG=/app/configs/config.yaml \
    CHURN_MODEL_DIR=/app/artifacts/model \
    WEB_CONCURRENCY=2
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD curl -fs http://localhost:8000/health || exit 1
CMD ["sh", "-c", "uvicorn churn.api:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY} --no-access-log"]

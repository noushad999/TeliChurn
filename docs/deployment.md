# Deployment

One container image serves every role: the API (default command), the monthly batch job and the dashboard.

## Local: Docker Compose

```bash
CHURN_API_KEYS=my-local-key docker compose up --build
```

| Service | Port | What it does |
|:--|:--|:--|
| `pipeline` | — | one-shot `churn pipeline` (simulate → train → score) into shared volumes |
| `api` | 8000 | scoring API, starts after the pipeline succeeds |
| `dashboard` | 8501 | CVM dashboard |
| `prometheus` | 9090 | scrapes `api:8000/metrics` |

## Kubernetes

```bash
kubectl create namespace churn
kubectl -n churn create configmap churn-config --from-file=config.yaml=configs/operator_example.yaml
kubectl -n churn apply -f deploy/k8s/secret.example.yaml     # replace placeholders first, or use External Secrets
kubectl apply -k deploy/k8s
```

| Manifest | Notes |
|:--|:--|
| `api-deployment.yaml` | 2+ replicas, readiness/liveness on `/health`, non-root, read-only root FS, all capabilities dropped |
| `api-hpa.yaml` | 2–10 replicas at 70% CPU |
| `api-service.yaml` | ClusterIP; expose through your ingress/API gateway |
| `network-policy.yaml` | only the `crm`, `callcentre`, `monitoring` and `churn` namespaces may reach the API |
| `monthly-cronjob.yaml` | 3rd of each month, 02:00 Asia/Dhaka: validate → train → score → backtest → reload |
| `pvc.yaml` | shared RWX volume for data, registry, scores and reports |

Before a real rollout, adjust `pvc.yaml` storage and `monthly-cronjob.yaml` resources to the base size (see `docs/architecture.md`).

## Airflow

Operators that orchestrate with Airflow can use `deploy/airflow/churn_monthly.py` instead of the CronJob. It builds
the same task graph from `KubernetesPodOperator` tasks and needs the `apache-airflow-providers-cncf-kubernetes` package.

## Configuration

| Variable | Used by | Purpose |
|:--|:--|:--|
| `CHURN_CONFIG` | all | path to the YAML config (supports `extends:`) |
| `CHURN_API_KEYS` | API | comma-separated API keys; unset = open dev mode (logged) |
| `CHURN_PII_SALT` | ingestion | secret salt for MSISDN pseudonymisation |
| `CHURN_DB_URI` | ingestion | SQL connection URI when `source.type: sql` |
| `CHURN_MODEL_DIR` | API | champion export directory (default `artifacts/model`) |
| `CHURN_SCORES` | API | batch scores for `/v1/subscribers/{id}` |
| `WEB_CONCURRENCY` | API | uvicorn workers per pod |

## Observability

- `GET /metrics`: `churn_api_requests_total{endpoint,status}`, `churn_api_request_seconds` (histogram), `churn_predictions_total{risk_band}`.
- JSON access logs on stdout: request id, endpoint, status, latency and model version. No features or identifiers are logged.
- Batch reports: `data_quality*.json`, `drift_<month>.json`, `backtest_<month>.json` and `metrics.json`, all shown in the dashboard's Model health tab.

Suggested alerts:

- API 5xx rate > 1%
- p95 latency > 50 ms
- score PSI > 0.25
- backtest status `DEGRADED`
- data contract `fail`

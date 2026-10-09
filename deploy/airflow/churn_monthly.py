"""Airflow DAG: monthly churn retrain, scoring, backtest and API reload.

For operators that orchestrate with Airflow instead of the Kubernetes CronJob in deploy/k8s.
Each task runs the same container image; `data_interval_start` is the month that just closed.
"""

from __future__ import annotations

import pendulum
from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.providers.cncf.kubernetes.secret import Secret

AS_OF = "{{ data_interval_start.strftime('%Y-%m') }}"
BACKTEST = "{{ data_interval_start.subtract(months=2).strftime('%Y-%m') }}"
SECRETS = [Secret("env", key, "churn-secrets", key) for key in ("CHURN_API_KEYS", "CHURN_PII_SALT", "CHURN_DB_URI")]


def churn_task(task_id: str, cmd: str) -> KubernetesPodOperator:
    return KubernetesPodOperator(
        task_id=task_id,
        name=f"churn-{task_id}",
        namespace="churn",
        image="registry.example.com/cvm/telco-churn:2.0.0",
        cmds=["sh", "-c"],
        arguments=[cmd],
        secrets=SECRETS,
        env_vars={"CHURN_CONFIG": "/etc/churn/config.yaml"},
        get_logs=True,
        is_delete_operator_pod=True,
    )


with DAG(
    dag_id="churn_monthly",
    schedule="0 2 3 * *",
    start_date=pendulum.datetime(2025, 1, 1, tz="Asia/Dhaka"),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": pendulum.duration(minutes=15)},
    tags=["cvm", "churn"],
) as dag:
    validate = churn_task("validate", "churn validate")
    train = churn_task("train", f"churn train --as-of {AS_OF}")
    score = churn_task(
        "score", f"churn score --month {AS_OF} && cp outputs/scores_{AS_OF}.parquet outputs/scores_latest.parquet"
    )
    backtest = churn_task("backtest", f"churn backtest --month {BACKTEST}")
    reload_api = churn_task(
        "reload_api", 'curl -fsS -X POST -H "X-API-Key: ${CHURN_API_KEYS%%,*}" http://churn-api.churn/admin/reload'
    )

    validate >> train >> score >> reload_api
    score >> backtest

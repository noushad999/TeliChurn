# Architecture

## System view

```
 Operator DWH / exports                                   Consumers
 ─────────────────────                                    ─────────
 subscribers, usage_monthly,  ──►  ingest.py  ──►  contract.py        CRM / campaign tool ◄── campaign_<month>.csv
 recharges, network_market,        (map, hash       (conform +          Call centre / apps ◄── FastAPI (api.py)
 contacts*, campaigns*             MSISDNs)          validate)          Retention team     ◄── dashboard.py
                                                        │               Monitoring         ◄── /metrics, reports/*.json
                                                        ▼
                                    features.py: point-in-time snapshots + 60-day silence label
                                                        │
                    ┌───────────────────────────────────┼─────────────────────────────────┐
                    ▼                                   ▼                                 ▼
             train.py (monthly)                  score.py (monthly)                backtest.py (monthly)
   churn LightGBM · baselines · SHAP        champion scores whole base          realised metrics once
   uplift X-learner on campaigns            reasons for actioned rows           outcomes are known +
   out-of-time evaluation                   EV targeting + sleeping-dog         measured campaign effect
   champion/challenger gate ──► registry    guard + 10% holdout + PSI           (treated vs holdout)
```
`*` optional sources. Without them, the related features are skipped.

## Monthly cycle

| Day | Step | Command | Output |
|:--|:--|:--|:--|
| 3rd, 02:00 | Validate the closed month against the contract | `churn validate` | `reports/data_quality.json` (blocks the run on errors) |
| | Retrain with rolling out-of-time windows | `churn train --as-of <closed month>` | new registry version; promoted only if it passes the gate |
| | Score the active base with the champion | `churn score --month <closed month>` | scores, campaign list with holdout arm, drift report |
| | Measure the month whose outcome window just closed | `churn backtest --month <closed month − 2>` | realised metrics, measured save rate, DEGRADED alert |
| | Hot-reload the API | `POST /admin/reload` | serving the new champion without downtime |

The same graph ships as a Kubernetes CronJob (`deploy/k8s/monthly-cronjob.yaml`) and as an Airflow DAG
(`deploy/airflow/churn_monthly.py`).

## Key design decisions

| Decision | Why |
|:--|:--|
| Churn = 60 days of silence after an active month | Prepaid customers never cancel. The label must be observable in the data an operator actually has. |
| Point-in-time snapshots, with a test that deletes the future | Leakage is the most common reason churn models look great offline and fail live. |
| Out-of-time split (train Mar–Jul, validate Aug, test Oct) | Mirrors deployment: at month end, only months whose 60-day window has closed are labelled. |
| No resampling | Keeps probabilities calibrated, which the expected-value maths needs. |
| LightGBM, depth 6 | Best accuracy on tabular data, fast to train and score, and cheap exact TreeSHAP. |
| Reason codes = SHAP summed by family, cause over symptom | Agents act on causes ("network issue"), not symptoms ("going quiet"). |
| Uplift X-learner as a *guard*, not a ranker | Retention effects are small and noisy. Risk ranks well, and uplift best identifies whom *not* to contact. It gives a +35% out-of-time profit on a randomised campaign. |
| Seed-ensembled uplift (5 models) | A single fit's profit swung 31–46k ৳ between seeds. The ensemble removes that luck. |
| 10% holdout on every campaign | The only way to know the true save rate. The backtest measured 9% where 30% had been assumed. |
| Registry gate on the *same rows* | Compares challenger and champion fairly even when data drifts. |
| Contract before features | Bad upstream data fails loudly at ingestion, not silently in model quality. |
| One container image for every role | API, batch jobs and dashboard stay version-locked. |

## Scaling to a national operator (80M+ subscribers)

The code runs single-node at 1M subscribers in under a minute. At national scale:

1. **Snapshots in the warehouse.** `features.py` is plain columnar logic (group-bys, windowed joins) and translates directly to Spark SQL or BigQuery. Alternatively, run Polars per division partition.
2. **Scoring is embarrassingly parallel.** Shard by `sub_id` hash across CronJob pods, then concatenate the Parquet outputs.
3. **Training** on 5 monthly snapshots of 80M rows: sample non-churners (with weights restored at calibration), or use LightGBM's distributed mode.
4. **SHAP** only for actioned rows, which is already implemented: critical band + campaign targets ≈ 5–8% of the base.

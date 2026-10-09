# Runbook

## Monthly job failed

| Symptom | Likely cause | Action |
|:--|:--|:--|
| `DataContractError` in `validate` or `train` | Upstream extract changed (renamed column, new code, late partition) | Read `reports/data_quality.json`. Fix the mapping in `source.tables.*`, or ask the DWH team to reload. Never bypass the contract. |
| `Cannot label cutoff` | Run started before the month closed, or `--as-of` is wrong | Re-run with the correct `--as-of`. The last closed month must have full usage and recharges. |
| Uplift step fails (`at least two labelled campaigns`) | Too few randomised campaigns with closed outcome windows | Temporarily remove the `uplift:` section (scoring falls back to risk-only targeting), and keep running campaigns with holdouts. |
| Out of memory | Base grew | Raise CronJob memory, or shard scoring by `sub_id` (see `docs/architecture.md`). |

## Challenger rejected

`churn models` shows the decision and both models' metrics on the same rows. A rejection is not an incident: the
champion keeps serving. Investigate if it happens two months in a row (data drift, label definition change, broken
feature). To override after review:

```bash
churn promote <version>          # recorded as a manual promotion
curl -X POST -H "X-API-Key: $KEY" http://churn-api/admin/reload
```

## Bad model in production

```bash
churn rollback                   # restores the previous champion and re-exports it
curl -X POST -H "X-API-Key: $KEY" http://churn-api/admin/reload
```

## Backtest says DEGRADED

1. Check `reports/drift_<month>.json` for major PSI shifts. Network events and tariff changes are the usual causes.
2. Check the data contract status for the scored month.
3. Retrain: `churn train --as-of <latest closed month>`. The gate compares it with the champion on fresh rows.
4. If the measured save rate differs materially from `campaign.save_rate`, update the config. Targeting depth depends on it.

## API

| Symptom | Action |
|:--|:--|
| 401 | Caller's key missing or rotated. Issue a new key into `CHURN_API_KEYS` and roll the deployment. |
| 503 on `/v1/subscribers/{id}` | No batch scores mounted. Check `CHURN_SCORES` and that `scores_latest.parquet` exists. |
| p95 latency up | Check `churn_api_request_seconds` by endpoint. Batch `/v1/predict` calls of 10k rows take ~1 s by design. Scale replicas (HPA) or split calls. |

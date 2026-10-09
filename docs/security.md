# Security and privacy

Subscriber data is personal data. The system is designed so that raw identifiers never leave ingestion, and so that
every access path is authenticated and observable.

## Personal data

| Control | Implementation |
|:--|:--|
| Pseudonymisation at the boundary | MSISDNs (and contact MSISDNs) are replaced at ingestion by a salted 64-bit BLAKE2b hash (`ingest.pseudonymize`). The salt lives only in `CHURN_PII_SALT` (a secret manager), never with the data. |
| Data minimisation | The contract accepts no names, NIDs, addresses or free text. Features are behavioural aggregates. |
| Joinability without exposure | The same MSISDN hashes to the same id in every table, so joins work but numbers cannot be read back. The CRM re-identifies through its own lookup. |
| No PII in logs | Access logs record request id, endpoint, status, latency and model version only. |
| Re-identification risk | Rotating the salt breaks linkability across periods. Rotate yearly, or when staff with salt access leave. |

## Access

| Control | Implementation |
|:--|:--|
| API authentication | `X-API-Key`. Keys come from `CHURN_API_KEYS`, only SHA-256 digests are held in memory, and comparison is constant-time. Issue one key per consuming system and rotate them. |
| Network | The Kubernetes `NetworkPolicy` admits only the CRM, call-centre, monitoring and churn namespaces. TLS terminates at the ingress/API gateway. |
| Least privilege | Non-root container (uid 10001), read-only root filesystem, all Linux capabilities dropped, read-only mounts for model and scores in the API. |
| Secrets | Delivered via Kubernetes Secrets or External Secrets. `deploy/k8s/secret.example.yaml` contains placeholders only. |
| Open endpoints | Only `/health` and `/metrics`, which expose no subscriber data. |

## Model governance

- **Reproducibility.** Every registry version stores the model, feature list, category levels, thresholds, drift reference, test metrics and data-quality status.
- **Change control.** Promotion is gated automatically, manual promotions and rollbacks are recorded with a reason, and history is never rewritten.
- **Explainability.** Every actioned subscriber carries SHAP reason codes and a driver, so a decision can be explained to the subscriber or a regulator.
- **Fairness checks to add per deployment.** Compare contact rates and offer value across divisions, urban/rural and gender, and exclude protected attributes from targeting if local policy requires it (`gender` can be dropped from the feature set in config).
- **Do-not-contact.** The sleeping-dog guard blocks offers predicted to push subscribers out. Operator DND and opt-out lists should be applied as a final filter in the CRM.

# Model card: churn model `v20260927-063508`

*Generated at training time. Status in registry: champion; first model: no champion yet.*

## Model

| | |
|:--|:--|
| Type | LightGBM binary classifier (gradient-boosted trees); reason codes via TreeSHAP |
| Label | churn = no usage and no recharge in the 1 months after the snapshot month (events: usage) |
| Population | subscribers with activity in the snapshot month |
| Training snapshots | 2014-07 (74,809 rows) |
| Validation | random subscriber holdout from the training months |
| Out-of-time test | 2014-08 (92,091 subscribers, 4.4% churn) |
| Features | 40 point-in-time features; excluded by policy: none |

## Intended use

- Rank active prepaid subscribers by 60-day churn risk for retention campaigns, with reason codes and a
  next best action for CRM and call-centre agents.
- Campaign selection combines risk, subscriber value and (when available) an uplift model that blocks
  subscribers predicted to react badly to contact. Every campaign keeps a random holdout.

**Out of scope:** credit, pricing or service-eligibility decisions about individuals; any use that denies
service. Scores reflect behaviour patterns, not intent, and reasons explain the model, not causation.

## Performance (out-of-time test month)

| Model | ROC-AUC | PR-AUC | KS | Brier | Capture@10% | Lift@10% |
|:--|--:|--:|--:|--:|--:|--:|
| LightGBM | 0.853 | 0.279 | 0.549 | 0.036 | 55.0% | 5.50 |
| Logistic regression | 0.807 | 0.173 | 0.491 | 0.042 | 45.2% | 4.52 |
| Rule: days since recharge | 0.742 | 0.090 | 0.411 | – | 23.9% | 2.39 |

## Drivers

- Recharge slowdown: 0.74
- Falling voice / data / revenue: 0.68
- New SIM in the early-life churn window: 0.33
- Multi-SIM user with a weak on-net community: 0.17

## Segments and fairness

Policy reach overall: 50.9% of churners. Contact rates are expected to differ between segments (they follow risk and value). Calibration gaps, weak within-segment ranking and low reach are flagged for review.

| Dimension | Segment | Subscribers | Churn | Predicted | ROC-AUC | Contacted | Churners reached |
|:--|:--|--:|--:|--:|--:|--:|--:|
| division | circle_109 | 92,091 | 4.4% | 4.2% | 0.853 | 11.3% | 50.9% |
| plan_type | prepaid | 92,091 | 4.4% | 4.2% | 0.853 | 11.3% | 50.9% |
| tenure | 1-3 years | 36,421 | 5.3% | 5.2% | 0.831 | 14.4% | 52.2% |
| tenure | 3+ years | 38,337 | 2.2% | 2.1% | 0.860 | 5.2% | 39.0% |
| tenure | 3-11 months | 17,333 | 7.1% | 6.9% | 0.830 | 18.3% | 56.9% |

**No segment breached the review thresholds.**

## Data and privacy

- Data contract status at training: pass.
- Subscriber identifiers are pseudonymised at ingestion (salted hash). No names, national ids or free text.
- Drift reference (validation month) is stored with the model; PSI is computed on every scoring run.

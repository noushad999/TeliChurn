# Model card: churn model `v20260927-063307`

*Generated at training time. Status in registry: champion; first model: no champion yet.*

## Model

| | |
|:--|:--|
| Type | LightGBM binary classifier (gradient-boosted trees); reason codes via TreeSHAP |
| Label | churn = no usage and no recharge in the 2 months after the snapshot month (events: usage, recharge) |
| Population | subscribers with activity in the snapshot month |
| Training snapshots | 2025-03, 2025-04, 2025-05, 2025-06, 2025-07 (1,585,011 rows) |
| Validation | 2025-08 |
| Out-of-time test | 2025-10 (306,094 subscribers, 3.8% churn) |
| Features | 92 point-in-time features; excluded by policy: none |

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
| LightGBM | 0.897 | 0.548 | 0.670 | 0.024 | 74.2% | 7.42 |
| Logistic regression | 0.876 | 0.404 | 0.631 | 0.028 | 68.7% | 6.87 |
| Rule: days since recharge | 0.780 | 0.246 | 0.410 | – | 48.1% | 4.81 |

**Uplift policy** (randomised campaign, 92,540 subscribers, average effect 3.8 churners prevented per 1,000 treated):

| Policy | Qini | Best realised net value |
|:--|--:|--:|
| Risk + sleeping-dog guard | 1.39 | 43,006 BDT |
| Churn risk only | 1.10 | 31,909 BDT |
| Uplift: X-learner | 1.43 | 26,631 BDT |

## Drivers

- Falling voice / data / revenue: 0.41
- Going quiet: 0.31
- Multi-SIM user with a weak on-net community: 0.21
- Customer profile / value segment: 0.19
- New SIM in the early-life churn window: 0.18
- Competitor pull: 0.16
- Close contacts recently left the network: 0.14
- Poor network experience: 0.09
- Recharge slowdown: 0.08
- Frequent emergency balance loans: 0.07

## Segments and fairness

Policy reach overall: 39.5% of churners. Contact rates are expected to differ between segments (they follow risk and value). Calibration gaps, weak within-segment ranking and low reach are flagged for review.

| Dimension | Segment | Subscribers | Churn | Predicted | ROC-AUC | Contacted | Churners reached |
|:--|:--|--:|--:|--:|--:|--:|--:|
| division | Barishal | 13,900 | 4.7% | 4.9% | 0.891 | 3.0% | 36.0% |
| division | Chattogram | 61,752 | 4.0% | 4.4% | 0.898 | 2.9% | 40.6% |
| division | Dhaka | 98,027 | 2.8% | 3.2% | 0.889 | 1.8% | 34.2% |
| division | Khulna | 31,043 | 3.9% | 4.1% | 0.904 | 3.0% | 41.0% |
| division | Mymensingh | 23,458 | 4.6% | 4.7% | 0.894 | 3.0% | 36.9% |
| division | Rajshahi | 32,669 | 4.3% | 4.6% | 0.909 | 3.2% | 43.4% |
| division | Rangpur | 25,993 | 4.6% | 4.8% | 0.888 | 3.4% | 42.7% |
| division | Sylhet | 19,252 | 5.4% | 5.6% | 0.894 | 4.3% | 44.2% |
| gender | F | 115,718 | 3.8% | 4.2% | 0.901 | 2.8% | 40.9% |
| gender | M | 190,376 | 3.9% | 4.1% | 0.895 | 2.7% | 38.6% |
| handset | feature_phone | 90,170 | 4.8% | 5.2% | 0.888 | 3.5% | 41.0% |
| handset | smartphone_3g | 48,208 | 3.7% | 4.0% | 0.899 | 1.9% | 31.4% |
| handset | smartphone_4g | 167,716 | 3.4% | 3.7% | 0.900 | 2.5% | 40.8% |
| plan_type | postpaid | 15,809 | 1.0% | 1.3% | 0.908 | 1.9% | 38.1% |
| plan_type | prepaid | 290,285 | 4.0% | 4.3% | 0.895 | 2.8% | 39.5% |
| tenure | 0-2 months | 30,850 | 9.2% | 10.0% | 0.866 | 5.2% | 32.8% |
| tenure | 1-3 years | 49,547 | 2.8% | 3.1% | 0.896 | 2.1% | 39.3% |
| tenure | 3+ years | 170,377 | 2.9% | 3.1% | 0.885 | 1.9% | 36.2% |
| tenure | 3-11 months | 55,320 | 4.7% | 5.1% | 0.897 | 4.5% | 52.9% |
| urban | rural | 166,996 | 4.7% | 5.0% | 0.894 | 3.3% | 40.7% |
| urban | urban | 139,098 | 2.9% | 3.2% | 0.896 | 2.0% | 37.0% |

**No segment breached the review thresholds.**

## Data and privacy

- Data contract status at training: pass.
- Subscriber identifiers are pseudonymised at ingestion (salted hash). No names, national ids or free text.
- Drift reference (validation month) is stored with the model; PSI is computed on every scoring run.

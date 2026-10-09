# Validation on real prepaid data

The main results in this repository come from a simulated operator. To check that the system works on
data it wasn't designed around, the same pipeline was run, unchanged, on a real public dataset.

## Dataset

The dataset is the **Indian prepaid telecom churn case study**:

- 99,999 prepaid subscribers of one telecom circle
- four months of usage and recharge aggregates (June–September 2014), 226 columns
- widely used for teaching churn modelling

It is not redistributed here. [`examples/real_data/prepare_indian_telco.py`](../examples/real_data/prepare_indian_telco.py)
downloads it from a public mirror and maps it onto the [data contract](data_contract.md).

```bash
python examples/real_data/prepare_indian_telco.py      # download + map to contract tables
export CHURN_PII_SALT=local-demo-salt
churn --config examples/real_data/config.yaml validate
churn --config examples/real_data/config.yaml train
```

The validation holdout is drawn from the pseudonymised subscriber ids, so the salt decides which subscribers
early-stop the model. With the salt above the results match the table below exactly. Other salts moved
out-of-time ROC-AUC between 0.847 and 0.853 in test runs.

## What the contract made visible

`churn validate` passed with **0 errors and 13 warnings**. The warnings are exactly the inputs this operator's export doesn't have:

- active days, SMS, complaints, call-drop rate and data speed
- per-competitor minutes, the network/market table
- the contact graph and randomised campaigns

Those features, the uplift model and the market context were skipped automatically. Nothing had to change in the pipeline code, only in the mapping.

Three honest adaptations were needed. All are documented in the adapter:

| Issue | Handling |
|:--|:--|
| Recharges are monthly aggregates, not transactions | The last recharge of each month is an exact event (its date and amount are given). The other top-ups become events on the 1st with the average amount. Counts, amounts and days-since-last-recharge stay exact; within-month gaps are approximate. |
| Only four months | 2-month lookback and a 30-day silence window. Train on July (label: August), test out-of-time on August (label: September). Early stopping on 20% of training subscribers, because there is no spare month. |
| Churn definition | 81% of subscribers with zero September usage still recharged. The dataset's standard definition ignores top-ups, so `label_events: [usage]` is used. The default rule (no usage *and* no recharge) would give 1.6% churn instead of 4.4%. |

## Results (out-of-time: August snapshot, September outcome, 92,091 subscribers, 4.4% churn)

| | **LightGBM** | Logistic regression | Recency rule |
|:--|:--:|:--:|:--:|
| ROC-AUC | **0.853** | 0.807 | 0.742 |
| PR-AUC | **0.279** | 0.173 | 0.090 |
| KS statistic | **0.549** | 0.491 | 0.411 |
| Churners reached by contacting 10% | **55.0%** | 45.2% | 23.9% |
| Lift in the top decile | **5.5×** | 4.5× | 2.4× |

<p align="center">
  <img src="../reports/real_data/cumulative_gains.png" width="49%" alt="Cumulative gains on real data">
  <img src="../reports/real_data/churn_drivers.png" width="49%" alt="Churn drivers on real data">
</p>

## Reading the results

- **Scores are lower than on the simulated operator (0.853 vs 0.897 ROC-AUC).** That is expected. The model sees 40 features instead of 92, only two months of history, and no active-day, network or social signals. The test is also *out-of-time*. Most published notebooks on this dataset report higher numbers from random splits that mix train and test months.
- **The same signals lead.** The strongest features on real data are the voice-usage trend, days since last recharge, tenure and the recharge-amount trend. Those are exactly the recency and trend signals the simulator encodes, which is evidence that the simulated results reflect real prepaid behaviour rather than artefacts of the generator.
- **The ranking still pays.** Contacting the top 10% reaches 55% of churners, 2.3 times what the classic recency rule reaches.

## What this does not show

The dataset has no randomised campaigns, so the uplift guard and the measured save rate can't be validated here. They still need an operator's test-and-learn data. The dataset's economics (INR) are also not calibrated, so campaign values are illustrative only.

# Interview guide: defending this project

Short answers to the questions a telco data-science panel (Grameenphone, Robi, Banglalink, bKash-style fintechs) is likely to ask. Every answer points to the code that backs it.

## Problem framing

**How do you define churn for prepaid?**
Prepaid customers don't cancel; they stop using the SIM. A subscriber active in month T is labelled churn if there's no usage and no recharge in T+1 and T+2 (≈60 days). Operators typically use 60–90 days of no revenue-generating activity; the window is configurable (`churn.inactivity_months`). → `features.build_snapshot`

**Why "active in the cutoff month" as the population?**
You can only run retention on someone who is still reachable and still spending. Subscribers who already went silent are a win-back problem, not a retention problem.

**Why a 60-day forward window instead of next month?**
It gives the campaign team time to act, and it avoids counting temporary dormancy (Eid village trips, using the other SIM for a few weeks) as churn.

## Data and leakage

**How do you know there's no leakage?**
1. Features only read data ≤ the last day of the cutoff month.
2. A test deletes everything after the cutoff and asserts every feature is bit-identical (`test_features_do_not_leak_future_data`).
3. Evaluation is **out-of-time**: the test month (Oct) is after every training label window.

**Why not a random train/test split?**
Subscribers appear in several monthly snapshots, and behaviour shifts over time (tariff hike, competitor promos). A random split puts the same person and the same month in both sets and overstates performance. The model is used on *next* month, so it's evaluated on a future month.

**Why no SMOTE/ADASYN for the 4% churn rate?**
Gradient boosting handles imbalance fine for *ranking*. Resampling distorts probabilities, and the campaign maths (`p × value − cost`) needs calibrated probabilities. Brier = 0.024 and the calibration curve sits on the diagonal. v1 of this repo oversampled *before* splitting, which leaked synthetic copies of test rows into training.

**Is the data real?**
The headline results are simulated, because subscriber data can't be published. I simulated an operator DWH with realistic grain (monthly usage, event-level recharges, network/market tables) and a behavioural churn process with deliberate noise: abrupt churn, recovery, temporary dormancy.

I then ran the same pipeline, unchanged, on a public dataset of 99,999 real prepaid subscribers. Only a mapping file and an adapter were written. Out-of-time ROC-AUC was 0.853, against 0.807 for logistic regression and 0.742 for the recency rule, and contacting 10% reached 55% of churners. The top features were the same recency and trend signals the simulator encodes. The contract also caught that the dataset's "churners" mostly still recharged, which is why the label events are configurable. See `docs/real_data.md`.

## Model

**Why LightGBM and not deep learning?**
Tabular, heterogeneous features with missing values and monotone-ish effects are where boosted trees are strongest. It trains in about a minute on 1.6M rows, scores ~120k subscribers/s, and gives exact TreeSHAP explanations. The logistic-regression baseline shows the non-linearity is worth it (PR-AUC 0.55 vs 0.40).

**Which metric do you optimise?**
Ranking and business value, not accuracy. PR-AUC and capture@top-k show how many churners a fixed-size campaign reaches. KS is familiar to risk teams. The final judge is campaign **profit** on the test month.

**What are the top features?**
`days_since_last_activity`, `days_since_last_recharge`, voice/revenue/data **trends**, tenure, and the recharge-gap features. Recency and trend beat static profile; that matches practitioner experience in prepaid markets.

## Explainability and action

**How does the call centre know what to do?**
Per-subscriber TreeSHAP values are summed into 10 driver families. Symptoms (going quiet, spending less) are separated from root causes (early-life SIM, network, competitor pull, social contagion, credit stress, multi-SIM). The action comes from the strongest root cause, because an offer should fix the cause, not the symptom. → `explain.primary_driver`

**Isn't SHAP slow?**
Yes, ~1000× a plain prediction. Batch runs explain only rows that get actioned (critical band + campaign targets): The full monthly run for 1M subscribers (contract, features, scores, uplift, reasons, drift) takes 84 s on 4 vCPUs. The API explains one subscriber in ~5 ms.

## Business

**How do you choose who to target?**
Expected value: `p(churn) × save_rate × margin × ARPU × 6 months − (contact + redeemed offer cost)`, targeting only where it's positive. On top of that sits the uplift guard: never contact a subscriber whose predicted uplift is negative. → `campaign.py`, `score.py`

**Why not just target the highest risk?**
Because the highest-risk subscribers include *lost causes* (their calling circle has already moved to a competitor) and some subscribers are *sleeping dogs*: contacting them makes them leave. Randomised campaigns (50/50 treatment/holdout) let us estimate uplift directly. On an out-of-time campaign, "risk × value + sleeping-dog guard" earned 43.0k ৳ vs 31.9k ৳ for risk-only (+35%). → `uplift.py`

**Why an X-learner, and why a seed ensemble?**
Retention effects are tiny (≈4 churners per 1,000) and noisy. The X-learner (Künzel et al., 2019) borrows strength across arms and beat the T-learner here. A single fit's profit swung from 31k to 46k ৳ across seeds, so production averages five fits. Otherwise the targeting decision depends on luck. Uplift *alone* ranks well on Qini but ignores value, so it's used as a veto rather than as the ranking.

**Where does `save_rate` come from?**
Measured, not assumed. Every campaign list keeps a random 10% holdout, and `churn backtest` compares churn between treated and held-out targets once the outcome window closes. It measured 9% (95% CI for prevented churn: 1.2–6.4 per 1,000) where 30% had been assumed, so the config was recalibrated to 15% for targeted high-risk subscribers.

## Production

**How is it deployed?**
One container image, three roles:
- **Monthly job** (Kubernetes CronJob or Airflow DAG): validate → train → score → backtest → hot-reload the API.
- **FastAPI service:** API keys, Prometheus metrics, JSON logs.
- **Streamlit dashboard** for the CVM team.

The model artifact freezes feature order, category levels and risk thresholds, so inference never re-fits anything (another v1 bug).

**How do you stop a bad model reaching production?**
Every training run is registered as a challenger. It's promoted only if, on its own out-of-time test month, it doesn't regress PR-AUC or capture@10% beyond tolerance against the champion *scored on the same rows*. Rejections are recorded, and `churn rollback` restores the previous champion in one step. → `registry.py`

**How do you know when to retrain?**
There are two signals:
- **Inputs:** every scoring run computes PSI per feature and for the score against the validation month (> 0.25 → investigate or retrain). It caught a simulated December network event (drop-rate PSI 0.55).
- **Outcomes:** once labels mature, `churn backtest` compares realised capture and PR-AUC with what the model promised, and flags `DEGRADED`.

Retraining is monthly anyway (`--as-of` rolls the windows), and the gate decides whether the new model serves.

## Data and privacy

**Will it work on our data?**
Map your extract's column names and codes in config (`configs/operator_example.yaml`) and run `churn validate`. The data contract checks types, nulls, keys, ranges, allowed codes, orphan ids, month gaps and volume jumps before any feature is built. Errors block the run. → `contract.py`, `ingest.py`

**How do you handle MSISDNs?**
They're pseudonymised at ingestion with a salted BLAKE2b hash. The salt lives in a secret manager, never with the data. Joins still work across tables, but no raw number reaches features, models, scores or logs.

**What would you change at Grameenphone scale (80M+ subscribers)?**
Build snapshots in the warehouse (Spark/BigQuery/Hive) or a feature store instead of in-memory Polars. Partition scoring by region/sub_id range across workers (it's embarrassingly parallel). Keep LightGBM; it scales fine. SHAP stays cheap because only actioned subscribers (≈5–8% of the base) are explained.

**Which telco-specific features matter most?**
Beyond recency and trends:
- **Competitor share of minutes** and its trend: calls drift to the network a subscriber is about to join.
- **Social contagion:** contacts from the CDR graph who recently went silent.

Both are point-in-time safe and add signal on top of usage features.

## Governance and delivery

**How do you know the model treats segments fairly?**
Every training run produces a segment report by division, urban/rural, gender, plan, handset and tenure. It covers calibration gap, within-segment ROC-AUC, the share contacted and the share of churners reached. Contact rates are *expected* to differ, because they follow risk and value. What gets flagged is miscalibration, weak ranking inside a segment, or a segment whose churners are reached at less than half the overall rate. All 21 segments passed. New SIMs are the weakest group (ROC-AUC 0.866), which fits: they have the least history. The report goes into a generated model card with intended and out-of-scope uses.

**Does the Docker/Kubernetes setup actually work?**
CI proves it on every push. It builds the image, runs the full pipeline inside the container, and starts the API from it. It then checks that an unauthenticated call gets 401, that a keyed prediction returns a probability, and that Prometheus metrics are exposed. The Kubernetes manifests are rendered with kustomize and validated in strict mode with kubeconform against the 1.30 schemas.


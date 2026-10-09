# Finding prepaid subscribers before they go silent

*A write-up of this project: the problem, the decisions that mattered, and what the results do and don't show.*

## The problem nobody labels for you

In Bangladesh more than 95% of mobile connections are prepaid. Prepaid customers almost never cancel. They stop recharging, move their spending to a second SIM, and go quiet. Nothing in the billing system records the moment they left.

So the first task is to define churn in a way a CRM team can act on. Here, a subscriber active in a month has churned if they show no usage *and* no recharge for the next 60 days. That window is long enough to rule out a subscriber who is travelling or has simply run out of balance for a week. It is short enough that a retention offer still has someone to reach. The window and the events that count as "activity" are both configuration, because operators define them differently. The real dataset below needed a different setting.

## Three decisions that mattered

### 1. Point-in-time snapshots and out-of-time tests

Every training row is a subscriber as they looked at the end of a month, built only from data up to that cutoff. A test deletes everything after the cutoff and asserts that no feature changes. The model is trained on five months, early-stopped on a later one, and tested on a month it has never seen. That is how it will be used: in production you only have labels for months whose 60-day window has closed.

The first version of this repository skipped all of this. It oversampled *before* splitting, which leaked synthetic copies of test rows into training, and reported a random-split accuracy. The rewrite starts from the question "what would the model have known on the day it scored?"

### 2. Features that describe telecom behaviour, not generic columns

The 92 features are what a prepaid CRM analyst looks at:

- recency and active-day trends
- recharge gaps and ticket sizes
- the network quality each subscriber experienced
- credit stress (emergency-balance loans)
- the share of their minutes going to a competitor's network
- how many of their regular contacts recently went silent

The last two matter most in a multi-SIM market. Calls drifting to Robi or Banglalink are often the first sign of a switch, and subscribers are much more likely to leave once their circle has left.

### 3. Predicting risk is not the same as deciding whom to call

A churn score answers *who is at risk*. A campaign only earns money on subscribers whose behaviour the offer *changes*. Some subscribers are going to leave whatever you do. Some will stay anyway. And some, the *sleeping dogs*, are more likely to leave if you contact them, because the call reminds them to look at other offers.

The simulated operator runs randomised campaigns (half treated, half held out). An uplift model, an X-learner averaged over five seeds, estimates the effect of contact for each subscriber. It is not used to rank on its own. On its own it ignores subscriber value and misses high-value churners. Instead, the list is ranked by risk × value, and the uplift model acts as a guard that removes predicted sleeping dogs. On the test campaign that lifted net campaign value from 31.9k to 43.0k ৳, a **35%** improvement over plain risk ranking.

Every live campaign also keeps a random 10% holdout. Two months later, `churn backtest` measures the real save rate. It came out at **9%**, where the configuration had originally assumed 30%. That gap is the reason the holdout exists.

## Results

| Out-of-time test | ROC-AUC | Churners reached by contacting 10% |
|:--|--:|--:|
| Simulated operator, October (306k subscribers, 3.8% churn) | 0.897 | 74.2% |
| Same pipeline, real prepaid data (92k subscribers, 4.4% churn) | 0.853 | 55.0% |
| Recency rule on the real data ("days since last recharge") | 0.742 | 23.9% |

The real-data run used a public dataset of 99,999 Indian prepaid subscribers. Only a mapping file and an adapter were written. The data contract passed with no errors and flagged the 13 inputs that export lacks. The pipeline skipped those features and the uplift model on its own.

The real score is lower than the simulated one, and it should be. The real export has 40 features instead of 92, a two-month lookback instead of three, and no network, social or campaign data. What carried over is the ranking: the same recency and trend signals lead, and the model reaches 2.3 times as many churners as the classic rule for the same campaign size.

## Running it like a product

A model notebook is not something an operator can buy. The rest of the repository is the part that makes it one:

- **Data contract.** Every extract is validated for types, keys, codes, orphans and month gaps before training. Identifiers are pseudonymised with a salted hash at the boundary.
- **Registry.** Each model is versioned. A challenger is promoted only if it doesn't lose to the champion on the same rows. Rollback is one command.
- **Monitoring.** Each scoring run computes PSI drift per feature. A backtest compares realised accuracy with what was promised.
- **Explanations.** SHAP values roll up into driver families. The root cause (not the symptom) picks the next best action for the call-centre agent.
- **Fairness.** A segment report checks calibration, ranking and reach by division, urban/rural, gender, plan, handset and tenure. It is written into a generated model card with intended and out-of-scope uses.
- **Serving.** A FastAPI service with hashed API keys, Prometheus metrics and hot reload, plus a Streamlit dashboard for the CVM team.
- **Deployment.** Docker, Kubernetes (with a CronJob, autoscaling and network policy) and an Airflow DAG.
- **CI.** CI builds the image, runs the whole pipeline inside it, calls the API from the container and validates the Kubernetes manifests on every push.

A monthly run for one million subscribers takes 84 seconds on four vCPUs.

## What this doesn't show

- **The simulated figures are not an operator's results.** They illustrate the method on data with known structure. The real-data run shows that the churn model transfers, not that the simulated campaign economics do.
- **The uplift guard is untested on real data.** It depends on randomised campaigns, which no public dataset has. An operator without holdouts should start them before relying on it.
- **Reason codes explain the model, not the world.** A driver is a hypothesis for the retention team. Only the holdout-measured effect is causal.

## If I were deploying this at an operator

1. Map one month of the real warehouse extract onto the contract and fix mappings until `churn validate` is clean.
2. Train on the last closed months. Check the model card's segment table with the CVM and legal teams before any campaign.
3. Run the first campaigns as pure test-and-learn: half treated, half held out, so the save rate and the uplift model are learned from the operator's own customers.
4. Recalibrate offer cost and save rate from the first backtest. Only then let the expected-value policy size the campaigns.

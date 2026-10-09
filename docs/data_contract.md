# Data contract

Every source (simulated, CSV, Parquet or SQL) is mapped onto these canonical tables before any feature is
computed. `churn validate` checks a source against the contract and writes `reports/data_quality.json`.
Map an operator's own column names and codes in the `source:` config section
(see [`configs/operator_example.yaml`](../configs/operator_example.yaml)).

Only the core columns are required; every other KPI and table is optional, and features built from
missing inputs are skipped (see the [real-data validation](real_data.md) for an operator export with 13 of them
missing).

**Errors block training and scoring:** missing required tables or columns, values that cannot be cast,
nulls in required fields, duplicate keys, out-of-range or unknown values above 0.1% of rows, and
subscriber ids missing from `subscribers`. **Warnings are logged:** missing optional sources (their
features are skipped), gaps in the monthly series, month-over-month volume jumps above 25%, and small
shares of out-of-range values.

Identifiers are pseudonymised at ingestion with a salted 64-bit BLAKE2b hash (`CHURN_PII_SALT`), so
raw MSISDNs never reach features, models, scores or logs.

### `subscribers` (required)

one row per SIM

Key: `sub_id`

| Column | Type | Required | Nullable | Rule | Notes |
|:--|:--|:--:|:--:|:--|:--|
| `sub_id` | id | ✓ |  |  | pseudonymised subscriber id (hash of MSISDN) |
| `activation_month` | date | ✓ |  |  | first day of the SIM activation month |
| `division` | str | ✓ |  |  |  |
| `urban` | bool | ✓ | ✓ |  |  |
| `gender` | str | ✓ | ✓ | M / F |  |
| `age` | int | ✓ | ✓ | [10, 100] |  |
| `plan_type` | str | ✓ |  | prepaid / postpaid |  |
| `handset` | str | ✓ | ✓ | feature_phone / smartphone_3g / smartphone_4g |  |
| `dual_sim` | bool | ✓ | ✓ |  |  |
| `app_user` | bool | ✓ | ✓ |  |  |
| `mfs_user` | bool | ✓ | ✓ |  |  |

### `usage_monthly` (required)

one row per subscriber-month with any revenue-generating activity

Key: `sub_id, month`

| Column | Type | Required | Nullable | Rule | Notes |
|:--|:--|:--:|:--:|:--|:--|
| `sub_id` | id | ✓ |  |  |  |
| `month` | date | ✓ |  |  | first day of the month |
| `voice_min` | float | ✓ |  | [0, ] |  |
| `onnet_min` | float | ✓ |  | [0, ] |  |
| `offnet_robi_min` | float |  |  | [0, ] |  |
| `offnet_banglalink_min` | float |  |  | [0, ] |  |
| `offnet_teletalk_min` | float |  |  | [0, ] |  |
| `data_mb` | float | ✓ |  | [0, ] |  |
| `sms_cnt` | int |  |  | [0, ] |  |
| `data_pack_cnt` | int |  |  | [0, ] |  |
| `revenue` | float | ✓ |  | [0, ] |  |
| `days_active` | int |  |  | [1, 31] |  |
| `last_active_day` | int |  |  | [1, 31] |  |
| `emergency_loan_cnt` | int |  |  | [0, ] |  |
| `complaint_cnt` | int |  |  | [0, ] |  |
| `drop_call_rate` | float |  | ✓ | [0, 1] |  |
| `data_speed_mbps` | float |  | ✓ | [0, ] |  |

### `recharges` (required)

one row per top-up transaction

Key: `—`

| Column | Type | Required | Nullable | Rule | Notes |
|:--|:--|:--:|:--:|:--|:--|
| `sub_id` | id | ✓ |  |  |  |
| `ts` | datetime | ✓ |  |  |  |
| `amount` | float | ✓ |  | [0.01, ] |  |
| `channel` | str | ✓ | ✓ |  |  |

### `network_market` (optional)

division-month network KPIs and market intelligence (optional)

Key: `division, month`

| Column | Type | Required | Nullable | Rule | Notes |
|:--|:--|:--:|:--:|:--|:--|
| `division` | str | ✓ |  |  |  |
| `month` | date | ✓ |  |  |  |
| `avg_drop_call_rate` | float | ✓ |  | [0, 1] |  |
| `network_outage` | bool | ✓ |  |  |  |
| `competitor_promo` | bool | ✓ |  |  |  |

### `contacts` (optional)

top on-net contacts per subscriber from CDRs (optional)

Key: `sub_id, contact_id`

| Column | Type | Required | Nullable | Rule | Notes |
|:--|:--|:--:|:--:|:--|:--|
| `sub_id` | id | ✓ |  |  |  |
| `contact_id` | id | ✓ |  |  |  |
| `minutes` | float | ✓ |  | [0, ] |  |

### `campaigns` (optional)

randomised retention campaigns: treatment vs holdout (optional)

Key: `campaign_month, sub_id`

| Column | Type | Required | Nullable | Rule | Notes |
|:--|:--|:--:|:--:|:--|:--|
| `campaign_month` | date | ✓ |  |  |  |
| `sub_id` | id | ✓ |  |  |  |
| `treatment` | bool | ✓ |  |  |  |
| `offer` | str | ✓ | ✓ |  |  |

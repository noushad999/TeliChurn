"""Training with out-of-time validation, baselines, business evaluation and artifact export."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn import evaluate as ev
from churn import ingest, model_card
from churn.campaign import campaign_summary, expected_net, profit_curve
from churn.config import month_label, parse_month
from churn.explain import FAMILY_LABELS, global_family_importance
from churn.fairness import segment_report
from churn.features import CATEGORICAL, LABEL, arpu_column, build_many, build_snapshot, feature_columns
from churn.model import ChurnModel
from churn.monitor import build_reference, save_reference
from churn.registry import Registry, export_champion, gate
from churn.uplift import UpliftModel, campaign_frame, qini_curve, value_curve
from churn.uplift import evaluate as evaluate_uplift


class Timer:
    def __init__(self):
        self.marks: dict[str, float] = {}

    def __call__(self, name: str):
        timer = self

        class _T:
            def __enter__(self):
                self.t0 = time.perf_counter()

            def __exit__(self, *exc):
                timer.marks[name] = round(time.perf_counter() - self.t0, 2)
                print(f"  [{timer.marks[name]:>6.1f}s] {name}")

        return _T()


def _to_pandas(df: pl.DataFrame, features: list[str], categories: dict[str, list[str]]) -> pd.DataFrame:
    X = df.select(features).to_pandas()
    for col, levels in categories.items():
        X[col] = pd.Categorical(X[col], categories=levels)
    return X


def _logreg_baseline(X_tr: pd.DataFrame, y_tr: np.ndarray, seed: int, max_rows: int = 300_000):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_tr), min(max_rows, len(X_tr)), replace=False)
    cats = [c for c in CATEGORICAL if c in X_tr.columns]
    num = [c for c in X_tr.columns if c not in cats]
    pre = ColumnTransformer(
        [
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
        ]
    )
    model = make_pipeline(pre, LogisticRegression(max_iter=500, C=0.5))
    model.fit(X_tr.iloc[idx], y_tr[idx])
    return model


def run(cfg: dict) -> dict:
    paths, tcfg, ccfg = cfg["paths"], cfg["training"], cfg["churn"]
    reports = Path(paths["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    Path(paths["features_dir"]).mkdir(parents=True, exist_ok=True)
    timer = Timer()
    snap_kw = dict(
        lookback_months=ccfg["lookback_months"],
        inactivity_months=ccfg["inactivity_months"],
        label_events=tuple(ccfg.get("label_events", ("usage", "recharge"))),
    )

    print("Training churn model")
    with timer("load + validate data (contract)"):
        tables, dq = ingest.load(cfg)
        dq.save(reports / "data_quality.json")
        print(
            f"  data contract: {'PASS' if dq.ok else 'FAIL'}, {len(dq.issues)} issue(s) -> reports/data_quality.json"
        )
    with timer("build feature snapshots"):
        train = build_many(tables, tcfg["train_cutoffs"], **snap_kw)
        if tcfg.get("valid_cutoff"):
            valid = build_snapshot(tables, tcfg["valid_cutoff"], **snap_kw)
        else:
            # short histories: hold out a random share of subscribers from the training months for
            # early stopping (grouped by subscriber); the test month stays strictly out-of-time
            held = (
                train.select("sub_id")
                .unique()
                .sample(fraction=tcfg.get("valid_fraction", 0.2), seed=cfg["seed"])
            )
            valid = train.join(held, on="sub_id", how="semi")
            train = train.join(held, on="sub_id", how="anti")
        test = build_snapshot(tables, tcfg["test_cutoff"], **snap_kw)
        for name, df in (("train", train), ("valid", valid), ("test", test)):
            df.write_parquet(Path(paths["features_dir"]) / f"{name}.parquet")

    excluded = set(cfg.get("features", {}).get("exclude") or [])
    # drop policy-excluded features and features the source never populates (e.g. unknown age)
    features = [
        f for f in feature_columns(train) if f not in excluded and train[f].null_count() < train.height
    ]
    cat_features = [c for c in CATEGORICAL if c in features]
    categories = {c: sorted(train[c].drop_nulls().unique().to_list()) for c in cat_features}
    X_tr, X_va, X_te = (_to_pandas(d, features, categories) for d in (train, valid, test))
    y_tr, y_va, y_te = (d[LABEL].to_numpy() for d in (train, valid, test))
    print(
        f"  train {len(y_tr):,} rows ({len(tcfg['train_cutoffs'])} snapshots, churn {y_tr.mean():.2%}) | "
        f"valid {len(y_va):,} ({y_va.mean():.2%}) | test {len(y_te):,} ({y_te.mean():.2%}) | {len(features)} features"
    )

    with timer("train LightGBM"):
        booster = lgb.train(
            tcfg["lgbm"] | {"seed": cfg["seed"]},
            lgb.Dataset(X_tr, y_tr, categorical_feature=cat_features, free_raw_data=False),
            num_boost_round=tcfg["num_boost_round"],
            valid_sets=[lgb.Dataset(X_va, y_va, categorical_feature=cat_features)],
            callbacks=[lgb.early_stopping(tcfg["early_stopping_rounds"], verbose=False)],
        )
    with timer("train logistic-regression baseline"):
        logreg = _logreg_baseline(X_tr, y_tr, cfg["seed"])

    with timer("evaluate on out-of-time test month"):
        p_va = booster.predict(X_va, num_iteration=booster.best_iteration)
        p_te = booster.predict(X_te, num_iteration=booster.best_iteration)
        p_lr = logreg.predict_proba(X_te)[:, 1]
        # the pre-ML business rule: longest time since last recharge = highest risk
        p_rule = test["days_since_last_recharge"].fill_null(999).to_numpy().astype(float)
        scores = {"LightGBM": p_te, "Logistic regression": p_lr, "Rule: days since recharge": p_rule}
        metrics = {name: ev.classification_metrics(y_te, s) for name, s in scores.items()}
        gains = ev.gains_table(y_te, p_te)
        gains.to_csv(reports / "gains_test.csv", index=False)

    b = cfg["risk_bands"]
    risk_thresholds = {
        "critical": float(np.quantile(p_va, 1 - b["critical"])),
        "high": float(np.quantile(p_va, 1 - b["critical"] - b["high"])),
        "medium": float(np.quantile(p_va, 1 - b["critical"] - b["high"] - b["medium"])),
    }

    with timer("SHAP reason codes + campaign economics"):
        model = ChurnModel(
            booster,
            {
                "features": features,
                "categories": categories,
                "risk_thresholds": risk_thresholds,
            },
        )
        sample = np.random.default_rng(cfg["seed"]).choice(len(X_te), min(20_000, len(X_te)), replace=False)
        family_importance = global_family_importance(model.contributions(X_te.iloc[sample]), features)
        arpu = test[arpu_column(test)].to_numpy()
        camp_cfg = cfg["campaign"]
        campaign = campaign_summary(y_te, p_te, arpu, camp_cfg)
        rng = np.random.default_rng(cfg["seed"])
        curves = {
            "LightGBM expected value": profit_curve(y_te, expected_net(p_te, arpu, camp_cfg), arpu, camp_cfg),
            "Rule: days since recharge": profit_curve(y_te, p_rule, arpu, camp_cfg),
            "Random targeting": profit_curve(y_te, rng.random(len(y_te)), arpu, camp_cfg),
        }

    uplift_results = None
    if tables.get("campaigns") is not None and cfg.get("uplift"):
        with timer("uplift model on randomised campaigns"):
            uplift_results, uplift_model = _train_uplift(
                cfg, tables, train, snap_kw, features, categories, booster, reports
            )

    with timer("segment & fairness report"):
        selected = expected_net(p_te, arpu, camp_cfg) > 0
        if uplift_results is not None:  # production policy: never contact predicted sleeping dogs
            selected &= uplift_model.predict(X_te) > 0
        segments = segment_report(test, p_te, selected, cfg.get("fairness"))

    with timer("charts"):
        ev.plot_roc_pr(y_te, scores, reports / "roc_pr.png")
        ev.plot_gains(y_te, scores, reports / "cumulative_gains.png")
        ev.plot_calibration(y_te, p_te, reports / "calibration.png")
        ev.plot_driver_importance(family_importance, FAMILY_LABELS, reports / "churn_drivers.png")
        ev.plot_profit(
            curves,
            campaign["policy_targets"],
            len(y_te),
            camp_cfg["currency"],
            reports / "campaign_profit.png",
        )

    with timer("register model + champion/challenger gate"):
        registry = Registry(paths["registry_dir"])
        version = registry.new_version()
        vdir = registry.path(version)
        top_features = sorted(
            zip(features, booster.feature_importance("gain").tolist(), strict=True), key=lambda kv: -kv[1]
        )[:25]
        model.metadata.update(
            {
                "model_version": version,
                "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "label_definition": f"no usage and no recharge in the {ccfg['inactivity_months']} months after the snapshot month",
                "lookback_months": ccfg["lookback_months"],
                "train_cutoffs": tcfg["train_cutoffs"],
                "valid_cutoff": tcfg["valid_cutoff"],
                "test_cutoff": tcfg["test_cutoff"],
                "best_iteration": booster.best_iteration,
                "train_rows": int(len(y_tr)),
                "test_metrics": metrics["LightGBM"],
                "data_quality": dq.to_dict()["status"],
            }
        )
        model.save(vdir)
        save_reference(build_reference(X_va, p_va), vdir)
        if uplift_results is not None:
            uplift_model.save(vdir)

        # fair comparison: score the current champion on exactly the challenger's test rows
        champion_metrics = None
        if (champ_dir := registry.champion_dir()) is not None:
            champ = ChurnModel.load(champ_dir)
            champion_metrics = ev.classification_metrics(y_te, champ.predict(champ.matrix(test)))
        promote, reason = gate(
            metrics["LightGBM"], champion_metrics, cfg.get("registry", {}).get("tolerance", 0.005)
        )
        registry.register(
            version,
            {
                "test_cutoff": tcfg["test_cutoff"],
                "roc_auc": metrics["LightGBM"]["roc_auc"],
                "pr_auc": metrics["LightGBM"]["pr_auc"],
                "capture_top10": metrics["LightGBM"]["capture_top10"],
                "champion_on_same_rows": champion_metrics
                and {k: champion_metrics[k] for k in ("roc_auc", "pr_auc", "capture_top10")},
            },
        )
        if promote:
            registry.promote(version, reason)
            export_champion(registry, paths["model_dir"])
        else:
            registry.reject(version, reason)

    results = {
        "model_version": version,
        "promoted": promote,
        "promotion_decision": reason,
        "data": {
            "train_rows": int(len(y_tr)),
            "valid_rows": int(len(y_va)),
            "test_rows": int(len(y_te)),
            "features": len(features),
            "train_churn_rate": float(y_tr.mean()),
            "test_churn_rate": float(y_te.mean()),
        },
        "best_iteration": booster.best_iteration,
        "metrics_test": metrics,
        "risk_thresholds": risk_thresholds,
        "driver_importance": family_importance,
        "top_features_gain": dict(top_features),
        "campaign_test": campaign,
        "uplift_test": uplift_results,
        "segments": segments,
        "timings_s": timer.marks,
    }
    (vdir / "metrics.json").write_text(json.dumps(results, indent=2))
    (reports / "metrics.json").write_text(json.dumps(results, indent=2))
    card = model_card.render(results, cfg, model.metadata)
    (vdir / "model_card.md").write_text(card)
    (reports / "model_card.md").write_text(card)
    _print_summary(results, gains, camp_cfg["currency"])
    if segments["flags"]:
        print(f"\nSegment review flags ({len(segments['flags'])}):")
        for f in segments["flags"]:
            print(f"  {f['dimension']}={f['segment']}: {'; '.join(f['issues'])}")
    print(f"\nModel {version}: {'PROMOTED to champion' if promote else 'REJECTED'} — {reason}")
    return results


def _train_uplift(
    cfg, tables, train, snap_kw, features, categories, booster, reports
) -> tuple[dict, UpliftModel]:
    """Fit uplift learners on earlier randomised campaigns; evaluate policies out-of-time on a later one.

    Production policy ("risk + sleeping-dog guard"): rank by churn risk × value, but never contact a
    subscriber the X-learner predicts the offer would push out (negative uplift).
    """
    ucfg, camp_cfg = dict(cfg["uplift"]), cfg["campaign"]
    campaigns = tables["campaigns"]
    if ucfg.get("auto"):
        # rolling mode: every campaign whose outcome window has closed; newest is the test campaign
        latest = parse_month(cfg["training"]["test_cutoff"])
        months = sorted(m for m in campaigns["campaign_month"].unique().to_list() if m <= latest)
        if len(months) < 2:
            raise ValueError("Uplift needs at least two labelled campaigns in rolling mode")
        ucfg["train_campaigns"] = [month_label(m) for m in months[:-1]]
        ucfg["test_campaign"] = month_label(months[-1])
    fit_frames = []
    for month in ucfg["train_campaigns"]:
        snap = train.filter(pl.col("snapshot") == month)
        if snap.height == 0:
            snap = build_snapshot(tables, month, **snap_kw)
        fit_frames.append(campaign_frame(snap, campaigns, month))
    fit = pl.concat(fit_frames, how="vertical_relaxed")
    test = campaign_frame(
        build_snapshot(tables, ucfg["test_campaign"], **snap_kw), campaigns, ucfg["test_campaign"]
    )

    X_fit, X_test = (_to_pandas(d, features, categories) for d in (fit, test))
    t_fit, t_test = (d["treatment"].cast(pl.Int8).to_numpy() for d in (fit, test))
    y_fit, y_test = (d[LABEL].to_numpy() for d in (fit, test))
    model = UpliftModel.fit(X_fit, t_fit, y_fit, learner="x", seed=cfg["seed"])
    model.metadata.update(
        {"train_campaigns": ucfg["train_campaigns"], "test_campaign": ucfg["test_campaign"]}
    )
    t_learner = UpliftModel.fit(X_fit, t_fit, y_fit, learner="t", seed=cfg["seed"])

    uplift = model.predict(X_test)
    risk = booster.predict(X_test)
    guarded = np.where(uplift > 0, risk, -1.0)
    rand = np.random.default_rng(cfg["seed"]).random(len(y_test))
    results = evaluate_uplift(
        y_test,
        t_test,
        {
            "Risk + sleeping-dog guard": guarded,
            "Uplift: X-learner": uplift,
            "Uplift: T-learner": t_learner.predict(X_test),
            "Churn risk only": risk,
            "Random": rand,
        },
    )
    results["sleeping_dog_share"] = float((uplift <= 0).mean())
    # economics: rank by expected value, measure realised incremental value on the randomised test
    value = camp_cfg["margin"] * test[arpu_column(test)].to_numpy() * camp_cfg["clv_months"]
    cost = camp_cfg["contact_cost"] + camp_cfg["offer_redeem_rate"] * camp_cfg["offer_cost"]
    rankings = {
        "Risk × value + sleeping-dog guard": np.where(uplift > 0, risk * value, -1.0),
        "Churn risk × value": risk * value,
        "Uplift × value": uplift * value,
        "Random": rand,
    }
    profit = {name: value_curve(y_test, t_test, score, value, cost) for name, score in rankings.items()}
    results["profit_best"] = {
        name: {"best_share": float(x[np.argmax(v)]), "best_net_value": float(v.max())}
        for name, (x, v) in profit.items()
    }
    qini = {
        name: qini_curve(y_test, t_test, sc)
        for name, sc in (
            ("Risk + sleeping-dog guard", guarded),
            ("Churn risk only", risk),
            ("Uplift: X-learner", uplift),
            ("Random", rand),
        )
    }
    ev.plot_uplift(
        qini,
        {k: v for k, v in profit.items() if k != "Uplift × value"},
        camp_cfg["currency"],
        reports / "uplift.png",
    )
    return results, model


def _print_summary(results: dict, gains: pd.DataFrame, currency: str) -> None:
    print("\nOut-of-time test results")
    print(f"  {'model':<28}{'ROC-AUC':>9}{'PR-AUC':>9}{'KS':>7}{'lift@10%':>10}{'capture@10%':>13}")
    for name, m in results["metrics_test"].items():
        print(
            f"  {name:<28}{m['roc_auc']:>9.3f}{m['pr_auc']:>9.3f}{m['ks']:>7.3f}"
            f"{m['lift_top10']:>10.2f}{m['capture_top10']:>13.1%}"
        )
    print("\nGains table (LightGBM, test month)")
    print(gains.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    c = results["campaign_test"]
    print(
        f"\nCampaign: target {c['policy_targets']:,} ({c['policy_target_share']:.1%}) -> net "
        f"{c['policy_profit']:,.0f} {currency} vs random same size {c['random_targeting_profit_same_k']:,.0f} {currency}"
    )
    u = results.get("uplift_test")
    if u:
        print(
            f"\nUplift (out-of-time campaign, {u['n']:,} subs, avg effect {u['average_effect_per_1000']:.1f}/1000)"
        )
        for name, r in u["rankings"].items():
            print(
                f"  {name:<28} Qini {r['qini_coefficient']:>7.2f}   prevented/1000 top10% "
                f"{r['prevented_per_1000_top10']:>6.1f}   top30% {r['prevented_per_1000_top30']:>6.1f}"
            )
        for name, r in u["profit_best"].items():
            print(
                f"  best campaign by {name:<22} net {r['best_net_value']:>12,.0f} {currency} at {r['best_share']:.0%}"
            )

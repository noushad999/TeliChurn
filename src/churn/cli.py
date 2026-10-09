"""Command line entry point: `churn <command>` (or `python -m churn <command>`)."""

from __future__ import annotations

import argparse
import time


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="churn", description="Telco prepaid churn prediction pipeline")
    parser.add_argument("--config", default=None, help="Path to config YAML (default: configs/config.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sim = sub.add_parser("simulate", help="Generate the synthetic operator data warehouse")
    p_sim.add_argument("--subscribers", type=int, default=None, help="Initial active base size")
    sub.add_parser("validate", help="Check the configured data source against the data contract")
    p_train = sub.add_parser("train", help="Build snapshots, train + evaluate, register model")
    p_train.add_argument(
        "--as-of", default=None, help="Last month with complete data (YYYY-MM): rolls all windows"
    )
    p_score = sub.add_parser("score", help="Batch-score the active base for a month")
    p_score.add_argument(
        "--month", default=None, help="Snapshot month YYYY-MM (default: training.score_cutoff)"
    )
    p_bt = sub.add_parser("backtest", help="Realised performance + measured campaign effect for a past month")
    p_bt.add_argument("--month", required=True, help="Scored month YYYY-MM whose outcome window has passed")
    sub.add_parser("models", help="List registered model versions and the champion")
    p_pr = sub.add_parser("promote", help="Promote a registered version to champion")
    p_pr.add_argument("version")
    sub.add_parser("rollback", help="Restore the previous champion")
    p_all = sub.add_parser("pipeline", help="simulate -> train -> score")
    p_all.add_argument("--subscribers", type=int, default=None)
    p_serve = sub.add_parser("serve", help="Start the scoring API")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--scores", default=None, help="Batch scores parquet for /v1/subscribers lookups")

    args = parser.parse_args(argv)

    from churn.config import load_config

    overrides = {}
    if getattr(args, "subscribers", None):
        overrides = {"simulation": {"n_subscribers": args.subscribers}}
    cfg = load_config(args.config, overrides)
    if getattr(args, "as_of", None):
        from churn.config import rolling_windows

        cfg["training"].update(
            rolling_windows(args.as_of, inactivity_months=cfg["churn"]["inactivity_months"])
        )
        if cfg.get("uplift"):
            cfg["uplift"]["auto"] = True
    t0 = time.perf_counter()

    if args.command == "validate":
        from pathlib import Path

        from churn import ingest

        _, report = ingest.load(cfg, strict=False)
        print(report.summary())
        report.save(Path(cfg["paths"]["reports_dir"]) / "data_quality.json")
        raise SystemExit(0 if report.ok else 1)
    if args.command in ("models", "promote", "rollback"):
        from churn.registry import Registry, export_champion

        reg = Registry(cfg["paths"]["registry_dir"])
        if args.command == "promote":
            reg.promote(args.version, reason="manual promotion")
            export_champion(reg, cfg["paths"]["model_dir"])
        elif args.command == "rollback":
            print(f"Rolled back to {reg.rollback()}")
            export_champion(reg, cfg["paths"]["model_dir"])
        print(f"{'version':<22}{'status':<12}{'test':<9}{'ROC-AUC':>8}{'PR-AUC':>8}{'cap@10':>8}  decision")
        for v in reg.versions():
            print(
                f"{v['version']:<22}{v['status']:<12}{v.get('test_cutoff', ''):<9}{v.get('roc_auc', 0):>8.3f}"
                f"{v.get('pr_auc', 0):>8.3f}{v.get('capture_top10', 0):>8.1%}  {v.get('decision', '')}"
            )
        return
    if args.command == "backtest":
        from churn import backtest

        backtest.run(cfg, args.month)
        return
    if args.command in ("simulate", "pipeline"):
        from churn import simulate

        simulate.run(cfg)
    if args.command in ("train", "pipeline"):
        from churn import train

        train.run(cfg)
    if args.command in ("score", "pipeline"):
        from churn import score

        score.run(cfg, getattr(args, "month", None))
    if args.command == "serve":
        import os
        from pathlib import Path

        import uvicorn

        from churn.api import create_app

        scores = args.scores or os.getenv("CHURN_SCORES")
        if scores is None:
            candidates = sorted(Path(cfg["paths"]["scores_dir"]).glob("scores_*.parquet"))
            scores = str(candidates[-1]) if candidates else None
        uvicorn.run(create_app(cfg["paths"]["model_dir"], scores), host=args.host, port=args.port)
        return
    print(f"\nDone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()

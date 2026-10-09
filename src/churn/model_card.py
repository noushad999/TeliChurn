"""Model card, generated for every registered version (reports/model_card.md + registry copy)."""

from __future__ import annotations

from churn.explain import FAMILY_LABELS
from churn.fairness import markdown_table


def render(results: dict, cfg: dict, metadata: dict) -> str:
    m = results["metrics_test"]
    ccfg, tcfg = cfg["churn"], cfg["training"]
    excluded = (cfg.get("features") or {}).get("exclude") or []
    fair = results.get("segments") or {}
    lines = [
        f"# Model card: churn model `{results['model_version']}`",
        "",
        f"*Generated at training time. Status in registry: "
        f"{'champion' if results.get('promoted') else 'challenger (not promoted)'}; "
        f"{results.get('promotion_decision', '')}.*",
        "",
        "## Model",
        "",
        "| | |",
        "|:--|:--|",
        "| Type | LightGBM binary classifier (gradient-boosted trees); reason codes via TreeSHAP |",
        f"| Label | churn = {metadata.get('label_definition')} (events: {', '.join(ccfg.get('label_events', ['usage', 'recharge']))}) |",
        "| Population | subscribers with activity in the snapshot month |",
        f"| Training snapshots | {', '.join(tcfg['train_cutoffs'])} ({results['data']['train_rows']:,} rows) |",
        f"| Validation | {tcfg.get('valid_cutoff') or 'random subscriber holdout from the training months'} |",
        f"| Out-of-time test | {tcfg['test_cutoff']} ({results['data']['test_rows']:,} subscribers, "
        f"{results['data']['test_churn_rate']:.1%} churn) |",
        f"| Features | {results['data']['features']} point-in-time features; excluded by policy: "
        f"{', '.join(excluded) if excluded else 'none'} |",
        "",
        "## Intended use",
        "",
        "- Rank active prepaid subscribers by 60-day churn risk for retention campaigns, with reason codes and a",
        "  next best action for CRM and call-centre agents.",
        "- Campaign selection combines risk, subscriber value and (when available) an uplift model that blocks",
        "  subscribers predicted to react badly to contact. Every campaign keeps a random holdout.",
        "",
        "**Out of scope:** credit, pricing or service-eligibility decisions about individuals; any use that denies",
        "service. Scores reflect behaviour patterns, not intent, and reasons explain the model, not causation.",
        "",
        "## Performance (out-of-time test month)",
        "",
        "| Model | ROC-AUC | PR-AUC | KS | Brier | Capture@10% | Lift@10% |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for name, r in m.items():
        brier = f"{r['brier']:.3f}" if "brier" in r else "–"
        lines.append(
            f"| {name} | {r['roc_auc']:.3f} | {r['pr_auc']:.3f} | {r['ks']:.3f} | {brier} "
            f"| {r['capture_top10']:.1%} | {r['lift_top10']:.2f} |"
        )
    if u := results.get("uplift_test"):
        lines += [
            "",
            f"**Uplift policy** (randomised campaign, {u['n']:,} subscribers, average effect "
            f"{u['average_effect_per_1000']:.1f} churners prevented per 1,000 treated):",
            "",
            "| Policy | Qini | Best realised net value |",
            "|:--|--:|--:|",
        ]
        best = u.get("profit_best", {})
        names = {
            "Risk + sleeping-dog guard": "Risk × value + sleeping-dog guard",
            "Churn risk only": "Churn risk × value",
            "Uplift: X-learner": "Uplift × value",
        }
        for name, key in names.items():
            if name in u["rankings"] and key in best:
                lines.append(
                    f"| {name} | {u['rankings'][name]['qini_coefficient']:.2f} "
                    f"| {best[key]['best_net_value']:,.0f} {cfg['campaign']['currency']} |"
                )
    lines += ["", "## Drivers", ""]
    for fam, v in results["driver_importance"].items():
        if v >= 0.005:
            lines.append(f"- {FAMILY_LABELS.get(fam, fam).split(':')[0]}: {v:.2f}")
    if fair:
        lines += [
            "",
            "## Segments and fairness",
            "",
            f"Policy reach overall: {fair['overall_reach']:.1%} of churners. Contact rates are expected to differ "
            "between segments (they follow risk and value). Calibration gaps, weak within-segment ranking and "
            "low reach are flagged for review.",
            "",
            markdown_table(fair),
            "",
        ]
        if fair["flags"]:
            lines.append("**Flags for review:**")
            lines.append("")
            for f in fair["flags"]:
                lines.append(f"- {f['dimension']} = {f['segment']}: {'; '.join(f['issues'])}")
        else:
            lines.append("**No segment breached the review thresholds.**")
    lines += [
        "",
        "## Data and privacy",
        "",
        f"- Data contract status at training: {metadata.get('data_quality', 'n/a')}.",
        "- Subscriber identifiers are pseudonymised at ingestion (salted hash). No names, national ids or free text.",
        "- Drift reference (validation month) is stored with the model; PSI is computed on every scoring run.",
        "",
    ]
    return "\n".join(lines)

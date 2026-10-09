"""Configuration loading and calendar helpers."""

from __future__ import annotations

import calendar
import copy
import os
from datetime import date
from pathlib import Path
from typing import Any

import yaml

_REPO_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def _default_config() -> Path:
    """$CHURN_CONFIG, else the repo's configs/config.yaml, else ./configs/config.yaml (installed package)."""
    if env := os.getenv("CHURN_CONFIG"):
        return Path(env)
    return _REPO_CONFIG if _REPO_CONFIG.exists() else Path.cwd() / "configs" / "config.yaml"


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load the YAML config and apply nested overrides such as {"simulation": {"n_subscribers": 1000}}.

    A config may start with `extends: <other.yaml>` (relative to itself) to override only what differs.
    """
    path = Path(path or _default_config())
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    if base := cfg.pop("extends", None):
        cfg = _deep_merge(load_config(path.parent / base), cfg)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def parse_month(month: str | date) -> date:
    """'2025-10' -> date(2025, 10, 1)."""
    if isinstance(month, date):
        return month.replace(day=1)
    year, mon = month.split("-")[:2]
    return date(int(year), int(mon), 1)


def add_months(month: date, n: int) -> date:
    idx = month.year * 12 + (month.month - 1) + n
    return date(idx // 12, idx % 12 + 1, 1)


def month_end(month: date) -> date:
    return month.replace(day=calendar.monthrange(month.year, month.month)[1])


def month_label(month: date) -> str:
    return f"{month.year:04d}-{month.month:02d}"


def rolling_windows(as_of: str, n_train: int = 5, inactivity_months: int = 2) -> dict[str, object]:
    """Out-of-time windows for a monthly retrain run with data complete through `as_of`.

    The newest labelled month is the test month; validation is one label-window earlier; training
    uses the `n_train` months before validation. As of 2025-12: train Mar-Jul, valid Aug, test Oct.
    """
    last = parse_month(as_of)
    test = add_months(last, -inactivity_months)
    valid = add_months(test, -inactivity_months)
    train = [add_months(valid, -k) for k in range(n_train, 0, -1)]
    return {
        "train_cutoffs": [month_label(m) for m in train],
        "valid_cutoff": month_label(valid),
        "test_cutoff": month_label(test),
        "score_cutoff": month_label(last),
    }

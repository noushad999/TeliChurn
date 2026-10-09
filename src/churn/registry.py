"""File-based model registry with a champion/challenger gate, promotion history and rollback.

Layout (under `paths.registry_dir`):

    registry.json            champion pointer + full history of versions and decisions
    <version>/               model.txt, metadata.json, drift_reference.json, uplift_*.txt, metrics.json

Every training run registers a *challenger*. It becomes *champion* only if, on the challenger's own
out-of-time test month, it is not worse than the current champion by more than the configured
tolerance. Both models are scored on exactly the same rows, so the comparison is fair even when the
data has moved. Scoring and the API always resolve the champion; `rollback` restores the previous
one in one step.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


class Registry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "registry.json"

    # ------------------------------------------------------------------ index
    def _read(self) -> dict:
        if self.index_path.exists():
            return json.loads(self.index_path.read_text())
        return {"champion": None, "versions": []}

    def _write(self, index: dict) -> None:
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(index, indent=2))
        tmp.replace(self.index_path)  # atomic swap: readers never see a half-written index

    # --------------------------------------------------------------- versions
    def new_version(self) -> str:
        version = datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M%S")
        existing = {v["version"] for v in self._read()["versions"]}
        suffix, candidate = 1, version
        while candidate in existing or (self.root / candidate).exists():
            suffix += 1
            candidate = f"{version}-{suffix}"
        (self.root / candidate).mkdir(parents=True)
        return candidate

    def path(self, version: str) -> Path:
        return self.root / version

    def versions(self) -> list[dict]:
        return self._read()["versions"]

    def champion(self) -> str | None:
        return self._read()["champion"]

    def champion_dir(self) -> Path | None:
        c = self.champion()
        return self.path(c) if c else None

    def register(self, version: str, summary: dict) -> None:
        index = self._read()
        index["versions"].append(
            {
                "version": version,
                "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "status": "challenger",
                **summary,
            }
        )
        self._write(index)

    def promote(self, version: str, reason: str) -> None:
        index = self._read()
        if not any(v["version"] == version for v in index["versions"]):
            raise KeyError(f"Unknown version {version}")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for v in index["versions"]:
            if v["version"] == index["champion"]:
                v["status"] = "retired"
                v["retired_at"] = now
            if v["version"] == version:
                v.update(status="champion", promoted_at=now, decision=reason)
        index["previous_champion"] = index["champion"]
        index["champion"] = version
        self._write(index)

    def reject(self, version: str, reason: str) -> None:
        index = self._read()
        for v in index["versions"]:
            if v["version"] == version:
                v.update(status="rejected", decision=reason)
        self._write(index)

    def rollback(self) -> str:
        index = self._read()
        previous = index.get("previous_champion")
        if not previous:
            raise RuntimeError("No previous champion to roll back to")
        self.promote(previous, reason=f"rollback from {index['champion']}")
        return previous

    def prune(self, keep: int = 10) -> None:
        """Delete rejected/retired versions beyond the most recent `keep` (champion is never deleted)."""
        index = self._read()
        removable = [v for v in index["versions"] if v["status"] in ("rejected", "retired")]
        for v in removable[:-keep] if keep else removable:
            shutil.rmtree(self.path(v["version"]), ignore_errors=True)
            v["status"] = f"{v['status']} (pruned)"
        self._write(index)


def gate(challenger: dict, champion: dict | None, tolerance: float) -> tuple[bool, str]:
    """Promotion rule on the same test rows: PR-AUC and top-decile capture must not regress beyond tolerance."""
    if champion is None:
        return True, "first model: no champion yet"
    d_pr = challenger["pr_auc"] - champion["pr_auc"]
    d_cap = challenger["capture_top10"] - champion["capture_top10"]
    detail = f"PR-AUC {champion['pr_auc']:.4f} → {challenger['pr_auc']:.4f}, capture@10% {champion['capture_top10']:.4f} → {challenger['capture_top10']:.4f}"
    if d_pr >= -tolerance and d_cap >= -tolerance:
        return True, f"challenger not worse than champion on the same test rows ({detail})"
    return False, f"challenger regresses beyond tolerance {tolerance} ({detail})"


def export_champion(registry: Registry, export_dir: str | Path) -> None:
    """Copy the champion to a fixed deployment path (what the API container mounts), atomically."""
    src, dst = registry.champion_dir(), Path(export_dir)
    if src is None:
        return
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp)
    old = dst.with_name(dst.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if dst.exists():
        dst.rename(old)
    tmp.rename(dst)
    shutil.rmtree(old, ignore_errors=True)


def resolve_model_dir(cfg: dict) -> Path:
    """Champion directory from the registry, else the legacy `paths.model_dir`."""
    reg_dir = cfg["paths"].get("registry_dir")
    if reg_dir and (champ := Registry(reg_dir).champion_dir()) is not None:
        return champ
    return Path(cfg["paths"]["model_dir"])

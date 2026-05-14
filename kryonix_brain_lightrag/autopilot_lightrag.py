"""
autopilot_lightrag.py — LightRAG storage/index domain observer and analyzer for Kryonix Brain Safe Autopilot.
"""
from __future__ import annotations

import os
import time
from typing import Any
from kryonix_brain_lightrag import config

async def observe() -> dict[str, Any]:
    """Observe LightRAG storage paths, locks, and manifest integrity."""
    res: dict[str, Any] = {
        "domain": "lightrag",
        "status": "ok",
        "metrics": {
            "has_index_lock": False,
            "lock_age_seconds": 0,
            "manifest_exists": False,
        },
        "errors": [],
    }

    try:
        lock_file = config.INDEX_LOCK_FILE
        if lock_file.exists():
            res["metrics"]["has_index_lock"] = True
            try:
                res["metrics"]["lock_age_seconds"] = int(time.time() - lock_file.stat().st_mtime)
            except Exception:
                pass

        if config.INDEX_MANIFEST_FILE.exists():
            res["metrics"]["manifest_exists"] = True

    except Exception as exc:
        res["status"] = "error"
        res["errors"].append(str(exc))

    return res


async def diagnose(obs: dict[str, Any]) -> dict[str, Any]:
    """Diagnose LightRAG anomalies based on observations."""
    diag: dict[str, Any] = {
        "domain": "lightrag",
        "ok": True,
        "anomalies": [],
        "recommendations": [],
    }

    metrics = obs.get("metrics", {})
    if metrics.get("has_index_lock") and metrics.get("lock_age_seconds", 0) > 3600:
        diag["ok"] = False
        diag["anomalies"].append("stale_index_lock")
        diag["recommendations"].append("Remover arquivo de lock obsoleto do indexador.")

    return diag


async def propose(diag: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate proposals for LightRAG anomalies."""
    proposals: list[dict[str, Any]] = []

    if not diag["ok"]:
        for anomaly in diag.get("anomalies", []):
            if anomaly == "stale_index_lock":
                proposals.append({
                    "action_name": "clean_stale_lock",
                    "domain": "lightrag",
                    "risk_level": "low",
                    "description": "Remover trava de indexação (.index.lock) antiga/órfã.",
                    "proposed_actions": ["kryonix brain index --clean-state"], # or specific unlock
                    "rollback_actions": ["Nenhum"],
                    "requires_host": "any",
                })

    return proposals

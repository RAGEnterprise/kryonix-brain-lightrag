"""
autopilot_cag.py — CAG domain observer and analyzer for Kryonix Brain Safe Autopilot.
"""
from __future__ import annotations

import json
from typing import Any
from kryonix_brain_lightrag import config

async def observe() -> dict[str, Any]:
    """Observe CAG pack status and freshness."""
    res: dict[str, Any] = {
        "domain": "cag",
        "status": "ok",
        "metrics": {
            "total_files": 0,
            "total_bytes": 0,
            "built_at": "",
            "freshness": "UNKNOWN",
            "backend": "unknown",
        },
        "errors": [],
    }

    try:
        from kryonix_brain_lightrag import cag
        st = cag.status(config.CAG_DIR)
        res["metrics"]["total_files"] = st.get("total_files", 0)
        res["metrics"]["total_bytes"] = st.get("total_bytes", 0)
        res["metrics"]["built_at"] = st.get("built_at", "")
        res["metrics"]["freshness"] = st.get("freshness", "UNKNOWN")
        res["metrics"]["backend"] = st.get("backend", "unknown")

        if st.get("status") == "missing" or not st.get("ok"):
            res["metrics"]["freshness"] = "MISSING"

    except Exception as exc:
        res["status"] = "error"
        res["errors"].append(str(exc))

    return res


async def diagnose(obs: dict[str, Any]) -> dict[str, Any]:
    """Diagnose CAG anomalies based on observations."""
    diag: dict[str, Any] = {
        "domain": "cag",
        "ok": True,
        "anomalies": [],
        "recommendations": [],
    }

    freshness = obs.get("metrics", {}).get("freshness", "UNKNOWN")
    if freshness in ("MISSING", "STALE"):
        diag["ok"] = False
        diag["anomalies"].append(f"cag_pack_{freshness.lower()}")
        diag["recommendations"].append("Reconstruir pacote de contexto técnico CAG.")

    return diag


async def propose(diag: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate proposals for CAG anomalies."""
    proposals: list[dict[str, Any]] = []

    if not diag["ok"]:
        for anomaly in diag.get("anomalies", []):
            if anomaly in ("cag_pack_missing", "cag_pack_stale"):
                proposals.append({
                    "action_name": "rebuild_cag_pack",
                    "domain": "cag",
                    "risk_level": "low",
                    "description": "Construir/atualizar o pacote de contexto técnico acelerado (CAG).",
                    "proposed_actions": ["kryonix brain cag build"],
                    "rollback_actions": ["Restaurar diretório CAG anterior"],
                    "requires_host": "any",
                })

    return proposals

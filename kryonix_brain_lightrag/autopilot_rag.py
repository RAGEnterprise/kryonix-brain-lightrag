"""
autopilot_rag.py — RAG domain observer and analyzer for Kryonix Brain Safe Autopilot.
"""
from __future__ import annotations

import json
from typing import Any
from kryonix_brain_lightrag import config

async def observe() -> dict[str, Any]:
    """Observe RAG vector storage and stats."""
    res: dict[str, Any] = {
        "domain": "rag",
        "status": "ok",
        "metrics": {
            "entities": 0,
            "relations": 0,
            "failed_docs": 0,
            "skipped_docs": 0,
            "consistency_status": "UNKNOWN",
        },
        "errors": [],
    }

    try:
        from kryonix_brain_lightrag import rag
        stats = await rag.stats()
        res["metrics"]["entities"] = stats.get("entities", 0)
        res["metrics"]["relations"] = stats.get("relations", 0)
        res["metrics"]["consistency_status"] = stats.get("consistency_status", "UNKNOWN")

        # Check failed docs
        if config.FAILED_INDEX_FILE.exists():
            try:
                data = json.loads(config.FAILED_INDEX_FILE.read_text(encoding="utf-8"))
                res["metrics"]["failed_docs"] = len(data)
            except Exception:
                pass

        if config.SKIPPED_LARGE_FILES_FILE.exists():
            try:
                data = json.loads(config.SKIPPED_LARGE_FILES_FILE.read_text(encoding="utf-8"))
                res["metrics"]["skipped_docs"] = len(data)
            except Exception:
                pass

    except Exception as exc:
        res["status"] = "error"
        res["errors"].append(str(exc))

    return res


async def diagnose(obs: dict[str, Any]) -> dict[str, Any]:
    """Diagnose RAG anomalies based on observations."""
    diag: dict[str, Any] = {
        "domain": "rag",
        "ok": True,
        "anomalies": [],
        "recommendations": [],
    }

    metrics = obs.get("metrics", {})
    if metrics.get("consistency_status") != "OK" and metrics.get("consistency_status") != "UNKNOWN":
        diag["ok"] = False
        diag["anomalies"].append("vdb_graph_inconsistency")
        diag["recommendations"].append("Reconstruir vdb_entities a partir do grafo.")

    if metrics.get("failed_docs", 0) > 0:
        diag["ok"] = False
        diag["anomalies"].append("failed_index_docs_present")
        diag["recommendations"].append("Tentar reindexar documentos com falha.")

    return diag


async def propose(diag: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate proposals for RAG anomalies."""
    proposals: list[dict[str, Any]] = []

    if not diag["ok"]:
        for anomaly in diag.get("anomalies", []):
            if anomaly == "vdb_graph_inconsistency":
                proposals.append({
                    "action_name": "repair_vdb",
                    "domain": "rag",
                    "risk_level": "medium",
                    "description": "Reparar e sincronizar Vector DB com o Knowledge Graph principal.",
                    "proposed_actions": ["kryonix brain repair-vdb"],
                    "rollback_actions": ["Restaurar backup do vdb_entities.json"],
                    "requires_host": "any",
                })
            elif anomaly == "failed_index_docs_present":
                proposals.append({
                    "action_name": "retry_failed_index",
                    "domain": "rag",
                    "risk_level": "low",
                    "description": "Re-processar chunks de documentos que falharam na ingestão anterior.",
                    "proposed_actions": ["kryonix brain index --retry-failed"],
                    "rollback_actions": ["Nenhum"],
                    "requires_host": "any",
                })

    return proposals

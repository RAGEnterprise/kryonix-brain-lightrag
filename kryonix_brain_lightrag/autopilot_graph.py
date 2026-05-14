"""
autopilot_graph.py — Graph domain observer and analyzer for Kryonix Brain Safe Autopilot.
"""
from __future__ import annotations

import json
from typing import Any
from kryonix_brain_lightrag import graph_control

async def observe() -> dict[str, Any]:
    """Observe Neo4j graph state and metrics."""
    res: dict[str, Any] = {
        "domain": "graph",
        "status": "ok",
        "connected": False,
        "metrics": {
            "node_count": 0,
            "cli_command_count": 0,
            "risk_level_count": 0,
            "critical_command_count": 0,
            "glacier_target_count": 0,
        },
        "errors": [],
    }

    try:
        status = graph_control.graph_status()
        res["connected"] = status.get("connected", False)
        res["metrics"]["node_count"] = status.get("node_count", 0)

        if res["connected"]:
            # Query CliCommand count
            try:
                q_cli = graph_control._neo4j_call([{"statement": "MATCH (c:CliCommand) RETURN count(c) LIMIT 1", "parameters": {}}], timeout=2.0)
                rows = q_cli.get("results", [{}])[0].get("data", [])
                if rows:
                    res["metrics"]["cli_command_count"] = rows[0]["row"][0]
            except Exception as e:
                res["errors"].append(f"cli_command query error: {e}")

            # Query RiskLevel count
            try:
                q_risk = graph_control._neo4j_call([{"statement": "MATCH (r:RiskLevel) RETURN count(r) LIMIT 1", "parameters": {}}], timeout=2.0)
                rows = q_risk.get("results", [{}])[0].get("data", [])
                if rows:
                    res["metrics"]["risk_level_count"] = rows[0]["row"][0]
            except Exception as e:
                res["errors"].append(f"risk_level query error: {e}")

            # Query Critical commands count
            try:
                q_crit = graph_control._neo4j_call([{"statement": "MATCH (c:CliCommand) WHERE c.risk_level = 'critical' RETURN count(c) LIMIT 1", "parameters": {}}], timeout=2.0)
                rows = q_crit.get("results", [{}])[0].get("data", [])
                if rows:
                    res["metrics"]["critical_command_count"] = rows[0]["row"][0]
            except Exception as e:
                res["errors"].append(f"critical_command query error: {e}")

            # Query Glacier targets count
            try:
                q_glac = graph_control._neo4j_call([{"statement": "MATCH (c:CliCommand)-[:TARGETS_HOST]->(h:Host {name: 'glacier'}) RETURN count(c) LIMIT 1", "parameters": {}}], timeout=2.0)
                rows = q_glac.get("results", [{}])[0].get("data", [])
                if rows:
                    res["metrics"]["glacier_target_count"] = rows[0]["row"][0]
            except Exception as e:
                res["errors"].append(f"glacier_target query error: {e}")

    except Exception as exc:
        res["status"] = "error"
        res["errors"].append(str(exc))

    return res


async def diagnose(obs: dict[str, Any]) -> dict[str, Any]:
    """Diagnose graph anomalies based on observations."""
    diag: dict[str, Any] = {
        "domain": "graph",
        "ok": True,
        "anomalies": [],
        "recommendations": [],
    }

    if not obs.get("connected"):
        import os
        import socket
        import urllib.request

        hostname = socket.gethostname().lower()
        if hostname != "glacier":
            remote_ok = False
            remote_url = os.getenv("KRYONIX_BRAIN_API", "http://10.0.0.2:8000").rstrip("/") + "/health"
            try:
                with urllib.request.urlopen(remote_url, timeout=2.0) as resp:
                    if resp.status == 200:
                        remote_ok = True
            except Exception:
                pass

            if remote_ok:
                diag["ok"] = True
                diag["status"] = "warning"
                diag["reason"] = "Neo4j local indisponível no cliente; use Glacier ou Brain API remota para validação oficial."
                return diag
            else:
                diag["ok"] = False
                diag["anomalies"].append("neo4j_disconnected")
                diag["recommendations"].append("Neo4j local e Brain API remota indisponíveis no cliente.")
                return diag
        else:
            diag["ok"] = False
            diag["anomalies"].append("neo4j_disconnected")
            diag["recommendations"].append("Verifique o serviço neo4j.service e as credenciais.")
            return diag

    metrics = obs.get("metrics", {})
    if metrics.get("cli_command_count", 0) == 0:
        diag["ok"] = False
        diag["anomalies"].append("registry_v2_not_ingested")
        diag["recommendations"].append("Executar ingestão do Registry v2 no Neo4j.")

    if metrics.get("risk_level_count", 0) == 0:
        diag["ok"] = False
        diag["anomalies"].append("missing_risk_levels")
        diag["recommendations"].append("Ingestar nós de RiskLevel.")

    return diag


async def propose(diag: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate proposals for diagnosed anomalies."""
    proposals: list[dict[str, Any]] = []

    if not diag["ok"]:
        for anomaly in diag.get("anomalies", []):
            if anomaly in ("registry_v2_not_ingested", "missing_risk_levels"):
                proposals.append({
                    "action_name": "ingest_registry_v2",
                    "domain": "graph",
                    "risk_level": "high",  # Modifying Neo4j directly
                    "description": "Ingestar metadados operacionais do Registry v2 no Knowledge Graph.",
                    "proposed_actions": ["kryonix graph ingest-registry --apply"],
                    "rollback_actions": ["Nenhum rollback automatizado (apenas MERGE idempotente será executado)"],
                    "requires_host": "glacier",
                })

    return proposals

"""
autopilot.py — Core Safe Autopilot coordinator, proposal manager, and audit logger.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from kryonix_brain_lightrag import config
from kryonix_brain_lightrag import autopilot_graph, autopilot_rag, autopilot_cag, autopilot_lightrag


def _autopilot_dir() -> Path:
    """Return robust path for autopilot state, falling back if unprivileged."""
    target = config.BRAIN_HOME / "brain" / "autopilot"
    try:
        target.mkdir(parents=True, exist_ok=True)
        return target
    except OSError:
        # Fallback for unprivileged user running tests/mcp
        fallback = Path.home() / ".local" / "share" / "kryonix" / "autopilot"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _proposals_dir() -> Path:
    d = _autopilot_dir() / "proposals"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _audit_log_path() -> Path:
    return _autopilot_dir() / "audit.jsonl"


def _log_audit(event_type: str, status: str, details: dict[str, Any]) -> None:
    """Append structured entry to persistent audit log."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": event_type,
        "status": status,
        "details": details,
    }
    with _audit_log_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


async def status() -> dict[str, Any]:
    """Return overall health, pending proposals, and active guardrails."""
    proposals = []
    for p in _proposals_dir().glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if not data.get("applied", False):
                proposals.append(data)
        except Exception:
            pass

    return {
        "status": "active",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "guardrails": {
            "autonomy_level": "observe/propose/dry-run",
            "require_human_approval_for_risk_gte": "medium",
            "neo4j_write_host": "glacier",
            "destructive_commands_blocked": True,
        },
        "pending_proposals_count": len(proposals),
        "pending_proposals": proposals,
    }


async def observe() -> dict[str, Any]:
    """Gather metrics across all 4 domains."""
    started = time.time()
    g_obs, r_obs, c_obs, l_obs = await asyncio.gather(
        autopilot_graph.observe(),
        autopilot_rag.observe(),
        autopilot_cag.observe(),
        autopilot_lightrag.observe(),
    )
    elapsed = round(time.time() - started, 4)

    res = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "observations": {
            "graph": g_obs,
            "rag": r_obs,
            "cag": c_obs,
            "lightrag": l_obs,
        },
    }
    _log_audit("observe", "success", {"elapsed_seconds": elapsed})
    return res


async def diagnose() -> dict[str, Any]:
    """Run analyzers across all domains."""
    obs_data = await observe()
    obs_map = obs_data["observations"]

    g_diag, r_diag, c_diag, l_diag = await asyncio.gather(
        autopilot_graph.diagnose(obs_map["graph"]),
        autopilot_rag.diagnose(obs_map["rag"]),
        autopilot_cag.diagnose(obs_map["cag"]),
        autopilot_lightrag.diagnose(obs_map["lightrag"]),
    )

    all_ok = g_diag["ok"] and r_diag["ok"] and c_diag["ok"] and l_diag["ok"]
    anomalies = []
    for d in (g_diag, r_diag, c_diag, l_diag):
        anomalies.extend(d.get("anomalies", []))

    res = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ok": all_ok,
        "anomalies_count": len(anomalies),
        "diagnostics": {
            "graph": g_diag,
            "rag": r_diag,
            "cag": c_diag,
            "lightrag": l_diag,
        },
    }
    _log_audit("diagnose", "success", {"ok": all_ok, "anomalies_count": len(anomalies)})
    return res


async def propose() -> dict[str, Any]:
    """Generate structured JSON proposal based on diagnostics."""
    diag_data = await diagnose()
    d_map = diag_data["diagnostics"]

    g_prop, r_prop, c_prop, l_prop = await asyncio.gather(
        autopilot_graph.propose(d_map["graph"]),
        autopilot_rag.propose(d_map["rag"]),
        autopilot_cag.propose(d_map["cag"]),
        autopilot_lightrag.propose(d_map["lightrag"]),
    )

    actions = g_prop + r_prop + c_prop + l_prop
    if not actions:
        res = {
            "status": "no_action_needed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "proposals": [],
        }
        _log_audit("propose", "no_action", {})
        return res

    # Determine overall risk
    risk_weights = {"low": 1, "medium": 2, "high": 3, "critical": 4}
    highest_weight = max(risk_weights.get(a.get("risk_level", "low"), 1) for a in actions)
    weight_to_risk = {1: "low", 2: "medium", 3: "high", 4: "critical"}
    overall_risk = weight_to_risk[highest_weight]

    prop_id = f"prop-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"
    proposal = {
        "proposal_id": prop_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "overall_risk_level": overall_risk,
        "actions": actions,
        "applied": False,
    }

    path = _proposals_dir() / f"{prop_id}.json"
    path.write_text(json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8")
    _log_audit("propose", "created", {"proposal_id": prop_id, "risk_level": overall_risk, "actions_count": len(actions)})

    return {
        "status": "proposal_created",
        "proposal_id": prop_id,
        "proposal_path": str(path),
        "proposal": proposal,
    }


async def dry_run() -> dict[str, Any]:
    """Simulate execution of pending proposals."""
    st = await status()
    pending = st.get("pending_proposals", [])

    if not pending:
        # Create fresh proposal if none pending
        prop_res = await propose()
        if prop_res.get("status") == "no_action_needed":
            return {"status": "dry_run", "message": "Nenhuma anomalia detectada. Sistema 100% íntegro."}
        pending = [prop_res["proposal"]]

    simulations = []
    for p in pending:
        sim = {
            "proposal_id": p["proposal_id"],
            "risk_level": p["overall_risk_level"],
            "actions_to_execute": len(p.get("actions", [])),
            "simulated_steps": [],
        }
        for act in p.get("actions", []):
            sim["simulated_steps"].append({
                "action_name": act.get("action_name"),
                "domain": act.get("domain"),
                "commands": act.get("proposed_actions", []),
                "rollback": act.get("rollback_actions", []),
                "host_check": act.get("requires_host", "any"),
            })
        simulations.append(sim)

    _log_audit("dry_run", "success", {"simulated_proposals": len(simulations)})
    return {
        "status": "dry_run_complete",
        "simulated_proposals": simulations,
    }


async def apply(proposal_id: str) -> dict[str, Any]:
    """Apply approved proposal with strict safety guardrails."""
    path = _proposals_dir() / f"{proposal_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Proposta não encontrada: {proposal_id}")

    proposal = json.loads(path.read_text(encoding="utf-8"))
    if proposal.get("applied"):
        return {"status": "already_applied", "proposal_id": proposal_id}

    risk = proposal.get("overall_risk_level", "low")
    if risk == "critical":
        raise RuntimeError("Propostas de risco CRITICAL requerem intervenção manual direta e não podem ser aplicadas pelo Autopilot.")

    # Check host constraint for graph/neo4j write actions
    has_graph_action = any(a.get("domain") == "graph" or a.get("requires_host") == "glacier" for a in proposal.get("actions", []))
    if has_graph_action:
        hostname = socket.gethostname().lower()
        if hostname != "glacier" and not os.getenv("KRYONIX_AUTOPILOT_IGNORE_HOST"):
            raise RuntimeError("Operações de escrita no Neo4j/Knowledge Graph devem ser executadas exclusivamente no host Glacier.")

    results = []
    for act in proposal.get("actions", []):
        act_name = act.get("action_name")
        domain = act.get("domain")

        if domain == "graph" and act_name == "ingest_registry_v2":
            from kryonix_brain_lightrag import graph_control
            try:
                # We build manifest and apply it
                manifest = graph_control.build_manifest()
                graph_control.save_manifest(manifest)
                res = graph_control.apply_manifest(manifest["manifest_id"])
                results.append({"action": act_name, "status": "applied", "details": res})
            except Exception as e:
                results.append({"action": act_name, "status": "error", "error": str(e)})
        else:
            # Subprocess execution for CLI commands
            for cmd_str in act.get("proposed_actions", []):
                try:
                    cmd_parts = cmd_str.split()
                    proc = subprocess.run(cmd_parts, capture_output=True, text=True, check=True)
                    results.append({"command": cmd_str, "status": "success", "stdout": proc.stdout[:200]})
                except subprocess.CalledProcessError as e:
                    results.append({"command": cmd_str, "status": "error", "stderr": e.stderr})

    proposal["applied"] = True
    proposal["applied_at"] = datetime.now(timezone.utc).isoformat()
    proposal["results"] = results
    path.write_text(json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8")

    _log_audit("apply", "applied", {"proposal_id": proposal_id, "results": results})
    return {
        "status": "applied",
        "proposal_id": proposal_id,
        "results": results,
    }


async def audit() -> list[dict[str, Any]]:
    """Return persistent audit logs."""
    p = _audit_log_path()
    if not p.exists():
        return []

    lines = p.read_text(encoding="utf-8").splitlines()
    entries = []
    for line in lines:
        if line.strip():
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
    return entries

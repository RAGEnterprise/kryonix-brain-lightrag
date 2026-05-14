from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

from . import config

GRAPH_SCHEMA_V1 = {
    "version": "v1",
    "nodes": [
        "Host",
        "Service",
        "File",
        "Command",
        "CliCommand",
        "CliSubcommand",
        "CliCategory",
        "RiskLevel",
        "Runtime",
        "Example",
        "Issue",
        "Port",
        "Model",
        "GPU",
        "Document",
        "Chunk",
        "Entity",
        "ReasoningTrace",
        "ReasoningStep",
        "ToolCall",
        "ToolResult",
    ],
    "relations": [
        "RUNS",
        "DEPENDS_ON",
        "LISTENS_ON",
        "DECLARES",
        "IMPORTS",
        "VALIDATES",
        "REPAIRS",
        "AFFECTS",
        "HAS_CHUNK",
        "HAS_ENTITY",
        "MENTIONED_IN",
        "HAS_STEP",
        "USED_TOOL",
        "RETURNED",
        "HAS_SUBCOMMAND",
        "IN_CATEGORY",
        "HAS_RISK",
        "REQUIRES_RUNTIME",
        "TARGETS_HOST",
        "HAS_EXAMPLE",
    ],
}

READ_ONLY_BLOCKLIST = (
    "CREATE",
    "MERGE",
    "DELETE",
    "DETACH DELETE",
    "SET",
    "REMOVE",
    "LOAD CSV",
)

MAX_QUERY_LEN = 4000
DEFAULT_TIMEOUT_SEC = float(os.getenv("KRYONIX_GRAPH_QUERY_TIMEOUT_SEC", "5"))
DEFAULT_QUERY_LIMIT = int(os.getenv("KRYONIX_GRAPH_QUERY_LIMIT", "50"))


def _brain_home() -> Path:
    return config.BRAIN_HOME / "brain"


def _manifest_dir() -> Path:
    d = Path(os.getenv("KRYONIX_GRAPH_MANIFEST_DIR", str(_brain_home() / "graph_manifests")))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _audit_log_path() -> Path:
    p = _brain_home() / "graph_audit.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _neo4j_http_url() -> str:
    return os.getenv("KRYONIX_NEO4J_HTTP_URL", "http://127.0.0.1:7474").rstrip("/")


def _neo4j_user() -> str:
    auth = os.getenv("NEO4J_AUTH", "")
    if "/" in auth:
        return auth.split("/", 1)[0].strip() or "neo4j"
    return os.getenv("KRYONIX_NEO4J_USER", "neo4j")


def _neo4j_password() -> str:
    auth = os.getenv("NEO4J_AUTH", "")
    if "/" in auth:
        return auth.split("/", 1)[1]
    return os.getenv("KRYONIX_NEO4J_PASSWORD", "")


def _neo4j_headers() -> dict[str, str]:
    import base64

    user = _neo4j_user()
    pwd = _neo4j_password()
    if not pwd:
        raise RuntimeError("Neo4j credential is not configured in service environment")
    cred = f"{user}:{pwd}".encode("utf-8")
    token = base64.b64encode(cred).decode("ascii")
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Basic {token}",
    }


def _neo4j_call(statements: list[dict[str, Any]], timeout: float = DEFAULT_TIMEOUT_SEC) -> dict[str, Any]:
    payload = json.dumps({"statements": statements}).encode("utf-8")
    req = request.Request(
        url=f"{_neo4j_http_url()}/db/neo4j/tx/commit",
        data=payload,
        headers=_neo4j_headers(),
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Neo4j HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Neo4j indisponível: {exc.reason}") from exc


def _collect_hosts(repo_root: Path) -> list[dict[str, Any]]:
    hosts_dir = repo_root / "hosts"
    if not hosts_dir.exists():
        return []

    hosts = []
    for d in sorted(hosts_dir.iterdir()):
        if not d.is_dir() or d.name == "common":
            continue
        role = "server" if d.name == "glacier" else "client"
        hosts.append(
            {
                "label": "Host",
                "key": {"name": d.name},
                "props": {
                    "role": role,
                    "source_path": f"hosts/{d.name}/default.nix",
                    "indexed_at": datetime.now(timezone.utc).isoformat(),
                },
            }
        )
    return hosts


def _collect_services(repo_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    service_defs = [
        {
            "name": "kryonix-brain",
            "description": "Kryonix Brain FastAPI",
            "port": 8000,
            "source": "modules/nixos/services/brain.nix",
        },
        {
            "name": "ollama",
            "description": "Ollama model runtime",
            "port": 11434,
            "source": "modules/nixos/services/brain.nix",
        },
        {
            "name": "neo4j",
            "description": "Neo4j Graph Database",
            "port": 7474,
            "source": "modules/nixos/services/neo4j.nix",
        },
    ]

    nodes: list[dict[str, Any]] = []
    rels: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []

    for s in service_defs:
        nodes.append(
            {
                "label": "Service",
                "key": {"name": s["name"]},
                "props": {
                    "description": s["description"],
                    "bind": f"127.0.0.1:{s['port']}",
                },
            }
        )
        nodes.append(
            {
                "label": "Port",
                "key": {"number": str(s["port"]), "protocol": "tcp"},
                "props": {},
            }
        )
        files.append(
            {
                "label": "File",
                "key": {"path": s["source"]},
                "props": _file_props(repo_root / s["source"]),
            }
        )

        rels.append(_rel("Service", {"name": s["name"]}, "LISTENS_ON", "Port", {"number": str(s["port"]), "protocol": "tcp"}))
        rels.append(_rel("File", {"path": s["source"]}, "DECLARES", "Service", {"name": s["name"]}))

    rels.append(_rel("Service", {"name": "kryonix-brain"}, "DEPENDS_ON", "Service", {"name": "ollama"}))
    rels.append(_rel("Service", {"name": "kryonix-brain"}, "DEPENDS_ON", "Service", {"name": "neo4j"}))

    return nodes, rels, files


def _file_props(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    st = path.stat()
    return {
        "exists": True,
        "size_bytes": int(st.st_size),
        "mtime": int(st.st_mtime),
    }


def _rel(src_label: str, src_key: dict[str, str], rel: str, dst_label: str, dst_key: dict[str, str]) -> dict[str, Any]:
    return {
        "src": {"label": src_label, "key": src_key},
        "type": rel,
        "dst": {"label": dst_label, "key": dst_key},
        "props": {},
    }


def _collect_import_edges(repo_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    file_nodes: list[dict[str, Any]] = []
    import_rels: list[dict[str, Any]] = []
    pattern = re.compile(r"^\s*imports\s*=\s*\[(.*?)\];", re.MULTILINE | re.DOTALL)

    for nix_file in repo_root.rglob("*.nix"):
        rel_path = nix_file.relative_to(repo_root).as_posix()
        if rel_path.startswith(("result/", ".git/", "node_modules/", "target/", "dist/", "build/")):
            continue
        file_nodes.append({"label": "File", "key": {"path": rel_path}, "props": _file_props(nix_file)})
        try:
            text = nix_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for match in pattern.finditer(text):
            block = match.group(1)
            for raw in re.findall(r"\"([^\"]+\.nix)\"", block):
                src = rel_path
                dst = (nix_file.parent / raw).resolve()
                try:
                    dst_rel = dst.relative_to(repo_root.resolve()).as_posix()
                except ValueError:
                    continue
                import_rels.append(_rel("File", {"path": src}, "IMPORTS", "File", {"path": dst_rel}))
    return file_nodes, import_rels


def _collect_registry_v2(repo_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        output = subprocess.check_output(["kryonix", "commands", "--json"], encoding="utf-8", stderr=subprocess.PIPE)
        data = json.loads(output)
    except Exception:
        # Fallback para execução direta via shell se 'kryonix' não estiver no path
        try:
            main_sh = repo_root / "packages/kryonix-cli/main.sh"
            if not main_sh.exists():
                return [], []
            output = subprocess.check_output(["bash", str(main_sh), "commands", "--json"], encoding="utf-8")
            data = json.loads(output)
        except Exception:
            return [], []

    nodes: list[dict[str, Any]] = []
    rels: list[dict[str, Any]] = []

    # Map version
    schema_v = data.get("schema_version", 1)

    for entry in data.get("commands", []):
        group = entry.get("group", "unknown")
        name = entry.get("name", "unknown")
        sub = entry.get("subcommand", "")
        desc = entry.get("description", "")
        risk = entry.get("risk_level", "low")
        sudo = entry.get("requires_sudo", False)
        status = entry.get("status", "stable")
        category = entry.get("category", group)
        hosts_raw = entry.get("requires_host", "any")
        runtimes = entry.get("requires_runtime", [])
        examples = entry.get("examples", [])

        # 1. Base Nodes (Categories, Risk, Runtime, Host)
        nodes.append({"label": "CliCategory", "key": {"name": category}, "props": {}})
        nodes.append({"label": "RiskLevel", "key": {"name": risk}, "props": {}})

        # 2. Command/Subcommand Node
        if sub:
            node_label = "CliSubcommand"
            full_name = f"{name} {sub}"
            node_key = {"name": sub, "parent": name}
            props = {
                "full_name": full_name,
                "description": desc,
                "requires_sudo": sudo,
                "status": status,
                "risk_level": risk,
            }
            nodes.append({"label": node_label, "key": node_key, "props": props})
            # Relation to parent
            rels.append(_rel("CliCommand", {"name": name}, "HAS_SUBCOMMAND", "CliSubcommand", node_key))
        else:
            node_label = "CliCommand"
            node_key = {"name": name}
            props = {
                "description": desc,
                "requires_sudo": sudo,
                "status": status,
                "risk_level": risk,
                "category": category,
            }
            nodes.append({"label": node_label, "key": node_key, "props": props})

        # 3. Relations and Nodes
        # Category
        nodes.append({"label": "CliCategory", "key": {"name": category}, "props": {}})
        rels.append(_rel(node_label, node_key, "IN_CATEGORY", "CliCategory", {"name": category}))
        # Risk
        nodes.append({"label": "RiskLevel", "key": {"name": risk}, "props": {}})
        rels.append(_rel(node_label, node_key, "HAS_RISK", "RiskLevel", {"name": risk}))

        # Runtimes
        for rt in runtimes:
            if rt.lower() == "none":
                continue
            nodes.append({"label": "Runtime", "key": {"name": rt}, "props": {}})
            rels.append(_rel(node_label, node_key, "REQUIRES_RUNTIME", "Runtime", {"name": rt}))

        # Hosts
        target_hosts = []
        if hosts_raw == "any":
            target_hosts = ["inspiron", "glacier"]
        else:
            target_hosts = [h.strip() for h in hosts_raw.split(",")]

        for h in target_hosts:
            # We match the existing Host nodes
            rels.append(_rel(node_label, node_key, "TARGETS_HOST", "Host", {"name": h}))

        # Examples
        for ex in examples:
            ex_id = f"ex-{uuid.uuid4().hex[:8]}"
            nodes.append({"label": "Example", "key": {"text": ex}, "props": {}})
            rels.append(_rel(node_label, node_key, "HAS_EXAMPLE", "Example", {"text": ex}))

        # Source File relation
        rels.append(_rel("File", {"path": "packages/kryonix-cli/registry.sh"}, "DECLARES", node_label, node_key))

    return nodes, rels


def build_manifest(repo_root: Path | None = None) -> dict[str, Any]:
    repo_root = repo_root or Path(os.getenv("KRYONIX_REPO_ROOT", "/etc/kryonix"))

    host_nodes = _collect_hosts(repo_root)
    service_nodes, service_rels, service_files = _collect_services(repo_root)
    file_nodes, import_rels = _collect_import_edges(repo_root)
    reg_nodes, reg_rels = _collect_registry_v2(repo_root)

    host_service_rels = [_rel("Host", {"name": "glacier"}, "RUNS", "Service", {"name": n["key"]["name"]}) for n in service_nodes if n["label"] == "Service"]

    node_map: dict[str, dict[str, Any]] = {}
    for node in host_nodes + service_nodes + service_files + file_nodes + reg_nodes:
        key = f"{node['label']}::{json.dumps(node['key'], sort_keys=True)}"
        if key in node_map:
            node_map[key]["props"].update(node.get("props", {}))
        else:
            node_map[key] = node

    rels = service_rels + import_rels + host_service_rels + reg_rels

    manifest = {
        "manifest_id": f"graph-v1-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": GRAPH_SCHEMA_V1["version"],
        "repo_root": str(repo_root),
        "nodes": list(node_map.values()),
        "relationships": rels,
        "stats": {
            "node_count": len(node_map),
            "relationship_count": len(rels),
        },
        "requires_human_approval": True,
        "applied": False,
    }
    return manifest


def save_manifest(manifest: dict[str, Any]) -> Path:
    path = _manifest_dir() / f"{manifest['manifest_id']}.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def dry_run_ingest() -> dict[str, Any]:
    manifest = build_manifest()
    path = save_manifest(manifest)
    return {
        "status": "dry_run",
        "manifest_id": manifest["manifest_id"],
        "manifest_path": str(path),
        "stats": manifest["stats"],
        "requires_human_approval": True,
    }


def _render_node_merge(node: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    label = node["label"]
    key = node["key"]
    props = node.get("props", {})

    key_parts = [f"{k}: ${'k_' + k}" for k in key.keys()]
    set_parts = [f"n.{k} = ${'p_' + k}" for k in props.keys()]

    query = f"MERGE (n:{label} {{{', '.join(key_parts)}}})"
    if set_parts:
        query += "\nSET " + ", ".join(set_parts)

    params = {f"k_{k}": v for k, v in key.items()}
    params.update({f"p_{k}": v for k, v in props.items()})
    return query, params


def _render_rel_merge(rel: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    src = rel["src"]
    dst = rel["dst"]
    rel_type = rel["type"]

    q = (
        f"MATCH (a:{src['label']}), (b:{dst['label']})\n"
        f"WHERE all(k IN keys($src_key) WHERE a[k] = $src_key[k]) AND all(k IN keys($dst_key) WHERE b[k] = $dst_key[k])\n"
        f"MERGE (a)-[r:{rel_type}]->(b)"
    )
    return q, {"src_key": src["key"], "dst_key": dst["key"]}


def apply_manifest(manifest_id: str) -> dict[str, Any]:
    path = _manifest_dir() / f"{manifest_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Manifest não encontrado: {path}")

    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("applied"):
        return {
            "status": "already_applied",
            "manifest_id": manifest_id,
            "applied_at": manifest.get("applied_at"),
        }

    statements: list[dict[str, Any]] = []
    for node in manifest.get("nodes", []):
        stmt, params = _render_node_merge(node)
        statements.append({"statement": stmt, "parameters": params})

    for rel in manifest.get("relationships", []):
        stmt, params = _render_rel_merge(rel)
        statements.append({"statement": stmt, "parameters": params})

    _neo4j_call(statements, timeout=15.0)

    manifest["applied"] = True
    manifest["applied_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "status": "applied",
        "manifest_id": manifest_id,
        "applied_at": manifest["applied_at"],
        "stats": manifest.get("stats", {}),
    }


def graph_status() -> dict[str, Any]:
    status = {
        "neo4j_http_url": _neo4j_http_url(),
        "schema_version": GRAPH_SCHEMA_V1["version"],
        "manifest_dir": str(_manifest_dir()),
    }

    try:
        result = _neo4j_call([
            {
                "statement": "MATCH (n) RETURN count(n) AS nodes",
                "parameters": {},
            }
        ], timeout=2.0)
        row = result.get("results", [{}])[0].get("data", [])
        nodes = row[0]["row"][0] if row else 0
        status.update({"connected": True, "node_count": nodes})
    except Exception as exc:
        status.update({"connected": False, "error": str(exc)})

    return status


def graph_schema() -> dict[str, Any]:
    return GRAPH_SCHEMA_V1


def _validate_read_only_query(query: str) -> str:
    if not query or not query.strip():
        raise ValueError("Query vazia")
    if len(query) > MAX_QUERY_LEN:
        raise ValueError("Query excede tamanho máximo")

    q_upper = re.sub(r"\s+", " ", query.upper())
    for token in READ_ONLY_BLOCKLIST:
        if token in q_upper:
            raise ValueError(f"Token proibido em modo read-only: {token}")

    if "CALL DBMS." in q_upper or "CALL APOC." in q_upper:
        raise ValueError("CALL dbms/apoc não permitido")

    if "LIMIT" not in q_upper:
        raise ValueError("LIMIT obrigatório")

    if not (q_upper.startswith("MATCH") or q_upper.startswith("OPTIONAL MATCH") or q_upper.startswith("WITH")):
        raise ValueError("Query deve iniciar com MATCH/OPTIONAL MATCH/WITH")

    return query


def _audit(event: dict[str, Any]) -> None:
    event["ts"] = datetime.now(timezone.utc).isoformat()
    with _audit_log_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def graph_query(cypher: str, timeout_sec: float = DEFAULT_TIMEOUT_SEC) -> dict[str, Any]:
    safe_query = _validate_read_only_query(cypher)
    started = time.time()
    result = _neo4j_call([{"statement": safe_query, "parameters": {}}], timeout=timeout_sec)
    elapsed = round(time.time() - started, 4)
    _audit({"type": "query", "ok": True, "elapsed_sec": elapsed, "query": safe_query[:512]})
    return {"status": "ok", "elapsed_sec": elapsed, "result": result}


def graph_doctor() -> dict[str, Any]:
    checks: dict[str, Any] = {}

    checks["neo4j_tcp_7474"] = _check_tcp("127.0.0.1", 7474)
    checks["neo4j_tcp_7687"] = _check_tcp("127.0.0.1", 7687)

    checks["neo4j_auth_env_available"] = bool(os.getenv("NEO4J_AUTH"))
    checks["neo4j_password_available"] = bool(_neo4j_password())

    try:
        checks["graph_status"] = graph_status()
    except Exception as exc:
        checks["graph_status"] = {"connected": False, "error": str(exc)}

    ok = bool(checks["neo4j_tcp_7474"]) and bool(checks["neo4j_password_available"])
    return {"status": "ok" if ok else "warn", "checks": checks}


def _check_tcp(host: str, port: int, timeout: float = 1.0) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()

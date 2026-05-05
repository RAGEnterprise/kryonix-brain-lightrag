"""
cag.py — CAG (Context-Augmented Generation) Python wrapper.

Calls the Rust binary `kryonix-cag` when available.
Falls back to a pure-Python implementation for environments where
the binary has not been compiled yet.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from kryonix_brain_lightrag.config import CAG_DIR

# ── Constants ──────────────────────────────────────────────────────────────────

CAG_BINARY_NAME = "kryonix-cag"
DEFAULT_CAG_DIR = CAG_DIR
DEFAULT_REPO = Path(os.environ.get("KRYONIX_REPO", "/etc/kryonix"))
MANIFEST_FILENAME = "manifest.json"

# ── Rust binary resolution ─────────────────────────────────────────────────────

def _find_rust_binary() -> Optional[Path]:
    """Search for the compiled kryonix-cag binary in standard locations."""
    # 1. Sibling cargo release output
    pkg_root = Path(__file__).parent.parent
    candidates = [
        pkg_root / "crates" / "kryonix-cag" / "target" / "release" / CAG_BINARY_NAME,
        pkg_root / "target" / "release" / CAG_BINARY_NAME,
        # System PATH
    ]
    for c in candidates:
        if c.exists() and os.access(c, os.X_OK):
            return c
    # Fall back to PATH
    found = shutil.which(CAG_BINARY_NAME)
    if found:
        return Path(found)
    return None


RUST_BINARY: Optional[Path] = _find_rust_binary()


def _run_rust(args: list[str]) -> dict[str, Any]:
    """Run the Rust binary and return parsed JSON stdout."""
    if RUST_BINARY is None:
        raise RuntimeError(
            f"kryonix-cag binary not found. Build with: "
            f"cargo build --release --manifest-path "
            f"<repo>/packages/kryonix-brain-lightrag/crates/kryonix-cag/Cargo.toml"
        )
    cmd = [str(RUST_BINARY)] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"kryonix-cag failed (exit {result.returncode}):\n{result.stderr}"
        )
    return json.loads(result.stdout) if result.stdout.strip() else {}


# ── Pure-Python fallback ───────────────────────────────────────────────────────

# Secret patterns for filtering (mirrors Rust security.rs)
import re as _re

_SECRET_PATTERNS = [
    _re.compile(r"ntn_[A-Za-z0-9]{20,}"),
    _re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    _re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    _re.compile(r"sk-[A-Za-z0-9]{20,}"),
    # PEM blocks — separate patterns per key type
    _re.compile(r"-----BEGIN RSA PRIVATE KEY-----"),
    _re.compile(r"-----BEGIN OPENSSH PRIVATE KEY-----"),
    _re.compile(r"-----BEGIN PRIVATE KEY-----"),
    _re.compile(r"-----BEGIN EC PRIVATE KEY-----"),
    # Common env var names
    _re.compile(r"OPENAI_API_KEY\s*=\s*\S+"),
    _re.compile(r"ANTHROPIC_API_KEY\s*=\s*\S+"),
    _re.compile(r"NOTION_TOKEN\s*=\s*\S+"),
    # Tailscale auth keys
    _re.compile(r"tskey-[A-Za-z0-9\-]{20,}"),
    # Generic secrets
    _re.compile(r"password\s*=\s*['\"]?\S{8,}['\"]?", _re.IGNORECASE),
    _re.compile(r"secret\s*=\s*['\"]?\S{8,}['\"]?", _re.IGNORECASE),
]


def _line_has_secret(line: str) -> bool:
    return any(p.search(line) for p in _SECRET_PATTERNS)


def _filter_content(text: str) -> tuple[str, list[int]]:
    """Return (filtered_text, [redacted_line_numbers])."""
    lines = text.splitlines()
    filtered = []
    redacted = []
    for i, line in enumerate(lines, 1):
        if _line_has_secret(line):
            filtered.append(f"[REDACTED:line:{i}]")
            redacted.append(i)
        else:
            filtered.append(line)
    return "\n".join(filtered), redacted


_INCLUDE_SUFFIXES = {".nix", ".md", ".toml", ".json", ".py", ".sh", ".txt"}
_EXCLUDE_DIRS = {
    ".git", "result", "node_modules", ".direnv", ".venv",
    "target", "__pycache__", ".obsidian", "themes",
}

_KEYWORD_TAGS: dict[str, list[str]] = {
    "glacier": ["glacier", "host-config"],
    "inspiron": ["inspiron", "host-config"],
    "brain": ["brain", "lightrag"],
    "rag": ["lightrag", "brain"],
    "lightrag": ["lightrag", "brain"],
    "ollama": ["ollama", "glacier"],
    "nvidia": ["gpu", "glacier"],
    "gpu": ["gpu"],
    "cuda": ["gpu"],
    "mcp": ["mcp"],
    "tailscale": ["tailscale", "networking"],
    "nix": ["nix"],
    "nixos": ["nix", "host-config"],
    "flake": ["flake", "nix"],
    "audio": ["audio"],
    "bluetooth": ["bluetooth"],
    "gaming": ["gaming"],
    "hyprland": ["desktop"],
    "desktop": ["desktop"],
    "ssh": ["ssh"],
    "storage": ["storage"],
    "btrfs": ["storage"],
    "vault": ["brain", "docs"],
    "obsidian": ["brain", "docs"],
    "bancos": ["local-sources", "nixos-sources"],
    "fontes": ["local-sources", "nixos-sources"],
    "locais": ["local-sources"],
    "local": ["local-sources"],
    "sources": ["local-sources", "nixos-sources"],
}


def _derive_tags(path: str) -> list[str]:
    p = path.lower()
    tags: set[str] = set()
    if "hosts/glacier" in p:
        tags |= {"glacier", "host-config"}
    if "hosts/inspiron" in p:
        tags |= {"inspiron", "host-config"}
    if "hosts/" in p:
        tags.add("host-config")
    if "modules/" in p:
        tags.add("nixos-module")
    if "profiles/" in p:
        tags.add("profile")
    if "home/" in p:
        tags.add("home-manager")
    if "desktop/" in p or "hyprland" in p:
        tags.add("desktop")
    if "packages/" in p:
        tags.add("package")
    if "nvidia" in p or "gpu" in p:
        tags.add("gpu")
    if "ollama" in p:
        tags.add("ollama")
    if "brain" in p:
        tags.add("brain")
    if "lightrag" in p:
        tags.add("lightrag")
    if "mcp" in p:
        tags.add("mcp")
    if "tailscale" in p:
        tags.add("tailscale")
    if "audio" in p or "pipewire" in p:
        tags.add("audio")
    if "bluetooth" in p:
        tags.add("bluetooth")
    if "gaming" in p or "steam" in p:
        tags.add("gaming")
    if "ssh" in p:
        tags.add("ssh")
    if "storage" in p or "btrfs" in p:
        tags.add("storage")
    if p.endswith(".nix"):
        tags.add("nix")
    if p.endswith(".md"):
        tags.add("docs")
    if "flake.nix" in p:
        tags.add("flake")
    if "agents" in p:
        tags.add("agent")
    if "nixos-local" in p or "local-sources" in p:
        tags |= {"local-sources", "nixos-sources"}
    return sorted(tags)


def _scan_repo_python(repo: Path, max_files: int = 1000) -> list[dict]:
    """Pure-Python file scanner — fallback when Rust binary unavailable."""
    results = []
    for root, dirs, files in os.walk(repo):
        # Prune excluded dirs in-place
        dirs[:] = [d for d in dirs if d not in _EXCLUDE_DIRS and not d.startswith(".")]
        for fname in files:
            path = Path(root) / fname
            if path.suffix not in _INCLUDE_SUFFIXES:
                continue
            try:
                rel = str(path.relative_to(repo))
                stat = path.stat()
                if stat.st_size > 256 * 1024:
                    continue
                raw = path.read_bytes()
                if b"\x00" in raw:
                    continue
                text = raw.decode("utf-8", errors="replace")
                content, redacted = _filter_content(text)
                import hashlib
                h = hashlib.blake2b(raw, digest_size=32).hexdigest()
                results.append({
                    "path": rel,
                    "size_bytes": stat.st_size,
                    "blake3": h,
                    "content": content,
                    "tags": _derive_tags(rel),
                })
            except Exception:
                continue
            if len(results) >= max_files:
                break
        if len(results) >= max_files:
            break
    results.sort(key=lambda x: x["path"])
    return results


def _build_manifest_python(profile: str, repo: Path, out: Path) -> dict:
    """Build a CAG manifest using the pure-Python scanner."""
    import hashlib

    files = _scan_repo_python(repo)

    # Build tag index
    tags: dict[str, list[str]] = {}
    for f in files:
        for tag in f["tags"]:
            tags.setdefault(tag, []).append(f["path"])

    # Content hash
    h = hashlib.blake2b(digest_size=32)
    for f in files:
        h.update(f["blake3"].encode())
    content_hash = h.hexdigest()

    manifest = {
        "version": 1,
        "profile": profile,
        "repo_root": str(repo),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "total_files": len(files),
        "total_bytes": sum(f["size_bytes"] for f in files),
        "content_hash": content_hash,
        "files": files,
        "tags": tags,
    }

    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def _confidence_label(score: float) -> str:
    if score >= 0.75:
        return "Alta"
    if score >= 0.45:
        return "Média"
    return "Baixa"


# ── Public API ─────────────────────────────────────────────────────────────────

def build(
    profile: str = "kryonix-core",
    repo: Path = DEFAULT_REPO,
    out: Path = DEFAULT_CAG_DIR,
) -> dict:
    """
    Build a CAG pack.
    Uses Rust binary when available, falls back to Python implementation.
    """
    if RUST_BINARY is not None:
        return _run_rust(["build", "--profile", profile, "--repo", str(repo), "--out", str(out)])
    else:
        manifest = _build_manifest_python(profile, repo, out)
        return {
            "version": manifest["version"],
            "profile": manifest["profile"],
            "repo_root": manifest["repo_root"],
            "built_at": manifest["built_at"],
            "total_files": manifest["total_files"],
            "total_bytes": manifest["total_bytes"],
            "content_hash": manifest["content_hash"][:16],
            "tag_count": len(manifest["tags"]),
            "backend": "python-fallback",
        }


def status(cag_dir: Path = DEFAULT_CAG_DIR) -> dict:
    """Return the summary of an existing CAG pack."""
    if RUST_BINARY is not None:
        return _run_rust(["status", "--dir", str(cag_dir)])
    # Python fallback
    manifest_path = cag_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest found at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    return {
        "version": manifest.get("version"),
        "profile": manifest.get("profile"),
        "repo_root": manifest.get("repo_root"),
        "built_at": manifest.get("built_at"),
        "total_files": manifest.get("total_files"),
        "total_bytes": manifest.get("total_bytes"),
        "content_hash": manifest.get("content_hash", "")[:16],
        "tag_count": len(manifest.get("tags", {})),
        "backend": "python-fallback",
    }


def route(query: str, cag_dir: Path = DEFAULT_CAG_DIR, top_k: int = 10) -> dict:
    """Route a query to the most relevant files in the CAG pack."""
    if RUST_BINARY is not None:
        return _run_rust(["route", query, "--dir", str(cag_dir), "--top-k", str(top_k), "--format", "json"])

    # Python fallback — delegate to routing.py
    from kryonix_brain_lightrag.routing import route_query_python
    manifest_path = cag_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest found at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    fallback = route_query_python(manifest, query, top_k)
    suggested = fallback.get("suggested_strategy", {})
    score = float(suggested.get("score", suggested.get("confidence", 0.2)))
    strategy = str(suggested.get("strategy", "hybrid")).lower()
    return {
        "query": fallback.get("query", query),
        "profile": manifest.get("profile", "unknown"),
        "strategy": strategy,
        "confidence": score,
        "confidence_label": _confidence_label(score),
        "reason": suggested.get("reason", ""),
        "matched_tags": fallback.get("matched_tags", []),
        "matched_files": fallback.get("matched_files", []),
        "total_tokens_est": fallback.get("total_tokens_est", 0),
        "backend": fallback.get("backend", "python-fallback"),
        "suggested_strategy": suggested,
    }


def clear_cache(cag_dir: Path = DEFAULT_CAG_DIR) -> dict:
    """Remove the CAG pack directory."""
    if RUST_BINARY is not None:
        return _run_rust(["clear-cache", "--dir", str(cag_dir)])
    import shutil
    if cag_dir.exists():
        shutil.rmtree(cag_dir)
        return {"status": "cleared", "dir": str(cag_dir)}
    return {"status": "not_found", "dir": str(cag_dir)}


def _ollama_request(prompt: str, system: str = "") -> str:
    """Minimal Ollama request to avoid heavy dependencies."""
    import http.client
    import json
    from kryonix_brain_lightrag.config import OLLAMA_BASE_URL, LLM_MODEL
    
    # Simple URL parser for http.client
    from urllib.parse import urlparse
    u = urlparse(OLLAMA_BASE_URL)
    host = u.hostname or "127.0.0.1"
    port = u.port or 11434
    
    conn = http.client.HTTPConnection(host, port, timeout=30)
    payload = {
        "model": LLM_MODEL,
        "prompt": f"System: {system}\n\nUser: {prompt}\n\nAssistant:",
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 512}
    }
    
    try:
        conn.request("POST", "/api/generate", json.dumps(payload), {"Content-Type": "application/json"})
        res = conn.getresponse()
        data = json.loads(res.read().decode())
        return data.get("response", "").strip()
    except Exception as e:
        return f"Error calling Ollama: {e}"
    finally:
        conn.close()


def ask(query: str, cag_dir: Path = DEFAULT_CAG_DIR, top_k: int = 5) -> dict:
    """Full CAG: retrieve context and generate an answer using the LLM."""
    routing_res = route(query, cag_dir=cag_dir, top_k=top_k)
    matched = routing_res.get("matched_files", [])
    
    if not matched:
        return {
            "answer": "Não encontrei arquivos relevantes no repositório para responder essa pergunta via CAG.",
            "sources": [],
            "status": "no_context"
        }

    # Build context block
    context_parts = []
    repo_root = Path(os.environ.get("KRYONIX_REPO", "/etc/kryonix"))
    
    for f in matched:
        path_str = f["path"]
        # Try to read the full file from the repo if possible
        full_path = repo_root / path_str
        content = ""
        
        if full_path.exists() and full_path.is_file():
            try:
                # Read first 50KB to keep it safe
                with open(full_path, "r", encoding="utf-8") as file:
                    content = file.read(51200)
                    if len(content) >= 51200:
                        content += "\n[TRUNCATED...]"
            except Exception:
                content = f.get("snippet", "")
        else:
            content = f.get("snippet", "")
            
        context_parts.append(f"FILE: {path_str}\nCONTENT:\n{content}\n---")
    
    context_str = "\n".join(context_parts)
    
    system_prompt = (
        "Você é o Kryonix CAG (Context-Augmented Generation). "
        "Sua tarefa é responder perguntas sobre o repositório Kryonix usando APENAS o contexto fornecido. "
        "Se o contexto não contiver a resposta, diga que não sabe. "
        "Responda em Português do Brasil, de forma técnica e direta.\n\n"
        "REGRAS OPERACIONAIS ESTRITAS (CONTRATO CANÔNICO):\n"
        "1. SEMPRE sugira comandos da CLI 'kryonix' para operações (ex: 'kryonix check --host glacier', 'kryonix rebuild --host glacier', 'kryonix test --host glacier', 'kryonix switch --host glacier').\n"
        "2. Use OBRIGATORIAMENTE a sintaxe 'kryonix <comando> --host <host>'.\n"
        "3. PROIBIÇÃO: NUNCA sugira 'kryonix <comando> .#<host>' ou 'kryonix build .#host'.\n"
        "4. DOCUMENTAÇÃO ALVO: Priorize sempre 'docs/cli/KRYONIX_COMMAND_CONTRACT.md' for comandos operacionais."
    )
    
    prompt = (
        f"CONTEXTO DO REPOSITÓRIO:\n{context_str}\n\n"
        f"PERGUNTA: {query}\n\n"
        "Com base no contexto acima, responda de forma técnica:"
    )
    
    answer = _ollama_request(prompt, system=system_prompt)
    
    return {
        "answer": answer,
        "sources": [f["path"] for f in matched],
        "matched_files": matched,
        "matched_tags": routing_res.get("matched_tags", []),
        "status": "success",
        "routing": {
            "strategy": routing_res.get("strategy"),
            "confidence": routing_res.get("confidence"),
            "confidence_label": routing_res.get("confidence_label"),
            "reason": routing_res.get("reason"),
        },
    }


def scan_secrets(cag_dir: Path = DEFAULT_CAG_DIR) -> dict:
    """Scan a built CAG directory for leaked secrets."""
    if RUST_BINARY is not None:
        return _run_rust(["scan", "--dir", str(cag_dir)])
    # Python fallback
    manifest_path = cag_dir / MANIFEST_FILENAME
    findings = []
    if manifest_path.exists():
        text = manifest_path.read_text()
        _, redacted = _filter_content(text)
        if redacted:
            findings.append({"path": "manifest.json", "lines": redacted})
    if findings:
        return {"status": "FAIL", "secrets_found": True, "findings": findings}
    return {"status": "clean", "secrets_found": False}


def clear_cache(cag_dir: Path = DEFAULT_CAG_DIR) -> dict:
    """Clear the CAG cache directory."""
    if RUST_BINARY is not None:
        return _run_rust(["clear-cache", "--dir", str(cag_dir)])
    if cag_dir.exists():
        import shutil
        try:
            shutil.rmtree(cag_dir)
            return {"status": "cleared", "path": str(cag_dir), "message": f"CAG directory {cag_dir} removed successfully."}
        except Exception as e:
            return {"status": "error", "message": str(e), "path": str(cag_dir)}
    return {"status": "not_found", "path": str(cag_dir), "message": "CAG directory does not exist."}



def backend_info() -> dict:
    """Return which backend is active."""
    return {
        "rust_binary": str(RUST_BINARY) if RUST_BINARY else None,
        "backend": "rust" if RUST_BINARY else "python-fallback",
    }

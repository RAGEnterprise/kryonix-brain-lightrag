"""
routing.py — Python routing logic for CAG packs.
Used as fallback when the Rust binary is not available.
"""
from __future__ import annotations

from typing import Any

# Keyword → list of tags with weights
_KEYWORD_TAG_WEIGHTS: dict[str, list[tuple[str, float]]] = {
    "glacier":    [("glacier", 2.0), ("host-config", 1.0), ("brain", 0.5)],
    "server":     [("glacier", 1.5), ("host-config", 1.0)],
    "servidor":   [("glacier", 1.5), ("host-config", 1.0)],
    "inspiron":   [("inspiron", 2.0), ("host-config", 1.0)],
    "workstation":[("inspiron", 1.5), ("host-config", 1.0)],
    "brain":      [("brain", 2.0), ("lightrag", 1.5), ("mcp", 0.5)],
    "rag":        [("lightrag", 2.0), ("brain", 1.5)],
    "lightrag":   [("lightrag", 2.0), ("brain", 1.0)],
    "índice":     [("lightrag", 1.5), ("brain", 1.0)],
    "indice":     [("lightrag", 1.5), ("brain", 1.0)],
    "vault":      [("brain", 1.0), ("docs", 0.5)],
    "obsidian":   [("brain", 1.0), ("docs", 0.5)],
    "nota":       [("docs", 1.5), ("brain", 0.5)],
    "note":       [("docs", 1.5), ("brain", 0.5)],
    "bancos":     [("local-sources", 3.5), ("nixos-sources", 1.0), ("docs", 1.0)],
    "fontes":     [("local-sources", 3.5), ("nixos-sources", 1.0), ("docs", 1.0)],
    "locais":     [("local-sources", 3.0), ("nixos-sources", 1.0), ("docs", 0.8)],
    "local":      [("local-sources", 2.0), ("docs", 0.5)],
    "source":     [("local-sources", 1.5), ("docs", 0.5)],
    "sources":    [("local-sources", 3.0), ("nixos-sources", 1.5), ("docs", 1.0)],
    "nixos":      [("local-sources", 2.5), ("nixos-sources", 1.5), ("nix", 1.0)],
    "nixpkgs":    [("local-sources", 2.5), ("nix", 1.0), ("docs", 0.8)],
    "home-manager": [("local-sources", 2.5), ("nix", 1.0), ("docs", 0.8)],
    "noogle":     [("local-sources", 2.5), ("nix", 0.8), ("docs", 0.8)],
    "nvidia":     [("gpu", 2.0), ("glacier", 1.0)],
    "gpu":        [("gpu", 2.0), ("glacier", 1.0)],
    "cuda":       [("gpu", 2.0), ("glacier", 1.0)],
    "ollama":     [("ollama", 2.0), ("glacier", 1.0), ("brain", 0.5)],
    "mcp":        [("mcp", 2.0), ("brain", 0.5)],
    "tailscale":  [("tailscale", 2.0), ("networking", 1.0)],
    "rede":       [("networking", 2.0), ("tailscale", 0.5)],
    "network":    [("networking", 2.0), ("tailscale", 0.5)],
    "firewall":   [("networking", 2.0)],
    "ssh":        [("ssh", 2.0), ("networking", 0.5)],
    "nix":        [("nix", 2.0), ("flake", 0.5)],
    "nixos":      [("nix", 2.0), ("host-config", 1.0)],
    "flake":      [("flake", 2.0), ("nix", 1.0)],
    "módulo":     [("nixos-module", 2.0), ("nix", 1.0)],
    "modulo":     [("nixos-module", 2.0), ("nix", 1.0)],
    "module":     [("nixos-module", 2.0), ("nix", 1.0)],
    "rebuild":    [("nix", 1.5), ("host-config", 1.0)],
    "switch":     [("nix", 1.5), ("host-config", 1.0)],
    "kryonix":    [("cli", 2.5), ("operations", 1.5), ("docs", 1.0)],
    "check":      [("operations", 2.0), ("cli", 1.0)],
    "home":       [("operations", 1.5), ("cli", 1.0)],
    "update":     [("operations", 1.5), ("cli", 0.5)],
    "boot":       [("operations", 1.5), ("cli", 0.5)],
    "test":       [("operations", 1.5), ("cli", 0.5)],
    "apply":      [("operations", 1.0), ("cli", 0.5)],
    "audio":      [("audio", 2.0)],
    "pipewire":   [("audio", 2.0)],
    "bluetooth":  [("bluetooth", 2.0), ("audio", 0.5)],
    "som":        [("audio", 2.0)],
    "gaming":     [("gaming", 2.0)],
    "steam":      [("gaming", 2.0)],
    "gamemode":   [("gaming", 2.0)],
    "desktop":    [("desktop", 2.0)],
    "hyprland":   [("desktop", 2.0)],
    "caelestia":  [("desktop", 2.0)],
    "storage":    [("storage", 2.0)],
    "btrfs":      [("storage", 2.0)],
    "disco":      [("storage", 2.0)],
    "disk":       [("storage", 2.0)],
    "agents":     [("agent", 2.0), ("docs", 1.0)],
    "docs":       [("docs", 1.5)],
    "documento":  [("docs", 1.5)],
    "roadmap":    [("docs", 1.5)],
    "procure":    [("docs", 0.5)],
    "busque":     [("docs", 0.5)],
    "encontre":   [("docs", 0.5)],
    "como":       [], # Prevent "como" from matching "disco"
    "no":         [], # Prevent "no" from matching "nota"
}


def get_path_multiplier(path: str, query_lower: str) -> float:
    """Determine path-based multiplier for scoring (Tiers)."""
    path_lower = path.lower()
    
    # Check if query is about disks
    disk_keywords = ["disco", "disk", "partição", "particao", "partition", "instalação", "instalacao", "install", "formatação", "formatacao", "format", "mount", "filesystem", "fs", "nvme", "sda", "vda", "disko", "zfs", "btrfs", "ext4", "storage"]
    is_disk_query = any(k in query_lower for k in disk_keywords)

    # Check if query is about archive
    archive_keywords = ["antigo", "legacy", "archive", "histórico", "historico", "history", "vault"]
    is_archive_query = any(k in query_lower for k in archive_keywords)

    # Tier 4: Disk/Install/ISO Penalty (0.1x)
    # Aggressive penalty for infrastructure/ISO files when not explicitly asked
    iso_keywords = ["iso", "live", "bootable", "usb", "flash", "instalação", "instalacao", "install"]
    is_iso_query = any(k in query_lower for k in iso_keywords)

    if (("disks.nix" in path_lower) or 
        ("disko" in path_lower) or 
        ("partition" in path_lower) or 
        ("live-iso" in path_lower) or
        ("install" in path_lower) or
        ("hardware-configuration" in path_lower)) and not (is_disk_query or is_iso_query):
        return 0.1

    # Tier 4: Archive/Legacy Penalty (0.2x)
    if (("archive/" in path_lower) or 
        ("legacy/" in path_lower) or 
        ("antigo/" in path_lower)) and not is_archive_query:
        return 0.2

    # Tier 1: Canonical Docs (2.0x for specific matches)
    if path_lower.endswith("agents.md") or path_lower.endswith("readme.md") or "docs/hosts/glacier.md" in path_lower:
        return 2.0

    # Tier 1: Canonical Docs (4.0x for specific matches)
    if (("docs/hosts/glacier-switch.md" in path_lower or "docs/hosts/glacier-rebuild.md" in path_lower) and 
        ("rebuild" in query_lower or "switch" in query_lower)):
        return 5.0

    if ("docs/ai/nixos-local-knowledge-sources.md" in path_lower or
        ".ai/skills/brain/nixos-local-sources.md" in path_lower) and any(
        term in query_lower for term in ["bancos", "fontes", "locais", "local", "source", "sources", "nixos"]
    ):
        return 10.0

    if ("docs/ai/" in path_lower or ".ai/skills/brain/" in path_lower) and any(
        term in query_lower for term in ["bancos", "fontes", "locais", "local", "source", "sources", "nixos"]
    ):
        return 6.0

    if any(
        marker in path_lower for marker in [
            "hosts/glacier/default.nix",
            "profiles/glacier-ai.nix",
            "modules/nixos/services/brain.nix",
            "hardware-configuration.nix",
        ]
    ) and any(term in query_lower for term in ["bancos", "fontes", "locais", "local", "source", "sources", "nixos"]):
        return 0.2

    if any(
        marker in path_lower for marker in [
            "docs/cli.md",
            "docs/operations.md",
            "docs/hosts/glacier-rebuild.md",
            "docs/hosts/glacier-switch.md",
            ".ai/skills/commands/rebuild-nixos.md",
            "packages/kryonix-cli.nix",
        ]
    ) and any(term in query_lower for term in ["check", "rebuild", "switch", "home", "update", "boot", "test", "apply", "kryonix"]):
        return 7.0

    if any(
        marker in path_lower for marker in [
            "hosts/glacier/default.nix",
            "profiles/glacier-ai.nix",
            "modules/nixos/services/brain.nix",
            "hardware-configuration.nix",
        ]
    ) and any(term in query_lower for term in ["check", "rebuild", "switch", "home", "update", "boot", "test", "apply", "kryonix"]):
        return 0.2

    # Tier 1: General Canonical Docs (3.0x)
    if ("docs/hosts/" in path_lower or 
        "docs/ai/" in path_lower or 
        ".ai/skills/" in path_lower or 
        path_lower.endswith("readme.md") or 
        path_lower.endswith("agents.md")):
        return 3.0

    # Tier 2: Key Configs (1.5x)
    if ("hosts/glacier/default.nix" in path_lower or 
        "profiles/glacier-ai.nix" in path_lower or 
        "modules/nixos/services/brain.nix" in path_lower or
        "modules/nixos/ai/" in path_lower or
        "flake.nix" in path_lower):
        return 1.5

    return 1.0


def suggest_strategy(query: str) -> dict[str, Any]:
    """Suggest the best search strategy (cag, rag, hybrid) for a query."""
    query_lower = query.lower()

    # RAG/Hybrid triggers: vault, deep knowledge, history, conversations
    rag_triggers = {
        "vault": 0.8,
        "histórico": 0.7,
        "historico": 0.7,
        "nota antiga": 0.6,
        "conversa anterior": 0.8,
        "brain": 0.4,
        "lightrag": 0.4,
        "pensamento": 0.5,
        "log": 0.7,
        "incidente": 0.8,
        "decisão": 0.7,
        "grounding": 0.6, "conhecimento": 0.5,
        "ontem": 0.6,
        "passado": 0.5,
        "conversamos": 0.7,
    }

    # CAG triggers: repo structure, specific configs, nix files, current implementation
    cag_triggers = {
        "cag": 1.0,
        "context": 0.9,
        "como funciona": 1.0,
        "onde fica": 1.0,
        "configuração": 0.9,
        "configuracao": 0.9,
        "nix": 0.9,
        "flake": 1.0,
        "host": 1.0,
        "glacier": 1.0,
        "inspiron": 1.0,
        "código": 0.9,
        "codigo": 0.9,
        "implementação": 0.9,
        "implementacao": 0.9,
        "módulo": 0.9,
        "modulo": 0.9,
        "package": 0.9,
        "pacote": 0.9,
        "setup": 0.8,
        "instalar": 0.7,
        "build": 0.8,
        "rebuild": 1.0,
        "como faço": 1.0,
        "como faco": 1.0,
        "guia": 0.8,
        "tutorial": 0.7,
        "cli": 0.9,
        "comando": 0.8,
        "kryonix": 1.0,
        "check": 0.9,
        "home": 0.8,
        "update": 0.7,
        "boot": 0.7,
        "test": 0.8,
        "docker": 0.4,
        "systemd": 0.9,
        "service": 0.8,
        "serviço": 0.8,
        "run": 0.6,
        "executar": 0.6,
        "seguro": 0.5,
    }

    rag_score = 0.0
    for t, weight in rag_triggers.items():
        if t in query_lower:
            rag_score = max(rag_score, weight)

    cag_score = 0.0
    for t, weight in cag_triggers.items():
        if t in query_lower:
            cag_score = max(cag_score, weight)

    # ── Decision Logic ────────────────────────────────────────────────────────
    # Explicitly force CAG for canonical terms even if RAG scores something
    canonical_terms = ["glacier", "inspiron", "nix", "flake", "rebuild", "switch", "kryonix", "cli", "diagnóstico", "diagnostico"]
    is_canonical = any(t in query_lower for t in canonical_terms)

    # If it's a technical "how to" or canonical term, bias heavily towards CAG
    if is_canonical and not any(t in query_lower for t in ["vault", "histórico", "historico", "log"]):
        cag_score = max(cag_score, 0.98)

    if cag_score > rag_score and cag_score > 0.3:
        confidence = float(cag_score)
        return {
            "strategy": "cag",
            "score": confidence,
            "confidence": confidence,
            "confidence_label": "Alta" if confidence > 0.7 else "Média" if confidence > 0.4 else "Baixa",
            "reason": "Query focuses on repository structure, hosts, or code implementation.",
        }
    if rag_score > 0.3:
        confidence = float(rag_score)
        return {
            "strategy": "rag",
            "score": confidence,
            "confidence": confidence,
            "confidence_label": "Alta" if confidence > 0.7 else "Média" if confidence > 0.4 else "Baixa",
            "reason": "Query mentions vault, history, or knowledge base concepts.",
        }
    
    # Default to hybrid with low confidence
    return {
        "strategy": "hybrid",
        "score": 0.2,
        "confidence": 0.2,
        "confidence_label": "Baixa",
        "reason": "General query, hybrid provides broad coverage.",
    }


def route_query_python(manifest: dict, query: str, top_k: int = 10) -> dict:
    """
    Pure-Python semantic router.
    Maps query keywords to manifest tags, then ranks files by tag overlap.
    """
    query_lower = query.lower()
    words = query_lower.split()
    tag_scores: dict[str, float] = {}

    for word in words:
        # Exact match
        if word in _KEYWORD_TAG_WEIGHTS:
            for tag, weight in _KEYWORD_TAG_WEIGHTS[word]:
                tag_scores[tag] = tag_scores.get(tag, 0.0) + weight
        # Substring match (strict: length > 3 to avoid noise)
        for kw, pairs in _KEYWORD_TAG_WEIGHTS.items():
            if len(word) > 3 and len(kw) > 3 and (word in kw or kw in word):
                for tag, weight in pairs:
                    tag_scores[tag] = tag_scores.get(tag, 0.0) + weight * 0.5

    # Top matched tags
    matched_tags = sorted(tag_scores, key=lambda t: -tag_scores[t])[:8]

    # Build reverse index: file_path → set of tags
    tags_index: dict[str, set[str]] = {}
    for tag, paths in manifest.get("tags", {}).items():
        for path in paths:
            tags_index.setdefault(path, set()).add(tag)

    # Score each file
    scored: list[tuple[str, float]] = []
    for f in manifest.get("files", []):
        path = f["path"]
        file_tags = tags_index.get(path, set())
        
        # 1. Base score from tags
        base_score = sum(
            tag_scores.get(t, 0.0) for t in matched_tags if t in file_tags
        )
        
        # 2. Path bonus
        path_lower = path.lower()
        path_bonus = 0.0
        for w in words:
            if len(w) > 3 and w in path_lower:
                path_bonus += 0.5
        
        total_pre_multiplier = base_score + path_bonus
        
        # 3. Tiered Priority and Penalties
        multiplier = get_path_multiplier(path, query_lower)
        final_score = total_pre_multiplier * multiplier
        
        if final_score > 0:
            scored.append((path, final_score))

    scored.sort(key=lambda x: -x[1])
    scored = scored[:top_k]

    # Build result
    total_tokens = 0
    matched_files = []
    path_to_file = {f["path"]: f for f in manifest.get("files", [])}
    for path, score in scored:
        f = path_to_file.get(path, {})
        content = f.get("content", "")
        snippet = content[:300]
        token_est = len(content) // 4
        total_tokens += token_est
        matched_files.append({
            "path": path,
            "score": round(score, 2),
            "tags": sorted(tags_index.get(path, set())),
            "snippet": snippet,
        })

    suggested = suggest_strategy(query)
    return {
        "query": query,
        "strategy": suggested.get("strategy", "hybrid"),
        "confidence": suggested.get("confidence", suggested.get("score", 0.2)),
        "confidence_label": suggested.get("confidence_label", "Baixa"),
        "reason": suggested.get("reason", ""),
        "matched_tags": matched_tags,
        "matched_files": matched_files,
        "total_tokens_est": total_tokens,
        "backend": "python-fallback",
        "suggested_strategy": suggested,
    }

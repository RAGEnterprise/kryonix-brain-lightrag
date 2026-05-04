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
    "doc":        [("docs", 1.5)],
    "roadmap":    [("docs", 1.5)],
    "procure":    [("docs", 0.5)],
    "busque":     [("docs", 0.5)],
    "encontre":   [("docs", 0.5)],
}


def route_query_python(manifest: dict, query: str, top_k: int = 10) -> dict:
    """
    Pure-Python semantic router.
    Maps query keywords to manifest tags, then ranks files by tag overlap.
    """
    words = query.lower().split()
    tag_scores: dict[str, float] = {}

    for word in words:
        # Exact match
        if word in _KEYWORD_TAG_WEIGHTS:
            for tag, weight in _KEYWORD_TAG_WEIGHTS[word]:
                tag_scores[tag] = tag_scores.get(tag, 0.0) + weight
        # Substring match
        for kw, pairs in _KEYWORD_TAG_WEIGHTS.items():
            if len(word) > 3 and (word in kw or kw in word):
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
        score = sum(
            tag_scores.get(t, 0.0) for t in matched_tags if t in file_tags
        )
        # Path bonus
        path_lower = path.lower()
        for w in words:
            if len(w) > 3 and w in path_lower:
                score += 0.5
        if score > 0:
            scored.append((path, score))

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

    return {
        "query": query,
        "matched_tags": matched_tags,
        "matched_files": matched_files,
        "total_tokens_est": total_tokens,
        "backend": "python-fallback",
    }

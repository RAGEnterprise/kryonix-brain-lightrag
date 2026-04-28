"""Export LightRAG graph to Obsidian-compatible markdown vault."""

import json
import os
import re
import shutil
import sys
from pathlib import Path

import networkx as nx

from .config import STORAGE_DIR, VAULT_PATH
from .rag import slugify


def _ensure_vault():
    """Create vault directory structure and Obsidian config."""
    os.makedirs(VAULT_PATH / "entities", exist_ok=True)
    os.makedirs(VAULT_PATH / "sources", exist_ok=True)
    os.makedirs(VAULT_PATH / "communities", exist_ok=True)
    os.makedirs(VAULT_PATH / ".obsidian", exist_ok=True)

    # Graph view config for Obsidian
    graph_config = {
        "collapse-filter": False,
        "search": "",
        "showTags": False,
        "showAttachments": False,
        "hideUnresolved": False,
        "showOrphans": True,
        "collapse-color-groups": False,
        "colorGroups": [
            {"query": "path:entities", "color": {"a": 1, "r": 100, "g": 200, "b": 255}},
            {"query": "path:sources", "color": {"a": 1, "r": 255, "g": 200, "b": 100}},
            {"query": "path:communities", "color": {"a": 1, "r": 200, "g": 255, "b": 100}},
        ],
        "collapse-display": False,
        "showArrow": True,
        "textFadeMultiplier": 0,
        "nodeSizeMultiplier": 1,
        "lineSizeMultiplier": 1,
        "collapse-forces": True,
        "centerStrength": 0.5,
        "repelStrength": 10,
        "linkStrength": 1,
        "linkDistance": 250,
        "scale": 1,
        "close": False,
    }
    with open(VAULT_PATH / ".obsidian" / "graph.json", "w", encoding="utf-8") as f:
        json.dump(graph_config, f, indent=2)


def _load_graph() -> nx.Graph | None:
    """Load the LightRAG graph."""
    graph_file = os.path.join(STORAGE_DIR, "graph_chunk_entity_relation.graphml")
    if os.path.exists(graph_file):
        return nx.read_graphml(graph_file)
    return None


def _get_communities(G: nx.Graph) -> dict[int, list[str]]:
    """Detect communities using Louvain method."""
    try:
        from networkx.algorithms.community import louvain_communities
        communities = louvain_communities(G, seed=42)
        result = {}
        for i, community in enumerate(communities):
            result[i] = sorted(community)
        return result
    except Exception:
        return {}


def _export_entities(G: nx.Graph):
    """Export each entity as an Obsidian note."""
    for node in G.nodes():
        data = G.nodes[node]
        slug = slugify(node)
        if not slug:
            continue

        # Get entity metadata
        entity_type = data.get("entity_type", "unknown")
        description = data.get("description", "")
        source_id = data.get("source_id", "")

        # Build note content
        lines = [
            f"# {node}",
            "",
            f"**Type:** {entity_type}",
            "",
        ]

        if description:
            lines.append(f"## Description")
            lines.append("")
            lines.append(description)
            lines.append("")

        # Add neighbor links
        neighbors = list(G.neighbors(node))
        if neighbors:
            lines.append("## Related Entities")
            lines.append("")
            for nb in sorted(neighbors):
                nb_slug = slugify(nb)
                edge_data = G.get_edge_data(node, nb) or {}
                rel_desc = edge_data.get("description", "")
                link_text = f"- [[entities/{nb_slug}|{nb}]]"
                if rel_desc:
                    link_text += f" - {rel_desc[:100]}"
                lines.append(link_text)
            lines.append("")

        # Add source references using file_path (not source_id)
        if source_id:
            sources = [s.strip() for s in source_id.split("<SEP>") if s.strip()]
            if sources:
                lines.append("## Appears in")
                lines.append("")
                for src in sources:
                    src_slug = slugify(src)
                    lines.append(f"- [[sources/{src_slug}|{src}]]")
                lines.append("")

        filepath = VAULT_PATH / "entities" / f"{slug}.md"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def _export_sources(G: nx.Graph):
    """Export source file references as Obsidian notes."""
    sources = set()
    for node in G.nodes():
        data = G.nodes[node]
        source_id = data.get("source_id", "")
        if source_id:
            for src in source_id.split("<SEP>"):
                src = src.strip()
                if src:
                    sources.add(src)

    for src in sources:
        slug = slugify(src)
        if not slug:
            continue

        # Find entities that appear in this source
        related = []
        for node in G.nodes():
            data = G.nodes[node]
            node_sources = data.get("source_id", "")
            if src in node_sources:
                related.append(node)

        lines = [
            f"# {src}",
            "",
            "## Entities in this file",
            "",
        ]
        for entity in sorted(related):
            ent_slug = slugify(entity)
            lines.append(f"- [[entities/{ent_slug}|{entity}]]")
        lines.append("")

        filepath = VAULT_PATH / "sources" / f"{slug}.md"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def _export_communities(G: nx.Graph):
    """Export community clusters as Obsidian notes."""
    communities = _get_communities(G)
    if not communities:
        return

    for idx, members in communities.items():
        lines = [
            f"# Community {idx}",
            "",
            f"**Members:** {len(members)}",
            "",
            "## Entities",
            "",
        ]
        for member in members[:50]:  # cap at 50 per note
            slug = slugify(member)
            lines.append(f"- [[entities/{slug}|{member}]]")
        if len(members) > 50:
            lines.append(f"- ... and {len(members) - 50} more")
        lines.append("")

        filepath = VAULT_PATH / "communities" / f"community-{idx}.md"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def _export_index(G: nx.Graph):
    """Export INDEX.md with top entities by degree."""
    degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    top = degrees[:30]

    lines = [
        "# Knowledge Graph Index",
        "",
        f"**Total entities:** {G.number_of_nodes()}",
        f"**Total relations:** {G.number_of_edges()}",
        "",
        "## Top Entities (by connections)",
        "",
        "| Entity | Connections |",
        "|--------|------------|",
    ]
    for node, deg in top:
        slug = slugify(node)
        lines.append(f"| [[entities/{slug}|{node}]] | {deg} |")
    lines.append("")

    with open(VAULT_PATH / "INDEX.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    """CLI entry point for kg-to-obsidian."""
    clean = "--clean" in sys.argv

    G = _load_graph()
    if G is None:
        print("No graph found. Run kg-index first.")
        return

    if clean and VAULT_PATH.exists():
        # Remove existing vault content (preserve .obsidian config)
        for sub in ["entities", "sources", "communities"]:
            sub_path = VAULT_PATH / sub
            if sub_path.exists():
                shutil.rmtree(sub_path)
        index_file = VAULT_PATH / "INDEX.md"
        if index_file.exists():
            index_file.unlink()

    _ensure_vault()
    print(f"Exporting to {VAULT_PATH}...")

    _export_entities(G)
    _export_sources(G)
    _export_communities(G)
    _export_index(G)

    # Count what we created
    ent_count = len(list((VAULT_PATH / "entities").glob("*.md")))
    src_count = len(list((VAULT_PATH / "sources").glob("*.md")))
    com_count = len(list((VAULT_PATH / "communities").glob("*.md")))

    print(f"Export complete: {ent_count} entities, {src_count} sources, {com_count} communities")
    print(f"Open in Obsidian: {VAULT_PATH}")


if __name__ == "__main__":
    main()

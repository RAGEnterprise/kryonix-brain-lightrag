import json
import os
import re
import shutil
from pathlib import Path
from datetime import datetime
import networkx as nx
from typing import List, Dict, Set, Tuple

# Project Paths
PROJECT_ROOT = Path(r"C:\Users\aguia\Documents\kryonix")
VAULT_ROOT = Path(r"C:\Users\aguia\Documents\kryonix-vault")
STORAGE_DIR = VAULT_ROOT / "11-LightRAG" / "rag_storage"
BACKUP_DIR = VAULT_ROOT / ".backups"
OUTPUT_DIR = VAULT_ROOT / "00-System"
ANALYSIS_FILE = VAULT_ROOT / "11-LightRAG" / "graph-analysis.md"

GRAPH_FILE = STORAGE_DIR / "graph_chunk_entity_relation.graphml"

def ensure_dirs():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def create_backup(path: Path):
    if not path.exists(): return
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Safe backup path
    rel_path = str(path.relative_to(VAULT_ROOT)).replace(os.sep, "_")
    backup_name = f"{rel_path}.{ts}.bak"
    shutil.copy2(path, BACKUP_DIR / backup_name)

def safe_write(path: Path, content: str):
    if path.exists():
        create_backup(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"Written: {path}")

def load_graph() -> nx.Graph:
    if not GRAPH_FILE.exists():
        raise FileNotFoundError(f"Graph file not found at {GRAPH_FILE}")
    return nx.read_graphml(GRAPH_FILE)

def cluster_entities(G: nx.Graph):
    clusters = {
        "Desktop": [],
        "NixOS": [],
        "CLI": [],
        "Infra": [],
        "Kryonix": []
    }
    
    keywords = {
        "Desktop": ["hyprland", "wayland", "caelestia", "desktop", "ui", "ux", "graphics", "monitor", "rofi", "dmenu", "waybar", "display", "nvidia", "amd"],
        "NixOS": ["nixos", "flake", "configuration", "module", "nix", "nixpkgs", "system", "kernel", "boot", "hardware", "disko"],
        "CLI": ["cli", "command", "terminal", "rag", "ragos", "kryonix-cli", "bash", "sh", "zsh", "fish", "powershell", "script"],
        "Infra": ["build", "deploy", "pipeline", "docker", "ci", "cd", "script", "infrastructure", "server", "network", "firewall", "tailscale", "libvirt"],
        "Kryonix": ["kryonix", "vault", "workspace", "project", "obsidian", "brain"]
    }
    
    # Pre-calculate degrees
    degrees = dict(G.degree())
    
    node_to_cluster = {}

    for node in G.nodes():
        node_lower = str(node).lower()
        best_cluster = "Kryonix" # Default
        max_matches = 0
        
        for cluster, kws in keywords.items():
            matches = sum(1 for kw in kws if kw in node_lower)
            if matches > max_matches:
                max_matches = matches
                best_cluster = cluster
        
        clusters[best_cluster].append(node)
        node_to_cluster[node] = best_cluster
        
    # Sort clusters by degree
    for cluster in clusters:
        clusters[cluster].sort(key=lambda x: degrees.get(x, 0), reverse=True)
        
    return clusters, node_to_cluster

def generate_moc_content(title: str, entities: List[str], clusters: Dict):
    content = f"# MOC - {title}\n\n"
    
    sections = {
        "Core": [],
        "Desktop": [],
        "CLI": [],
        "Infra": [],
        "Relacionado": []
    }
    
    # Select top 50
    selected = entities[:50]
    
    for ent in selected:
        # Determine category
        cat = "Relacionado"
        if ent in clusters["Desktop"]: cat = "Desktop"
        elif ent in clusters["CLI"]: cat = "CLI"
        elif ent in clusters["Infra"]: cat = "Infra"
        elif ent in clusters["NixOS"]: cat = "Core"
        
        sections[cat].append(ent)
    
    for sec, items in sections.items():
        if items:
            content += f"## {sec}\n"
            for item in items:
                content += f"- [[{item}]]\n"
            content += "\n"
            
    content += "## Relacionado\n"
    content += "- [[MOC - Kryonix]]\n"
    content += "- [[LightRAG]]\n"
    content += "- [[Obsidian]]\n\n"
    
    content += "---\n"
    content += f"Gerado em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n"
    return content

def main():
    ensure_dirs()
    print("Iniciando análise do grafo...")
    G = load_graph()
    
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    
    clusters, node_to_cluster = cluster_entities(G)
    
    # 1. Graph Analysis
    total_nodes = len(G.nodes())
    total_edges = len(G.edges())
    orphans = [n for n, d in degrees.items() if d == 0]
    low_conn = [n for n, d in degrees.items() if d == 1]
    
    analysis_content = f"""# Análise do Grafo - Kryonix

- **Total de Entidades**: {total_nodes}
- **Total de Relações**: {total_edges}
- **Nós Isolados (Órfãos)**: {len(orphans)}
- **Nós com Baixa Conectividade (Grau 1)**: {len(low_conn)}

## Top 20 Entidades (Mais Conectadas)
"""
    for n, d in sorted_nodes[:20]:
        analysis_content += f"- [[{n}]] (Grau: {d})\n"

    analysis_content += "\n## Top 20 Relações\n"
    sorted_edges = sorted(G.edges(data=True), key=lambda x: float(x[2].get('weight', 1)), reverse=True)
    for u, v, d in sorted_edges[:20]:
        analysis_content += f"- [[{u}]] <-> [[{v}]] (Peso: {d.get('weight', 1)})\n"

    analysis_content += "\n## Sugestões de Melhoria\n"
    analysis_content += "1. **MOCs Estruturais**: Foram gerados hubs para Desktop, NixOS, CLI e Infra.\n"
    analysis_content += "2. **Reparo Semântico**: Links automáticos sugeridos para nós órfãos no arquivo correspondente.\n"
    analysis_content += "3. **Refinamento**: Rodar `rag index --refine` para aumentar a densidade do grafo.\n"

    safe_write(ANALYSIS_FILE, analysis_content)

    # 2. MOC Generation
    moc_configs = [
        ("Kryonix", clusters["Kryonix"] + clusters["NixOS"][:10]),
        ("Desktop", clusters["Desktop"]),
        ("NixOS", clusters["NixOS"]),
        ("CLI", clusters["CLI"]),
        ("Infra", clusters["Infra"])
    ]

    for title, entities in moc_configs:
        path = OUTPUT_DIR / f"MOC - {title}.md"
        content = generate_moc_content(title, entities, clusters)
        safe_write(path, content)

    # 3. Orphan Repair
    orphan_file = OUTPUT_DIR / "Sugestões de Conexão - Órfãos.md"
    orphan_content = "# Sugestões de Conexão - Órfãos\n\n"
    orphan_content += "Lista de entidades sem conexões e sugestões de linkagem baseadas em clusters de conhecimento:\n\n"
    
    # Process up to 1000 orphans
    for orphan in orphans[:1000]:
        cluster = node_to_cluster.get(orphan, "Kryonix")
        # Find 3 best nodes in same cluster (hubs)
        hubs = [h for h in clusters[cluster] if h != orphan and degrees.get(h, 0) > 2][:3]
        
        orphan_content += f"### [[{orphan}]]\n"
        orphan_content += f"Cluster sugerido: **{cluster}**\n"
        for hub in hubs:
            orphan_content += f"- [[{hub}]] (Hub sugerido)\n"
        orphan_content += "\n"
        
    safe_write(orphan_file, orphan_content)
    
    print("Processamento concluído com sucesso.")

if __name__ == "__main__":
    main()

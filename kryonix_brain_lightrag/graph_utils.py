import json
import os
import re
import shutil
from pathlib import Path
from datetime import datetime
import networkx as nx
from typing import List, Dict, Set, Tuple, Optional
from .config import (
    VAULT_DIR, WORKING_DIR, OBSIDIAN_EXPORT_DIR,
    VAULT_EXCLUDE_DIRS
)
# Removed top-level .rag import to avoid circular dependencies
import xml.etree.ElementTree as ET

# Re-use logic from generate_mocs.py script but as a library
BACKUP_DIR = VAULT_DIR / ".backups"

def safe_filename(name: str) -> str:
    """Sanitize entity names for safe filesystem storage."""
    name = str(name).strip()
    name = name.replace("/", " - ").replace("\\", " - ")
    name = re.sub(r'[<>:"|?*]', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name[:120]

def get_unique_path(base_dir: Path, filename: str) -> Path:
    safe_name = safe_filename(filename)
    if not safe_name.endswith(".md"):
        safe_name += ".md"
    target = base_dir / safe_name
    try:
        abs_target = target.absolute()
        abs_vault = VAULT_DIR.absolute()
        if not str(abs_target).startswith(str(abs_vault)):
            target = base_dir / "safe_name_fallback.md"
    except:
        target = base_dir / "safe_name_fallback.md"
    if not target.exists():
        return target
    stem = Path(safe_name).stem
    counter = 1
    while True:
        new_name = f"{stem} ({counter}).md"
        new_target = base_dir / new_name
        if not new_target.exists():
            return new_target
        counter += 1
        if counter > 100: break
    return target

def create_backup(path: Path):
    if not path.exists(): return
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    try:
        rel_path = str(path.relative_to(VAULT_DIR)).replace(os.sep, "_")
    except ValueError:
        rel_path = str(path.name)
    backup_name = f"{rel_path}.{ts}.bak"
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, BACKUP_DIR / backup_name)

def safe_write(path: Path, content: str, backup: bool = True):
    if backup and path.exists():
        create_backup(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def validate_graphml(path: Path) -> tuple[bool, str]:
    """Strict validation of GraphML file."""
    if not path.exists():
        return False, "Arquivo não existe."
    if path.stat().st_size == 0:
        return False, "Arquivo está vazio (0 bytes)."
    
    # XML Syntax check
    try:
        ET.parse(path)
    except ET.ParseError as e:
        return False, f"XML inválido: {e}"
    except Exception as e:
        return False, f"Erro ao ler XML: {e}"
        
    # NetworkX load check
    try:
        G = nx.read_graphml(path)
        if len(G.nodes) == 0:
            return False, "Grafo não possui nós (está vazio)."
    except Exception as e:
        return False, f"NetworkX falhou ao carregar: {e}"
        
    return True, "OK"

def atomic_write_graphml(G: nx.Graph, path: Path):
    """Write GraphML atomically with validation and backup."""
    tmp_path = path.with_suffix(".tmp.graphml")
    
    # 1. Write to tmp
    nx.write_graphml(G, tmp_path)
    
    # 2. Validate tmp
    valid, err = validate_graphml(tmp_path)
    if not valid:
        if tmp_path.exists(): tmp_path.unlink()
        raise IOError(f"Falha na validação atômica do Grafo: {err}")
        
    # 3. Backup original (limit to one backup here, rotating is handled elsewhere)
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = path.with_suffix(f".bak-{ts}.graphml")
        shutil.copy2(path, backup)
        
    # 4. Replace
    try:
        if os.name == 'nt' and path.exists():
            # Windows: use replace which is mostly atomic, but handle existing
            os.replace(str(tmp_path), str(path))
        else:
            tmp_path.replace(path)
    except Exception as e:
        # Fallback to non-atomic if replace fails
        if tmp_path.exists():
            shutil.move(str(tmp_path), str(path))

def atomic_write_json(data: dict, path: Path):
    """Write JSON atomically with validation."""
    tmp_path = path.with_suffix(".tmp.json")
    
    # 1. Write to tmp
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # 2. Validate tmp
    if tmp_path.stat().st_size == 0:
        if tmp_path.exists(): tmp_path.unlink()
        raise IOError("Falha na validação atômica do JSON: Arquivo vazio.")
    
    try:
        with open(tmp_path, "r", encoding="utf-8") as f:
            json.load(f)
    except Exception as e:
        if tmp_path.exists(): tmp_path.unlink()
        raise IOError(f"Falha na validação atômica do JSON: {e}")

    # 3. Replace
    try:
        os.replace(str(tmp_path), str(path))
    except Exception:
        if tmp_path.exists():
            shutil.move(str(tmp_path), str(path))

def cluster_entities(G: nx.Graph):
    clusters = {"Desktop": [], "NixOS": [], "CLI": [], "Infra": [], "Kryonix": []}
    keywords = {
        "Desktop": ["hyprland", "wayland", "caelestia", "desktop", "ui", "ux", "graphics", "monitor", "rofi", "dmenu", "waybar", "display", "nvidia", "amd"],
        "NixOS": ["nixos", "flake", "configuration", "module", "nix", "nixpkgs", "system", "kernel", "boot", "hardware", "disko"],
        "CLI": ["cli", "command", "terminal", "rag", "ragos", "kryonix-cli", "bash", "sh", "zsh", "fish", "powershell", "script"],
        "Infra": ["build", "deploy", "pipeline", "docker", "ci", "cd", "script", "infrastructure", "server", "network", "firewall", "tailscale", "libvirt"],
        "Kryonix": ["kryonix", "vault", "workspace", "project", "obsidian", "brain"]
    }
    node_to_cluster = {}
    for node in G.nodes():
        node_lower = str(node).lower()
        best_cluster = "Kryonix"
        max_matches = 0
        for cluster, kws in keywords.items():
            matches = sum(1 for kw in kws if kw in node_lower)
            if matches > max_matches:
                max_matches = matches
                best_cluster = cluster
        clusters[best_cluster].append(node)
        node_to_cluster[node] = best_cluster
    return clusters, node_to_cluster

async def generate_mocs(verbose=False):
    from .rag import get_rag_async, get_graph
    rag = await get_rag_async()
    G = get_graph()
    if G is None: return "Grafo não carregado."
    
    # Ranking logic
    ranked_entities = await _get_ranked_entities(G)
    clusters, node_to_cluster = cluster_entities(G)
    output_dir = VAULT_DIR / "00-System"
    
    moc_configs = [
        ("Kryonix", [e["name"] for e in ranked_entities if node_to_cluster.get(e["name"]) == "Kryonix" or node_to_cluster.get(e["name"]) == "NixOS"]),
        ("Desktop", [e["name"] for e in ranked_entities if node_to_cluster.get(e["name"]) == "Desktop"]),
        ("NixOS", [e["name"] for e in ranked_entities if node_to_cluster.get(e["name"]) == "NixOS"]),
        ("CLI", [e["name"] for e in ranked_entities if node_to_cluster.get(e["name"]) == "CLI"]),
        ("Infra", [e["name"] for e in ranked_entities if node_to_cluster.get(e["name"]) == "Infra"])
    ]

    for title, entities in moc_configs:
        if not entities: continue
        limited_entities = entities[:50]
        filename = f"MOC - {title}.md"
        path = get_unique_path(output_dir, filename)
        
        summary = "Resumo automático pendente."
        try:
            sample_nodes = limited_entities[:15]
            summary_prompt = f"Gere um resumo técnico de 2 frases em português sobre o cluster '{title}' baseado nestas entidades: {', '.join(sample_nodes)}. Preserve termos técnicos."
            res = await rag.llm_response(summary_prompt)
            if res: summary = res.strip()
        except: pass

        content = f"# MOC - {title}\n\n"
        content += f"## Resumo do Cluster\n{summary}\n\n"
        
        content += "## Mais Importantes (Ranking Semântico)\n"
        for node_name in limited_entities[:10]:
            safe_n = str(node_name).replace("/", " - ").replace("\\", " - ")
            content += f"- [[{safe_n}]]\n"
        content += "\n"
        
        content += "## Relacionamentos Críticos\n"
        critical_edges = []
        for node in limited_entities[:20]:
            for nb in G.neighbors(node):
                edge = G.get_edge_data(node, nb)
                weight = float(edge.get("weight", 1)) if edge else 1
                if weight > 5:
                    critical_edges.append((node, nb, weight, edge.get("description", "")))
        
        critical_edges.sort(key=lambda x: x[2], reverse=True)
        for src, tgt, w, desc in critical_edges[:10]:
            safe_src = str(src).replace("/", " - ").replace("\\", " - ")
            safe_tgt = str(tgt).replace("/", " - ").replace("\\", " - ")
            content += f"- [[{safe_src}]] → [[{safe_tgt}]] : {desc} (Peso: {w})\n"
        content += "\n"
        
        content += "## Entidades do Cluster\n"
        for ent in limited_entities:
            safe_ent = str(ent).replace("/", " - ").replace("\\", " - ")
            content += f"- [[{ent}]]\n"
            
        content += "\n## Sistema\n- [[MOC - Kryonix]]\n- [[LightRAG]]\n- [[Obsidian]]\n\n---\n"
        content += f"Gerado em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n"
        safe_write(path, content)

    # Top Entities report
    await _generate_top_entities_report(ranked_entities, node_to_cluster)
    await _generate_analysis_files(G, dict(G.degree()), clusters, node_to_cluster)
    
    return f"MOCs e Ranking gerados em {output_dir}"

async def _get_ranked_entities(G: nx.Graph) -> List[Dict]:
    """Calculate ranking based on degree, frequency and centrality."""
    degrees = dict(G.degree())
    try:
        pagerank = nx.pagerank(G, weight='weight')
    except:
        pagerank = {n: 0 for n in G.nodes()}
        
    # Load frequency from entity_chunks
    entity_chunks_path = Path(WORKING_DIR) / "kv_store_entity_chunks.json"
    freq = {}
    if entity_chunks_path.exists():
        try:
            data = json.loads(entity_chunks_path.read_text(encoding="utf-8"))
            for node, chunks in data.items():
                freq[node] = len(chunks)
        except: pass

    ranked = []
    for node in G.nodes():
        d = degrees.get(node, 0)
        p = pagerank.get(node, 0)
        f = freq.get(node, 0)
        # Normalized score (simple heuristic)
        score = (d * 0.4) + (p * 1000 * 0.4) + (f * 0.2)
        ranked.append({
            "name": node,
            "degree": d,
            "pagerank": p,
            "frequency": f,
            "score": round(score, 2)
        })
    
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked

async def _generate_top_entities_report(ranked_entities, node_to_cluster):
    report_path = config.OBSIDIAN_EXPORT_DIR / "top_entities.md"
    content = "# Top 100 Entidades - Kryonix\n\n"
    content += "| Ranking | Entidade | Score | Cluster | Conexões | Freq |\n"
    content += "|---------|----------|-------|---------|----------|------|\n"
    
    for i, e in enumerate(ranked_entities[:100]):
        safe_n = str(e["name"]).replace("/", " - ").replace("\\", " - ")
        cluster = node_to_cluster.get(e["name"], "N/A")
        content += f"| {i+1} | [[{safe_n}]] | {e['score']} | {cluster} | {e['degree']} | {e['frequency']} |\n"
        
    safe_write(report_path, content)

async def _generate_analysis_files(G, degrees, clusters, node_to_cluster):
    analysis_path = config.OBSIDIAN_EXPORT_DIR / "graph-analysis.md"
    orphans = [n for n, d in degrees.items() if d == 0]
    analysis_content = f"# Análise do Grafo - Kryonix\n\n"
    analysis_content += f"- **Total de Entidades**: {len(G.nodes())}\n"
    analysis_content += f"- **Total de Relações**: {len(G.edges())}\n"
    analysis_content += f"- **Órfãos**: {len(orphans)}\n\n"
    analysis_content += "## Top 10 Entidades\n"
    for n, d in sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]:
        safe_n = str(n).replace("/", " - ").replace("\\", " - ")
        analysis_content += f"- [[{safe_n}]] ({d})\n"
    safe_write(analysis_path, analysis_content)

    orphan_path = VAULT_DIR / "00-System" / "Sugestões de Conexão - Órfãos.md"
    orphan_content = "# Sugestões de Conexão - Órfãos\n\n"
    for orphan in orphans[:200]:
        cluster = node_to_cluster.get(orphan, "Kryonix")
        hubs = [h for h in clusters[cluster] if h != orphan and degrees.get(h, 0) > 2][:3]
        safe_o = str(orphan).replace("/", " - ").replace("\\", " - ")
        orphan_content += f"### [[{safe_o}]]\n"
        for hub in hubs: 
            safe_h = str(hub).replace("/", " - ").replace("\\", " - ")
            orphan_content += f"- [[{safe_h}]]\n"
        orphan_content += "\n"
    safe_write(orphan_path, orphan_content)

def export_obsidian(verbose=False, limit: int = 500):
    G = get_graph()
    if G is None: return "Grafo não carregado."
    export_dir = OBSIDIAN_EXPORT_DIR
    export_dir.mkdir(parents=True, exist_ok=True)
    degrees = dict(G.degree())
    sorted_nodes = sorted(G.nodes(data=True), key=lambda x: degrees.get(x[0], 0), reverse=True)
    if limit > 0: nodes_to_export = sorted_nodes[:limit]
    else: nodes_to_export = sorted_nodes
    ent_dir = export_dir / "Entidades"
    ent_dir.mkdir(parents=True, exist_ok=True)
    for node, data in nodes_to_export:
        filename = f"{node}.md"
        path = get_unique_path(ent_dir, filename)
        safe_node = str(node).replace("/", " - ").replace("\\", " - ")
        content = f"# {safe_node}\n\n**Tipo**: {data.get('entity_type', 'unknown')}\n**Grau**: {degrees.get(node, 0)}\n\n## Descrição\n{data.get('description', '')}\n\n## Relações\n"
        for nb in G.neighbors(node):
            edge = G.get_edge_data(node, nb) or {}
            safe_nb = str(nb).replace("/", " - ").replace("\\", " - ")
            content += f"- [[{safe_nb}]] ({edge.get('description', '')})\n"
        content += "\n## Backlinks\n```dataview\nlist from [[{safe_node}]]\n```\n"
        path.write_text(content, encoding="utf-8")
    return f"Grafo exportado para {export_dir}"

async def heal_graph(verbose=False, limit_orphans=50):
    from .rag import get_rag_async, get_graph
    graph_path = WORKING_DIR / "graph_chunk_entity_relation.graphml"
    valid, err = validate_graphml(graph_path)
    if not valid:
        return f"[ERRO] GraphML inválido: {err}. Rode: .\\rag.bat repair-graph"

    rag = await get_rag_async()
    G = get_graph()
    if G is None: return "Grafo não carregado."
    degrees = dict(G.degree())
    orphans = [n for n, d in degrees.items() if d < 2]
    if not orphans: return "Nenhum órfão encontrado."
    to_heal = orphans[:limit_orphans]
    healed_count = 0
    for orphan in to_heal:
        try:
            candidates = await rag.entities_vdb.query(orphan, top_k=5)
            candidates = [c for c in candidates if c["entity_name"] != orphan]
            if not candidates: continue
            candidate_notes = [f"- {c['entity_name']}: {G.nodes[c['entity_name']].get('description', '')}" for c in candidates if c['entity_name'] in G.nodes]
            heal_prompt = f"Create relationships in format: (\"relationship\"<source>, <target>, <desc>, <kws>, <w>)\nEntity: {orphan}\nDesc: {G.nodes[orphan].get('description','')}\nCandidates:\n" + "\n".join(candidate_notes)
            res = await rag.llm_response(heal_prompt)
            if res and "(" in res:
                await rag.ainsert([res], ids=[f"heal-{slugify(orphan)}"])
                healed_count += 1
        except: pass
    return f"Cura concluída. {healed_count} entidades re-conectadas."

def slugify(text: str) -> str:
    value = str(text).strip().lower()
    value = re.sub(r"[/\\]", "-", value)
    value = re.sub(r"[^a-z0-9\s_-]", "", value)
    value = re.sub(r"\s+", "-", value)
    return value[:100]

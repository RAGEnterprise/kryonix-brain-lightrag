from __future__ import annotations

import hashlib
import json
import re
import asyncio
from pathlib import Path
from datetime import datetime

import networkx as nx
from lightrag import LightRAG
from lightrag import QueryParam

from .config import (
    CHUNK_OVERLAP_TOKEN_SIZE,
    CHUNK_TOKEN_SIZE,
    EMBEDDING_BATCH_NUM,
    LLM_MODEL,
    LLM_MODEL_MAX_ASYNC,
    MAX_PARALLEL_INSERT,
    WORKING_DIR,
    RESPONSE_LANGUAGE,
    ANSWER_SYSTEM_PROMPT
)
from .llm import embedding_func, llm_func

from rich.console import Console
console = Console()


_rag_instance: LightRAG | None = None


def doc_id(rel_path: str) -> str:
    digest = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:12]
    return f"doc-{digest}"


def slugify(text: str) -> str:
    value = str(text).strip().lower()
    value = re.sub(r"[/\\]", "-", value)
    value = re.sub(r"[^a-z0-9\s_-]", "", value)
    value = re.sub(r"\s+", "-", value)
    return value[:180] if value else "unknown"

# ── Prompts ──────────────────────────────────────────────────────
ENTITY_EXTRACTION_PROMPT = """-Goal-
Given a text document that is potentially relevant to this project, your task is to identify all entities and their relationships with high semantic density.

-Steps-
1. Identify all entities. For each entity, extract:
   - name: name of the entity, capitalized
   - type: type of the entity (e.g., ORGANIZATION, PERSON, TECHNOLOGY, CONFIG, FILE, MODULE, FUNCTION, etc.)
   - description: comprehensive summary of the entity's role and importance.

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are related.
   Focus on FUNCTIONAL and ARCHITECTURAL relationships. Prioritize:
   - DEPENDENCY: source depends on target to function
   - COMPOSITION: source is part of or contains target
   - USAGE: source uses or invokes target
   - EXECUTION CONTEXT: source runs within or is managed by target
   - CONFIGURATION: source configures target

   For each relationship, extract:
   - source: name of the source entity
   - target: name of the target entity
   - description: explain the EXACT functional relationship
   - keywords: specific relationship type (e.g. depends_on, implements, manages, configures, part_of, uses)
   - weight: an integer from 1 to 10.

3. Output the results in the following STRICT format:
("entity"<entity_name>, <entity_type>, <entity_description>)
("relationship"<source_entity>, <target_entity>, <relationship_description>, <relationship_keywords>, <relationship_weight>)

-Rules-
- Output only the tuples. No intro/outro.
- Avoid generic relationships.
- IMPORTANT: When finished, you MUST output the exact string "<|COMPLETE|>" on a new line.
"""

def _get_rag() -> LightRAG:
    kwargs = {
        "working_dir": str(WORKING_DIR),
        "llm_model_name": LLM_MODEL,
        "llm_model_max_async": LLM_MODEL_MAX_ASYNC,
        "max_parallel_insert": MAX_PARALLEL_INSERT,
        "embedding_batch_num": EMBEDDING_BATCH_NUM,
        "chunk_token_size": CHUNK_TOKEN_SIZE,
        "chunk_overlap_token_size": CHUNK_OVERLAP_TOKEN_SIZE,
    }
    
    kwargs["llm_model_func"] = llm_func
    kwargs["embedding_func"] = embedding_func
    kwargs["llm_model_kwargs"] = {
        "options": {
            "num_ctx": 4096,
            "temperature": 0.1,
        }
    }
    
    from lightrag.prompt import PROMPTS
    PROMPTS["entity_extraction_user_prompt"] = ENTITY_EXTRACTION_PROMPT
    
    return LightRAG(**kwargs)


def get_rag() -> LightRAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = _get_rag()
    return _rag_instance


async def get_rag_async() -> LightRAG:
    rag = get_rag()
    await rag.initialize_storages()
    return rag


# ── Query Intelligence ───────────────────────────────────────────

async def expand_query_semantically(query_text: str) -> str:
    """Expand query with related entities and technical keywords."""
    rag = await get_rag_async()
    try:
        # Find similar entities in vector DB
        candidates = await rag.entities_vdb.query(query_text, top_k=5)
        related_entities = [c["entity_name"] for c in candidates]
        
        if related_entities:
            expansion = f" (Contexto relacionado: {', '.join(related_entities)})"
            return query_text + expansion
    except:
        pass
    return query_text

async def extract_keywords_llm(query_text: str) -> list[str]:
    """Extract technical keywords from the query."""
    rag = await get_rag_async()
    prompt = f"Extraia apenas os termos técnicos e entidades chave desta pergunta como uma lista separada por vírgulas. Pergunta: {query_text}"
    try:
        res = await llm_func(prompt)
        if res:
            return [k.strip() for k in res.split(",")]
    except:
        pass
    return []

async def _manual_grounding(entities: list[dict], relations: list[dict], query_text: str) -> list[dict]:
    """
    CORREÇÃO 1 & 2: Entity/Relation -> Chunks mapping.
    CORREÇÃO 7: Integridade e validação.
    """
    storage_path = Path(WORKING_DIR)
    ent_chunks_path = storage_path / "kv_store_entity_chunks.json"
    rel_chunks_path = storage_path / "kv_store_relation_chunks.json"
    text_chunks_path = storage_path / "kv_store_text_chunks.json"
    
    # Load mappings
    try:
        with open(ent_chunks_path, "r", encoding="utf-8") as f: ent_map = json.load(f)
        with open(rel_chunks_path, "r", encoding="utf-8") as f: rel_map = json.load(f)
        with open(text_chunks_path, "r", encoding="utf-8") as f: text_map = json.load(f)
    except Exception as e:
        console.print(f"[red][ERROR] Falha ao carregar storage para grounding: {e}[/red]")
        return []

    chunk_ids = set()
    
    # 1. Collect from entities
    ent_count = 0
    for ent in entities:
        name = ent.get("entity_name")
        if name in ent_map:
            cids = ent_map[name].get("chunk_ids", [])
            chunk_ids.update(cids)
            ent_count += len(cids)
            
    # 2. Collect from relations
    rel_count = 0
    for rel in relations:
        src = rel.get("src_id")
        tgt = rel.get("tgt_id")
        # LightRAG uses <SEP> for relation keys in KV storage
        rel_key = f"{src}<SEP>{tgt}"
        rel_key_rev = f"{tgt}<SEP>{src}"
        
        target_key = None
        if rel_key in rel_map: target_key = rel_key
        elif rel_key_rev in rel_map: target_key = rel_key_rev
        
        if target_key:
            cids = rel_map[target_key].get("chunk_ids", [])
            chunk_ids.update(cids)
            rel_count += len(cids)

    # 3. Fetch content and validate
    final_chunks = []
    for cid in chunk_ids:
        if cid in text_map:
            chunk_data = text_map[cid]
            content = chunk_data.get("content", "").strip()
            if content:
                final_chunks.append({
                    "chunk_id": cid,
                    "content": content,
                    "file_path": chunk_data.get("file_path", "unknown")
                })

    # CORREÇÃO 6: Logs obrigatórios
    console.print(f"[dim][DEBUG] Grounding: {len(entities)} ents -> {ent_count} chunks, {len(relations)} rels -> {rel_count} chunks[/dim]")
    
    # CORREÇÃO 4: Fallback se zero
    if not final_chunks:
        console.print("[yellow][WARN] Grounding falhou (0 chunks mapeados). Tentando Vector Fallback...[/yellow]")
        rag = await get_rag_async()
        try:
            hits = await rag.chunks_vdb.query(query_text, top_k=5)
            for h in hits:
                cid = h.get("id")
                if cid in text_map:
                    final_chunks.append({
                        "chunk_id": cid,
                        "content": text_map[cid].get("content", ""),
                        "file_path": text_map[cid].get("file_path", "unknown")
                    })
            console.print(f"[dim][DEBUG] Vector fallback: {len(final_chunks)} chunks encontrados[/dim]")
        except Exception as e:
            console.print(f"[red]Erro no fallback: {e}[/red]")

    return final_chunks

async def query(term: str, mode: str = "hybrid", lang: str = None, verbose: bool = False) -> str:
    rag = await get_rag_async()
    target_lang = lang or RESPONSE_LANGUAGE
    
    # 1. Expand query semanticamente
    expanded_query = await expand_query_semantically(term)
    
    # 2. Extract keywords
    keywords = await extract_keywords_llm(term)
    
    # 3. RAG Pipeline com Grounding Manual (CORREÇÃO 3)
    try:
        # Usamos aquery_data para obter o grafo relevante
        params = QueryParam(mode=mode, top_k=10)
        data_res = await rag.aquery_data(expanded_query, param=params)
        
        if data_res.get("status") != "success":
            return f"[ERRO] Falha na busca de dados: {data_res.get('message')}"
            
        data = data_res.get("data", {})
        entities = data.get("entities", [])
        relations = data.get("relationships", [])
        
        # Grounding manual
        grounded_chunks = await _manual_grounding(entities, relations, expanded_query)
        
        # CORREÇÃO 5: Proteção contra falso positivo
        if not grounded_chunks:
            error_msg = "[ERRO] Nenhum chunk disponível para grounding. Abortando para evitar alucinação."
            console.print(f"[red]{error_msg}[/red]")
            return error_msg
            
        # 4. Construção do Contexto (CORREÇÃO 3)
        # Deduplicação e ordenação já ocorrem no _manual_grounding
        # Aqui aplicamos limite se necessário (usaremos os primeiros 15 chunks para segurança de tokens)
        final_chunks = grounded_chunks[:15]
        
        context_str = "--- CONTEXTO DO GRAFO (ENTIDADES) ---\n"
        for ent in entities[:20]:
            context_str += f"- {ent['entity_name']} ({ent['entity_type']}): {ent['description']}\n"
            
        context_str += "\n--- CONTEXTO DO GRAFO (RELAÇÕES) ---\n"
        for rel in relations[:20]:
            context_str += f"- {rel['src_id']} -> {rel['tgt_id']}: {rel['description']}\n"
            
        context_str += "\n--- CONTEXTO DE TEXTO (CHUNKS GROUNDING) ---\n"
        for i, chunk in enumerate(final_chunks):
            context_str += f"[Chunk {i+1} from {chunk['file_path']}]:\n{chunk['content']}\n\n"
            
        # Log final
        console.print(f"[bold green]Final context: {len(entities)} entities, {len(relations)} relations, {len(final_chunks)} chunks[/bold green]")
        
        # 5. Resposta do LLM com Grounding Forçado
        system_prompt = f"Você é um assistente técnico do Kryonix. Use o contexto fornecido para responder. Se a informação não estiver no contexto, diga que não sabe. Responda em {target_lang}.\n\nCONTEXTO:\n{context_str}"
        prompt = f"Pergunta: {term}"
        
        answer = await llm_func(prompt, system_prompt=system_prompt)
        return answer
        
    except Exception as e:
        console.print(f"[red][CRITICAL ERROR] Pipeline de busca falhou: {e}[/red]")
        import traceback
        if verbose: console.print(traceback.format_exc())
        return f"Erro crítico no processamento da consulta: {e}"

async def get_query_context(term: str, mode: str = "hybrid") -> dict:
    """Retrieve raw context chunks that would be used for a query."""
    rag = await get_rag_async()
    params = QueryParam(mode=mode)
    # LightRAG internal: aquery usually calls _query
    # We can try to simulate the retrieval part
    try:
        # This is a bit of a hack since LightRAG doesn't expose a clean 'retrieve-only' method
        # But we can check the VDBs directly
        ent_hits = await rag.entities_vdb.query(term, top_k=5)
        rel_hits = await rag.relationships_vdb.query(term, top_k=5)
        chunk_hits = await rag.chunks_vdb.query(term, top_k=5)
        
        return {
            "entities": [h["entity_name"] for h in ent_hits],
            "relations": [f"{h['src_id']} -> {h['tgt_id']}" for h in rel_hits],
            "chunks": [h["id"] for h in chunk_hits],
            "counts": {
                "entities": len(ent_hits),
                "relations": len(rel_hits),
                "chunks": len(chunk_hits)
            }
        }
    except:
        return {"error": "Could not retrieve raw context"}

async def insert_single(text: str, source: str = "manual") -> None:
    rag = await get_rag_async()
    wrapped = f"SOURCE: {source}\n---\n{text}"
    deterministic_id = doc_id(f"{source}:{wrapped[:200]}")
    await rag.ainsert([wrapped], ids=[deterministic_id], file_paths=[source])

async def stats() -> dict:
    from .graph_utils import validate_graphml
    graph_file = Path(WORKING_DIR) / "graph_chunk_entity_relation.graphml"
    
    full_docs_path = Path(WORKING_DIR) / "kv_store_full_docs.json"
    full_entities_path = Path(WORKING_DIR) / "kv_store_full_entities.json"
    full_relations_path = Path(WORKING_DIR) / "kv_store_full_relations.json"
    
    def _count_json(p):
        if not p.exists(): return 0
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                return len(data)
        except: return 0
        
    def _count_relations(p):
        if not p.exists(): return 0
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Count individual pairs
                return sum(len(d.get("relation_pairs", [])) for d in data.values() if isinstance(d, dict))
        except: return 0

    kv_docs = _count_json(full_docs_path)
    kv_entities = _count_json(full_entities_path)
    kv_relations = _count_relations(full_relations_path)

    valid, err = validate_graphml(graph_file)
    if not valid:
        return {
            "error": f"GraphML inválido: {err}",
            "entities": 0,
            "relations": 0,
            "docs": kv_docs,
            "kv_full_entities": kv_entities,
            "kv_full_relations": kv_relations,
            "consistency_status": "CRITICAL_ERROR",
            "working_dir": str(WORKING_DIR)
        }

    graph = get_graph()
    g_nodes = graph.number_of_nodes() if graph else 0
    g_edges = graph.number_of_edges() if graph else 0
    
    status = "OK"
    error_msg = None
    
    if g_nodes < kv_entities:
        status = "INCONSISTENT"
        error_msg = f"Grafo tem {g_nodes} nós, mas KV tem {kv_entities} entidades."
    elif kv_relations > 0 and g_edges == 0:
        status = "INCONSISTENT"
        error_msg = f"Grafo tem 0 arestas, mas KV tem {kv_relations} relações."

    return {
        "entities": g_nodes,
        "relations": g_edges,
        "docs": kv_docs,
        "kv_full_entities": kv_entities,
        "kv_full_relations": kv_relations,
        "consistency_status": status,
        "error": error_msg,
        "working_dir": str(WORKING_DIR),
    }

async def detailed_diagnostics() -> dict:
    """Perform deep audit of the RAG pipeline grounding."""
    info = await stats()
    rag = await get_rag_async()
    G = get_graph()
    
    diag = {
        "stats": info,
        "grounding": {
            "entities_with_descriptions": 0,
            "entities_missing_descriptions": 0,
            "edges_with_descriptions": 0,
            "nodes_with_source": 0,
            "total_chunks_in_vdb": 0,
            "orphan_nodes": 0
        },
        "integrity": "OK"
    }
    
    if G:
        for n, d in G.nodes(data=True):
            if d.get("description"): diag["grounding"]["entities_with_descriptions"] += 1
            else: diag["grounding"]["entities_missing_descriptions"] += 1
            if d.get("source_id"): diag["grounding"]["nodes_with_source"] += 1
            if G.degree(n) == 0: diag["grounding"]["orphan_nodes"] += 1
            
        for _, _, d in G.edges(data=True):
            if d.get("description"): diag["grounding"]["edges_with_descriptions"] += 1
            
    # Count VDB chunks
    chunks_path = Path(WORKING_DIR) / "vdb_chunks.json"
    if chunks_path.exists():
        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                diag["grounding"]["total_chunks_in_vdb"] = len(data.get("data", []))
        except: pass
        
    if diag["grounding"]["entities_missing_descriptions"] > info["entities"] * 0.5:
        diag["integrity"] = "WARNING: Many nodes missing semantic data"
    if info["consistency_status"] != "OK":
        diag["integrity"] = "CRITICAL: Storage/Graph inconsistency"
        
    return diag

def get_graph() -> nx.Graph | None:
    from .graph_utils import validate_graphml
    graph_file = Path(WORKING_DIR) / "graph_chunk_entity_relation.graphml"
    valid, _ = validate_graphml(graph_file)
    if not valid:
        return None
        
    try: 
        return nx.read_graphml(graph_file)
    except Exception: 
        return None

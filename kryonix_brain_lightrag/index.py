"""Indexer: scan workspace files and batch-insert into LightRAG.

Modes:
  --first-run     Full index from clean storage. Aborts if storage is dirty.
  --resume        Continue a previous first-run using manifest as checkpoint.
  --full          Re-index everything (ignores manifest).
  --incremental   Re-index only files changed since last run (by SHA1).
  --retry-failed  Retry only files recorded as failed in failed_index_files.json.
  --refine        Selective reprocessing of chunks that have no entities/relations.
  --dry-run       Show what would be indexed without inserting.
  --smoke         Index 3 smallest files, run a search probe, then exit.
  --limit N       Process at most N files (useful for testing).
"""

from __future__ import annotations

import asyncio
import datetime
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import time
import networkx as nx
from pathlib import Path

from .config import (
    WORKSPACE_ROOT, INCLUDE_EXTENSIONS,
    INDEX_BATCH_SIZE, INDEX_HEARTBEAT_SECONDS,
    LLM_PROVIDER, LLM_MODEL, EMBEDDING_MODEL, WORKING_DIR,
    CHUNK_TOKEN_SIZE, EMBEDDING_BATCH_NUM, VAULT_DIR,
    FAILED_INDEX_FILE, SKIPPED_LARGE_FILES_FILE,
    INDEX_MANIFEST_FILE, INDEX_LOCK_FILE, MAX_FILE_SIZE_FIRST_RUN_KB,
    SCOPE_MODE, should_exclude_path, PROFILES, _apply_profile,
    REFINE_STATE_FILE, REFINE_REPORT_FILE,
    INDEX_REPO, INDEX_VAULT, VAULT_INCLUDE_DIRS, VAULT_EXCLUDE_DIRS
)
from .rag import get_rag_async, doc_id, get_graph
from .llm import embedding_func
from .graph_utils import validate_graphml, atomic_write_graphml

PROFILE      = os.getenv("LIGHTRAG_PROFILE", "")
PROFILE_NAME = os.getenv("LIGHTRAG_PROFILE_NAME", "safe")
VERBOSE      = os.getenv("LIGHTRAG_VERBOSE", "0") == "1"

MAX_FILE_SIZE_FIRST_RUN = MAX_FILE_SIZE_FIRST_RUN_KB * 1024


# ── Manifest ────────────────────────────────────────────────────
# Schema:
#   { "rel_path": { "doc_id", "sha1", "size", "status", "indexed_at", "elapsed_sec" }, ... }

def _load_manifest() -> dict[str, dict]:
    if INDEX_MANIFEST_FILE.exists():
        try:
            with open(INDEX_MANIFEST_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Migrate old format { path: sha1 } → new format
            migrated: dict[str, dict] = {}
            for k, v in data.items():
                if isinstance(v, str):
                    migrated[k] = {"sha1": v, "status": "done"}
                else:
                    migrated[k] = v
            return migrated
        except Exception:
            return {}
    return {}


def _save_manifest(manifest: dict[str, dict]) -> None:
    INDEX_MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _load_json_list(path: Path) -> list:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ── Storage guard ────────────────────────────────────────────────

def _check_storage_clean() -> None:
    """Abort if storage already contains data (guard for --first-run)."""
    dirty_signals = [
        WORKING_DIR / "kv_store_full_docs.json",
        WORKING_DIR / "kv_store_text_chunks.json",
        WORKING_DIR / "kv_store_doc_status.json",
        WORKING_DIR / "vdb_chunks.json",
        WORKING_DIR / "vdb_entities.json",
        WORKING_DIR / "vdb_relationships.json",
    ]
    graph_file = WORKING_DIR / "graph_chunk_entity_relation.graphml"

    for f in dirty_signals:
        if f.exists():
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if data:
                    print(
                        f"ERROR: Storage is not empty ({f.name} has {len(data)} records).\n"
                        "Run first:\n"
                        "  .\\rag.bat index --reset\n"
                        "Then retry --first-run."
                    )
                    sys.exit(1)
            except Exception:
                pass

    if graph_file.exists():
        try:
            text = graph_file.read_text(encoding="utf-8")
            if "<node " in text:
                print(
                    "ERROR: Storage is not empty (graph has existing nodes).\n"
                    "Run first:\n"
                    "  .\\rag.bat index --reset\n"
                    "Then retry --first-run."
                )
                sys.exit(1)
        except Exception:
            pass


def _ensure_storage_health() -> None:
    """Check for obvious corruption (0-byte files, invalid XML/JSON) and report."""
    if not WORKING_DIR.exists():
        return

    from .graph_utils import validate_graphml
    graph_file = WORKING_DIR / "graph_chunk_entity_relation.graphml"
    
    corrupted = []

    if graph_file.exists():
        valid, err = validate_graphml(graph_file)
        if not valid:
            corrupted.append((graph_file, err))

    # Check VDBs
    for vdb in ["vdb_entities.json", "vdb_relationships.json", "vdb_chunks.json"]:
        p = WORKING_DIR / vdb
        if p.exists():
            if p.stat().st_size == 0:
                corrupted.append((p, "Arquivo vazio (0 bytes)."))
            else:
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        json.load(f)
                except Exception as e:
                    corrupted.append((p, f"JSON inválido: {e}"))

    if corrupted:
        print("\n[CRITICAL] CORRUPÇÃO DE ARMAZENAMENTO DETECTADA!")
        for p, err in corrupted:
            print(f"  - {p.name}: {err}")
        print("\nO gráfico ou banco de vetores foi corrompido (provavelmente por interrupção brusca).")
        print("Tente recuperar usando:")
        print("  .\\rag.bat repair-graph")
        print("Ou restaure um backup manual em:")
        print(f"  {WORKING_DIR}")
        print("\nO processo será encerrado para evitar propagação da corrupção.")
        sys.exit(1)


# ── Helpers ──────────────────────────────────────────────────────

def _file_hash(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _detect_lang(ext: str) -> str:
    mapping = {
        ".nix": "nix", ".py": "python", ".sh": "bash",
        ".md": "markdown", ".json": "json", ".toml": "toml",
        ".yaml": "yaml", ".yml": "yaml", ".conf": "config",
        ".cfg": "config", ".rs": "rust", ".go": "go",
        ".ts": "typescript", ".tsx": "tsx", ".js": "javascript",
        ".jsx": "jsx",
    }
    return mapping.get(ext, "text")


def _wrap_content(rel_path: str, content: str) -> str:
    ext = os.path.splitext(rel_path)[1]
    lang = _detect_lang(ext)
    source_type = "vault" if rel_path.startswith("vault/") else "repo"
    return f"SOURCE: {source_type}\nFILE: {rel_path}\nLANG: {lang}\n---\n{content}"


def _fmt_eta(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


# ── File discovery ───────────────────────────────────────────────

def _collect_files(mode: str) -> list[tuple[str, str, int]]:
    """Return [(abs_path, rel_path, size_bytes), ...] sorted by size ascending."""
    exts = INCLUDE_EXTENSIONS.get(SCOPE_MODE, INCLUDE_EXTENSIONS["code_docs_config"])
    found: list[tuple[str, str, int]] = []

    # 1. Collect from Repo
    if INDEX_REPO:
        root = str(WORKSPACE_ROOT)
        for ext_pattern in exts:
            pattern = os.path.join(root, "**", ext_pattern)
            for abs_path in glob.glob(pattern, recursive=True):
                rel = os.path.relpath(abs_path, root).replace("\\", "/")
                if should_exclude_path(rel):
                    continue
                try:
                    size = os.path.getsize(abs_path)
                    if size > 0:
                        found.append((abs_path, f"repo/{rel}", size))
                except: pass

    # 2. Collect from Vault
    if INDEX_VAULT:
        v_root = str(VAULT_DIR)
        for inc_dir in VAULT_INCLUDE_DIRS:
            dir_path = os.path.join(v_root, inc_dir)
            if not os.path.exists(dir_path): continue
            
            for ext_pattern in ["*.md"]: # Only index markdown from vault
                pattern = os.path.join(dir_path, "**", ext_pattern)
                for abs_path in glob.glob(pattern, recursive=True):
                    rel = os.path.relpath(abs_path, v_root).replace("\\", "/")
                    
                    # Check exclusion
                    if any(rel.startswith(ex) for ex in VAULT_EXCLUDE_DIRS):
                        continue
                    
                    try:
                        size = os.path.getsize(abs_path)
                        if size > 0:
                            found.append((abs_path, f"vault/{rel}", size))
                    except: pass

    # Deduplicate
    seen: set[str] = set()
    deduped: list[tuple[str, str, int]] = []
    for item in found:
        if item[0] not in seen:
            seen.add(item[0])
            deduped.append(item)

    deduped.sort(key=lambda x: x[2])

    if mode in ("full", "dry-run", "first-run", "smoke", "resume", "scan"):
        return deduped

    if mode == "retry-failed":
        failed_list = set(_load_json_list(FAILED_INDEX_FILE))
        return [f for f in deduped if f[1] in failed_list]

    # Incremental
    manifest = _load_manifest()
    changed: list[tuple[str, str, int]] = []
    for abs_path, rel, size in deduped:
        record = manifest.get(rel, {})
        if record.get("sha1") != _file_hash(abs_path):
            changed.append((abs_path, rel, size))
    return changed


# ── Core indexer ─────────────────────────────────────────────────

async def _do_index(
    files: list[tuple[str, str, int]],
    *,
    mode: str = "first-run",
    limit: int = 0,
) -> None:
    rag = await get_rag_async()
    manifest = _load_manifest()
    skipped: list[str] = _load_json_list(SKIPPED_LARGE_FILES_FILE)
    failed: list[str] = _load_json_list(FAILED_INDEX_FILE)

    # Print config header
    if VERBOSE:
        print(f"[CONFIG] provider={LLM_PROVIDER}")
        print(f"[CONFIG] profile={PROFILE_NAME}")
        print(f"[CONFIG] llm={LLM_MODEL}")
        print(f"[CONFIG] embedding={EMBEDDING_MODEL}")
        print(f"[CONFIG] working_dir={WORKING_DIR}")
        print()

    # Build work queue
    queue: list[tuple[str, str, str, str, int]] = []  # (abs, rel, content, doc_id, size)

    for abs_path, rel, size in files:
        # Resume: skip already-done files
        if mode == "resume":
            record = manifest.get(rel, {})
            if record.get("status") == "done":
                if VERBOSE:
                    print(f"[PULADO] já indexado: {rel}")
                continue

        # first-run: skip over-sized files
        if mode in ("first-run", "smoke") and size > MAX_FILE_SIZE_FIRST_RUN:
            if rel not in skipped:
                skipped.append(rel)
            if VERBOSE:
                print(f"[PULADO] tamanho={size // 1024}KB > {MAX_FILE_SIZE_FIRST_RUN_KB}KB: {rel}")
            continue

        # Manifest dedup: skip if already done (avoid LightRAG duplicate errors)
        record = manifest.get(rel, {})
        if record.get("status") == "done":
            if VERBOSE:
                print(f"[PULADO] já indexado: {rel}")
            continue

        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                raw = fh.read()
            if not raw.strip():
                continue
            queue.append((abs_path, rel, _wrap_content(rel, raw), doc_id(rel), size))
        except Exception as e:
            if VERBOSE:
                print(f"[PULADO] erro de leitura em {rel}: {e}")

    _save_json(SKIPPED_LARGE_FILES_FILE, skipped)

    if limit > 0:
        queue = queue[:limit]

    if not queue:
        print("Nenhum arquivo para indexar.")
        return

    total = len(queue)
    done_count = 0
    fail_count = len(failed)
    t0 = time.time()
    times: list[float] = []

    for i in range(0, total, INDEX_BATCH_SIZE):
        batch = queue[i: i + INDEX_BATCH_SIZE]
        abs_paths = [b[0] for b in batch]
        fps       = [b[1] for b in batch]
        contents  = [b[2] for b in batch]
        ids       = [b[3] for b in batch]
        sizes     = [b[4] for b in batch]

        batch_num = (i // INDEX_BATCH_SIZE) + 1
        total_batches = (total + INDEX_BATCH_SIZE - 1) // INDEX_BATCH_SIZE

        # ETA calculation
        if times:
            avg = sum(times) / len(times)
            remaining_batches = total_batches - batch_num + 1
            eta_str = _fmt_eta(avg * remaining_batches)
        else:
            eta_str = "?"

        remaining = total - i
        if VERBOSE:
            print(
                f"[PROGRESSO] {i + 1}/{total} concluídos={done_count} "
                f"falhas={fail_count} pulados={len(skipped)} restante={remaining}"
            )
            for j, (fp, sz) in enumerate(zip(fps, sizes)):
                print(f"[ARQUIVO] {fp} tamanho={sz / 1024:.1f}KB")
            if times:
                print(f"[ETA] aprox={eta_str}")

        primary_file = fps[0] if fps else "unknown"
        t_batch_start = time.time()

        async def heartbeat(t_start: float, label: str) -> None:
            while True:
                await asyncio.sleep(INDEX_HEARTBEAT_SECONDS)
                elapsed_hb = int(time.time() - t_start)
                if VERBOSE:
                    print(f"[BATIDA] ainda processando {label} decorrido={elapsed_hb}s")

        hb_task = asyncio.create_task(heartbeat(t_batch_start, primary_file))

        try:
            await rag.ainsert(contents, ids=ids, file_paths=fps)

            elapsed = time.time() - t_batch_start
            times.append(elapsed)
            done_count += len(batch)

            for abs_path, rel, _, _, _ in batch:
                sha = _file_hash(abs_path)
                manifest[rel] = {
                    "doc_id": doc_id(rel),
                    "sha1": sha,
                    "size": os.path.getsize(abs_path),
                    "status": "done",
                    "indexed_at": datetime.datetime.utcnow().isoformat() + "Z",
                    "elapsed_sec": round(elapsed, 1),
                }
                # Remove from failed list if it succeeded
                if rel in failed:
                    failed.remove(rel)

            _save_manifest(manifest)
            _save_json(FAILED_INDEX_FILE, failed)

            if VERBOSE:
                for fp in fps:
                    print(f"[CONCLUÍDO] {fp} decorrido={elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - t_batch_start
            fail_count += len(batch)
            for abs_path, rel, _, _, size in batch:
                print(f"[ERRO] {rel} falhou após {elapsed:.0f}s: {e}")
                sha = _file_hash(abs_path)
                manifest[rel] = {
                    "doc_id": doc_id(rel),
                    "sha1": sha,
                    "size": size,
                    "status": "failed",
                    "indexed_at": datetime.datetime.utcnow().isoformat() + "Z",
                    "elapsed_sec": round(elapsed, 1),
                }
                if rel not in failed:
                    failed.append(rel)
            _save_manifest(manifest)
            _save_json(FAILED_INDEX_FILE, failed)

        finally:
            hb_task.cancel()

    total_elapsed = time.time() - t0
    print(
        f"\nIndexação concluída. "
        f"concluídos={done_count} falhas={fail_count} pulados={len(skipped)} "
        f"tempo_total={_fmt_eta(total_elapsed)}"
    )


# ── Refine / Repair ─────────────────────────────────────────────

def _is_useful_content(text: str, min_chars: int = 100) -> tuple[bool, str]:
    if not text or not text.strip():
        return False, "empty"
    if len(text) < min_chars:
        return False, "too_short"
    if not re.search(r'[a-zA-Z]', text):
        return False, "no_letters"
    # Boilerplate detection (enhanced)
    # If more than 20% of chars are brackets/braces, it's likely noise/config
    symbol_count = text.count("{") + text.count("}") + text.count("[") + text.count("]")
    if symbol_count > len(text) * 0.2:
        return False, "boilerplate_symbols"
    
    # Path repetition or mostly paths
    if text.count("/") > len(text) * 0.1 or text.count("\\") > len(text) * 0.1:
        if len(re.findall(r'[a-zA-Z]{4,}', text)) < 3: # Few real words
            return False, "mostly_paths"

    return True, "ok"


def _recover_source(cid: str, chunks_kv: dict, full_docs: dict, manifest: dict) -> str:
    chunk_data = chunks_kv.get(cid, {})
    content = ""
    if isinstance(chunk_data, dict):
        content = chunk_data.get("content", "")
    elif isinstance(chunk_data, str):
        content = chunk_data

    # 1. content line
    if content:
        m = re.search(r'^FILE: (.*)$', content, re.M)
        if m: return m.group(1).strip()

    # 2. Try manifest/full_docs
    return "unknown_source"


async def _do_refine(
    limit: int = 0,
    only_useful: bool = False,
    known_source_only: bool = False,
    min_chars: int = 100,
) -> None:
    """Identifica chunks com 0 entidades/relações e re-executa a extração."""
    print(f"[REFINE] Iniciando refinamento (limit={limit}, only_useful={only_useful}, min_chars={min_chars})")
    rag = await get_rag_async()
    
    # Load state
    state = _load_json(REFINE_STATE_FILE)
    if "chunks" not in state: state["chunks"] = {}

    chunks_kv = _load_json(WORKING_DIR / "kv_store_text_chunks.json")
    entity_chunks = _load_json(WORKING_DIR / "kv_store_entity_chunks.json")
    relation_chunks = _load_json(WORKING_DIR / "kv_store_relation_chunks.json")
    
    if not chunks_kv:
        print("[REFINE] Nenhum chunk encontrado no armazenamento.")
        return

    # Find chunks with data
    chunks_with_data: set[str] = set()
    for chunk_ids in entity_chunks.values():
        if isinstance(chunk_ids, list):
            chunks_with_data.update(chunk_ids)
    for chunk_ids in relation_chunks.values():
        if isinstance(chunk_ids, list):
            chunks_with_data.update(chunk_ids)
            
    all_chunk_ids = set(chunks_kv.keys())
    candidates = list(all_chunk_ids - chunks_with_data)
    
    report = {
        "total_candidates": len(candidates),
        "total_processed": 0,
        "total_skipped": 0,
        "total_zero_extract": 0,
        "total_success": 0,
        "total_errors": 0,
        "skipped_reasons": {},
        "top_bad_sources": {},
        "duration_sec": 0,
        "started_at": datetime.datetime.now().isoformat()
    }

    def _inc_reason(r):
        report["skipped_reasons"][r] = report["skipped_reasons"].get(r, 0) + 1

    print(f"[REFINE] Encontrados {len(candidates)} chunks candidatos.")

    queue: list[tuple[str, str, str, int]] = [] # (cid, content, source, length)
    
    for cid in candidates:
        chunk_data = chunks_kv[cid]
        content = chunk_data.get("content", "") if isinstance(chunk_data, dict) else str(chunk_data)
        source = _recover_source(cid, chunks_kv, {}, {})
        
        # Check state
        cstate = state["chunks"].get(cid, {"attempts": 0})
        if cstate.get("attempts", 0) >= 2:
            report["total_skipped"] += 1
            _inc_reason("max_attempts_reached")
            continue

        # Known source filter
        if known_source_only and source == "unknown_source":
            report["total_skipped"] += 1
            _inc_reason("filter_known_source_only")
            continue

        # Filtering
        useful, reason = _is_useful_content(content, min_chars)
        
        # Advanced unknown_source logic
        if source == "unknown_source" and only_useful:
            # Rule: skip unknown_source if len < 500 unless has FILE: or technical content
            if len(content) < 500 and "FILE:" not in content:
                # Basic technical content check: does it have many real words?
                if len(re.findall(r'[a-zA-Z]{4,}', content)) < 10:
                    report["total_skipped"] += 1
                    _inc_reason("unknown_source_low_value")
                    continue

        if only_useful and not useful:
            report["total_skipped"] += 1
            _inc_reason(f"filter_{reason}")
            if VERBOSE:
                print(f"[REFINE] Pulo: {cid} motivo={reason} source={source}")
            continue

        queue.append((cid, content, source, len(content)))

    # Sort queue: Known source first, then by length DESC, then unknown_source
    queue.sort(key=lambda x: (x[2] != "unknown_source", x[3]), reverse=True)

    print(f"[REFINE] Fila de processamento: {len(queue)} chunks após filtros.")
    
    if limit > 0:
        queue = queue[:limit]
        print(f"[REFINE] Limitando a {limit} chunks.")

    if not queue:
        print("[REFINE] Nada para processar.")
        _save_json(REFINE_REPORT_FILE, report)
        return

    t0 = time.time()
    
    for i, (cid, content, source, clen) in enumerate(queue):
        print(f"[REFINE] Processando {i+1}/{len(queue)} source={source} cid={cid[:8]} len={clen}")
        
        cstate = state["chunks"].get(cid, {"attempts": 0, "source": source})
        cstate["attempts"] += 1
        cstate["updated_at"] = datetime.datetime.now().isoformat()
        
        try:
            # Wrap content with source for better extraction
            wrapped = f"SOURCE: {source}\n---\n{content}" if "FILE:" not in content else content
            new_doc_id = f"refine-{cid[:10]}"
            
            await rag.ainsert([wrapped], ids=[new_doc_id])
            
            # Verify extraction results
            # Reload storage to check new ID
            tmp_chunks = _load_json(WORKING_DIR / "kv_store_text_chunks.json")
            tmp_entities = _load_json(WORKING_DIR / "kv_store_entity_chunks.json")
            tmp_relations = _load_json(WORKING_DIR / "kv_store_relation_chunks.json")
            
            n_ent, n_rel = _get_doc_extraction_stats(new_doc_id, tmp_chunks, tmp_entities, tmp_relations)
            
            if n_ent > 0 or n_rel > 0:
                cstate["last_result"] = "ok"
                report["total_success"] += 1
                if VERBOSE:
                    print(f"[REFINE] Sucesso: extraído {n_ent} entidades, {n_rel} relações")
            else:
                cstate["last_result"] = "zero_extract"
                report["total_zero_extract"] += 1
                print(f"[REFINE] zero_extract source={source} cid={cid[:8]} ent=0 rel=0 attempts={cstate['attempts']}")
            
        except Exception as e:
            print(f"[REFINE] Erro: {e}")
            cstate["last_result"] = "error"
            cstate["last_reason"] = str(e)
            report["total_errors"] += 1
            
        state["chunks"][cid] = cstate
        report["total_processed"] += 1
        
        # Save every 5 chunks to avoid data loss
        if (i + 1) % 5 == 0:
            _save_json(REFINE_STATE_FILE, state)

    report["duration_sec"] = round(time.time() - t0, 1)
    report["finished_at"] = datetime.datetime.now().isoformat()
    
    _save_json(REFINE_STATE_FILE, state)
    _save_json(REFINE_REPORT_FILE, report)
    
    print(f"[REFINE] Concluído. Processados={report['total_processed']} Sucesso={report['total_success']} Zero={report['total_zero_extract']} Erros={report['total_errors']}")


def _get_doc_extraction_stats(doc_id: str, chunks_kv: dict, entity_chunks: dict, relation_chunks: dict) -> tuple[int, int]:
    """Retorna contagem de entidades e relações vinculadas a um documento."""
    # Find all chunks belonging to this doc
    doc_chunks = []
    for cid, data in chunks_kv.items():
        if isinstance(data, dict) and data.get("full_doc_id") == doc_id:
            doc_chunks.append(cid)
    
    if not doc_chunks:
        return 0, 0

    num_entities = 0
    num_relations = 0
    
    # Check entities
    doc_chunks_set = set(doc_chunks)
    for cids in entity_chunks.values():
        if not isinstance(cids, list): continue
        if any(cid in doc_chunks_set for cid in cids):
            num_entities += 1
                
    # Check relations
    for cids in relation_chunks.values():
        if not isinstance(cids, list): continue
        if any(cid in doc_chunks_set for cid in cids):
            num_relations += 1
                
    return num_entities, num_relations


def _show_refine_report():
    report = _load_json(REFINE_REPORT_FILE)
    if not report:
        print("Relatório não encontrado.")
        return
    print("\n=== LIGHTRAG REFINE REPORT ===")
    for k, v in report.items():
        if k != "skipped_reasons" and k != "top_bad_sources":
            print(f"{k}: {v}")
    if report.get("skipped_reasons"):
        print("\nMotivos de pulo:")
        for r, count in report["skipped_reasons"].items():
            print(f"  - {r}: {count}")
    print("==============================\n")


def cmd_reset_refine_state() -> None:
    if REFINE_STATE_FILE.exists():
        REFINE_STATE_FILE.unlink()
        print("Refine state reset.")
    else:
        print("No refine state to reset.")


def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


# ── Smoke test ───────────────────────────────────────────────────

async def _do_smoke() -> None:
    """Index the 3 smallest valid files, probe embedding + LLM, then exit."""
    print("[SMOKE] Starting smoke test...")

    files = _collect_files("smoke")
    if not files:
        print("[SMOKE] FAIL: no files found to index.")
        sys.exit(1)

    candidates = files[:3]
    print(f"[SMOKE] Indexing {len(candidates)} smallest files:")
    for _, rel, size in candidates:
        print(f"  {rel} ({size} bytes)")

    await _do_index(candidates, mode="smoke", limit=0)

    # Probe: stats
    from . import rag as rag_mod
    s = await rag_mod.stats()
    print(f"[SMOKE] Stats: entities={s['entities']} relations={s['relations']} docs={s['docs']}")

    # Probe: LLM + embedding via query
    try:
        ans = await rag_mod.query("What is this project about?", mode="naive")
        if ans and len(ans) > 5:
            print(f"[SMOKE] LLM query OK: {ans[:80]}...")
        else:
            print(f"[SMOKE] WARN: short LLM response: {ans!r}")
    except Exception as e:
        print(f"[SMOKE] FAIL: query error: {e}")
        sys.exit(1)

    print("[SMOKE] PASS ✓")


# ── Reset / clean ─────────────────────────────────────────────────

def cmd_reset() -> None:
    if WORKING_DIR.exists():
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = BRAIN_HOME / "backups" / f"rag_storage.failed-{ts}"
        try:
            shutil.move(str(WORKING_DIR), str(backup))
            print(f"Storage archived to {backup}")
        except Exception as e:
            print(f"Failed to archive storage: {e}")
            # Force-delete if move failed
            try:
                shutil.rmtree(str(WORKING_DIR), ignore_errors=True)
            except Exception:
                pass

    # Ensure the directory is created fresh and truly empty
    WORKING_DIR.mkdir(parents=True, exist_ok=True)

    # Defensive: wipe any stray KV files that may have been recreated immediately
    _KV_FILES = [
        "kv_store_full_docs.json", "kv_store_text_chunks.json",
        "kv_store_doc_status.json", "kv_store_full_entities.json",
        "kv_store_full_relations.json", "kv_store_entity_chunks.json",
        "kv_store_relation_chunks.json", "kv_store_llm_response_cache.json",
        "vdb_entities.json", "vdb_relationships.json", "vdb_chunks.json",
        "graph_chunk_entity_relation.graphml",
    ]
    for fname in _KV_FILES:
        fp = WORKING_DIR / fname
        if fp.exists():
            try:
                fp.unlink()
            except Exception:
                pass

    cmd_clean_state()
    print("Storage reset successfully.")


def cmd_clean_state() -> None:
    for f in [INDEX_MANIFEST_FILE, FAILED_INDEX_FILE, SKIPPED_LARGE_FILES_FILE, INDEX_LOCK_FILE]:
        if f.exists():
            try:
                f.unlink()
            except Exception:
                pass
    print("State cleaned successfully.")


# ── Entry point ───────────────────────────────────────────────────

async def cmd_repair_vdb() -> None:
    """Reconstruct vdb_entities.json from graph data."""
    print("[REPAIR] Starting vdb_entities reconstruction...", flush=True)
    
    G = get_graph()
    
    if G is None:
        print("[REPAIR] FAIL: Graph not loaded from disk.", flush=True)
        return

    vdb_path = WORKING_DIR / "vdb_entities.json"
    if vdb_path.exists():
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = vdb_path.with_suffix(f".corrupted-{ts}.json")
        vdb_path.rename(backup)
        print(f"[REPAIR] Moved corrupted VDB to {backup.name}", flush=True)

    rag = await get_rag_async()
    entities = []
    # Collect entities from graph
    for node, data in G.nodes(data=True):
        if data.get("entity_type"): # It's an entity
            entities.append({
                "entity_name": node,
                "description": data.get("description", ""),
                "entity_type": data.get("entity_type", "")
            })
            
    if not entities:
        print("[REPAIR] No entities found in graph. Nothing to repair.", flush=True)
        return

    print(f"[REPAIR] Reconstructing {len(entities)} entities...", flush=True)
    
    # Batch process embeddings
    batch_size = 16
    total = len(entities)
    reconstructed = 0
    
    for i in range(0, total, batch_size):
        batch = entities[i : i + batch_size]
        # Format for embedding: Name + Description
        texts = [f"{e['entity_name']} {e['description']}" for e in batch]
        
        try:
            print(f"[REPAIR] Batch {i//batch_size + 1}: Embedding {len(batch)} entities...", flush=True)
            # Use direct embedding func to bypass LightRAG worker queue
            embeddings = await embedding_func(texts)
            
            vdb_data = {}
            for j, e in enumerate(batch):
                vdb_data[e["entity_name"]] = {
                    "entity_name": e["entity_name"],
                    "content": e["description"],
                    "vector": embeddings[j].tolist()
                }
            
            # Upsert into NanoVectorDB
            # rag.entities_vdb is a NanoVectorDB instance
            print(f"[REPAIR] Batch {i//batch_size + 1}: Upserting...", flush=True)
            await rag.entities_vdb.upsert(vdb_data)
            reconstructed += len(batch)
            if reconstructed % 100 == 0 or reconstructed == total:
                print(f"[REPAIR] Progress: {reconstructed}/{total} entities.", flush=True)
            
        except Exception as e:
            print(f"[REPAIR] Error processing batch at index {i}: {e}", flush=True)
            
    # Save the VDB
    try:
        # rag.entities_vdb is a NanoVectorDBStorage wrapper
        # We need to get the actual NanoVectorDB client to call save()
        client = await rag.entities_vdb._get_client()
        client.save()
        print(f"[REPAIR] Successfully reconstructed {reconstructed} entities in vdb_entities.json.", flush=True)
    except Exception as e:
        print(f"[REPAIR] Failed to save vdb_entities.json: {e}", flush=True)


async def cmd_repair_graph(rebuild: bool = False) -> None:
    """Repair or reconstruct graph_chunk_entity_relation.graphml."""
    target = WORKING_DIR / "graph_chunk_entity_relation.graphml"
    print(f"[REPAIR-GRAPH] Alvo: {target}", flush=True)
    if rebuild:
        print("[REPAIR-GRAPH] Modo --rebuild ATIVADO. Ignorando backups e reconstruindo via KV...", flush=True)

    # 1. Backup current if exists
    if target.exists():
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = target.with_suffix(f".failed-{ts}.graphml")
        shutil.copy2(target, backup)
        print(f"[REPAIR-GRAPH] Backup do atual: {backup.name}")

    # Load KV counts for validation
    ent_kv = _load_json(WORKING_DIR / "kv_store_full_entities.json")
    rel_kv = _load_json(WORKING_DIR / "kv_store_full_relations.json")
    kv_nodes_count = len(ent_kv)
    kv_rels_count = sum(len(d.get("relation_pairs", [])) for d in rel_kv.values() if isinstance(d, dict))

    # 2. Search for valid backups in rag_storage and 11-LightRAG
    best_backup = None
    if not rebuild:
        backup_dirs = [WORKING_DIR, config.BRAIN_HOME / "backups"]
        all_backups = []
        for d in backup_dirs:
            if not d.exists(): continue
            all_backups.extend(list(d.glob("*.bak-*.graphml")))
            all_backups.extend(list(d.glob("*.graphml.bak-*")))
        
        max_score = -1
        print(f"[REPAIR-GRAPH] Procurando backups em {len(backup_dirs)} diretórios...")
        for b in all_backups:
            valid, err = validate_graphml(b)
            if valid:
                try:
                    G_tmp = nx.read_graphml(b)
                    nodes = len(G_tmp.nodes)
                    edges = len(G_tmp.edges)
                    # Score: edges are more important than nodes
                    score = (edges * 10) + nodes
                    
                    # Semantic sanity: if we have relations in KV, backup must have edges
                    if kv_rels_count > 0 and edges == 0:
                        score = -1 # Discard
                        
                    if score > max_score:
                        max_score = score
                        best_backup = b
                except: pass

    if best_backup:
        print(f"[REPAIR-GRAPH] Restaurando do melhor backup: {best_backup.name}")
        shutil.copy2(best_backup, target)
        print("[REPAIR-GRAPH] Restauração concluída.")
        return

    # 3. Reconstruct from VDB and KV stores
    print("[REPAIR-GRAPH] Iniciando reconstrução a partir dos arquivos VDB (Fonte de Verdade)...", flush=True)
    
    vdb_entities_path = WORKING_DIR / "vdb_entities.json"
    vdb_relations_path = WORKING_DIR / "vdb_relationships.json"
    
    if not vdb_entities_path.exists():
        print(f"[REPAIR-GRAPH] FAIL: {vdb_entities_path.name} não encontrado.")
        sys.exit(1)

    try:
        with open(vdb_entities_path, "r", encoding="utf-8") as f:
            ent_vdb = json.load(f).get("data", [])
        with open(vdb_relations_path, "r", encoding="utf-8") as f:
            rel_vdb = json.load(f).get("data", [])
    except Exception as e:
        print(f"[REPAIR-GRAPH] FAIL ao ler VDB: {e}")
        sys.exit(1)

    if not ent_vdb:
        print("[REPAIR-GRAPH] FAIL: VDB de entidades vazio.")
        sys.exit(1)

    # Load mapping of entities/relations to chunks
    ent_chunks = _load_json(WORKING_DIR / "kv_store_entity_chunks.json")
    rel_chunks = _load_json(WORKING_DIR / "kv_store_relation_chunks.json")

    G = nx.Graph()
    
    # Add entities (Nodes)
    print(f"[REPAIR-GRAPH] Processando {len(ent_vdb)} entidades...")
    for item in ent_vdb:
        name = item.get("entity_name")
        if not name: continue
        
        content = item.get("content", "Recuperado via repair-graph")
        etype = "UNKNOWN"
        if "Type:" in content:
            m = re.search(r'Type:\s*([^\n<]+)', content)
            if m: etype = m.group(1).strip()
            
        # Get chunks for this entity
        cids = []
        if name in ent_chunks:
            record = ent_chunks[name]
            if isinstance(record, dict):
                cids = record.get("chunk_ids", [])
            elif isinstance(record, list):
                cids = record
        
        source_id_str = "<SEP>".join(cids)
        
        G.add_node(
            name,
            entity_type=etype,
            description=content,
            source_id=source_id_str,
            clusters="",
        )
            
    # Add relations (Edges)
    print(f"[REPAIR-GRAPH] Processando {len(rel_vdb)} relações...")
    for item in rel_vdb:
        src = item.get("src_id")
        tgt = item.get("tgt_id")
        content = item.get("content", "Relação recuperada")
        
        if src and tgt:
            if src not in G: 
                G.add_node(src, entity_type="UNKNOWN", description="No description (ghost node)", source_id="")
            if tgt not in G: 
                G.add_node(tgt, entity_type="UNKNOWN", description="No description (ghost node)", source_id="")
            
            # Get chunks for this relation
            rid = f"{src} -> {tgt}"
            cids = []
            if rid in rel_chunks:
                record = rel_chunks[rid]
                if isinstance(record, dict):
                    cids = record.get("chunk_ids", [])
                elif isinstance(record, list):
                    cids = record

            source_id_str = "<SEP>".join(cids)

            G.add_edge(
                src, tgt,
                weight=1.0,
                description=content,
                keywords="recovered",
                source_id=source_id_str,
            )

    print(f"[REPAIR-GRAPH] Grafo reconstruído: {len(G.nodes)} nós, {len(G.edges)} arestas.")
    if len(G.nodes) > 0:
        atomic_write_graphml(G, target)
        print("[REPAIR-GRAPH] Sucesso na reconstrução semântica.")
    else:
        print("[REPAIR-GRAPH] FAIL: Nenhum nó reconstruído.")
        sys.exit(1)


async def cmd_index(args_obj=None) -> None:
    """Main entry point for indexer logic."""
    if args_obj is None:
        # Minimal object to simulate args
        class Args:
            def __init__(self):
                self.args = sys.argv[1:]
                self.first_run = "--first-run" in self.args
                self.resume = "--resume" in self.args
                self.full = "--full" in self.args
                self.smoke = "--smoke" in self.args
                self.retry_failed = "--retry-failed" in self.args
                self.refine = "--refine" in self.args
                self.repair_vdb = "--repair-vdb" in self.args
                self.reset = "--reset" in self.args
                self.clean_state = "--clean-state" in self.args
                self.dry_run = "--dry-run" in self.args
                self.only_useful = "--only-useful" in self.args
                self.known_source_only = "--known-source-only" in self.args
                self.reset_refine_state = "--reset-refine-state" in self.args
                self.report = "--report" in self.args
                self.limit = 0
                if "--limit" in self.args:
                    idx = self.args.index("--limit")
                    try: self.limit = int(self.args[idx+1])
                    except: pass
                self.min_chars = 100
                if "--min-chars" in self.args:
                    idx = self.args.index("--min-chars")
                    try: self.min_chars = int(self.args[idx+1])
                    except: pass
        args_obj = Args()

    def get_arg(name, default=False):
        return getattr(args_obj, name, default)

    if get_arg("reset"):
        cmd_reset()
        return
    if get_arg("clean_state"):
        cmd_clean_state()
        return
    if get_arg("reset_refine_state"):
        cmd_reset_refine_state()
        return
    if get_arg("report"):
        _show_refine_report()
        return
    if get_arg("repair_vdb"):
        await cmd_repair_vdb()
        return

    # Pre-flight health check
    _ensure_storage_health()

    if INDEX_LOCK_FILE.exists():
        stale = False
        try:
            content = INDEX_LOCK_FILE.read_text().strip()
            if not content:
                # If empty, check if it's been there for more than 10 seconds
                if time.time() - os.path.getmtime(INDEX_LOCK_FILE) > 10:
                    stale = True
            else:
                pid = int(content)
                if pid != os.getpid():
                    try:
                        os.kill(pid, 0)
                    except (OSError, ProcessLookupError):
                        stale = True
        except Exception:
            # Fallback for any read error
            if time.time() - os.path.getmtime(INDEX_LOCK_FILE) > 1800:
                stale = True

        if stale:
            print(f"INFO: Removing stale lock file ({INDEX_LOCK_FILE.name}).")
            INDEX_LOCK_FILE.unlink()
        else:
            print("ERROR: Indexer is already running (locked).")
            return

    try:
        # Write current PID to lock file
        INDEX_LOCK_FILE.write_text(str(os.getpid()))
        mode = "incremental"
        if get_arg("first_run"): mode = "first-run"
        elif get_arg("resume"): mode = "resume"
        elif get_arg("full"): mode = "full"
        elif get_arg("retry_failed"): mode = "retry-failed"
        elif get_arg("refine"): mode = "refine"
        elif get_arg("dry_run"): mode = "dry-run"
        elif get_arg("smoke"): mode = "smoke"

        limit = get_arg("limit", 0)
        only_useful = get_arg("only_useful")
        known_source_only = get_arg("known_source_only")
        min_chars = get_arg("min_chars", 100)

        if mode == "first-run":
            _check_storage_clean()

        files = _collect_files(mode)

        if mode == "dry-run":
            print(f"[SIMULAÇÃO] Indexaria {len(files)} arquivos:")
            for _, rel, size in files[:30]:
                print(f"  {rel} ({size / 1024:.1f}KB)")
            return

        if mode == "smoke":
            await _do_smoke()
            return

        if mode == "refine":
            await _do_refine(limit=limit, only_useful=only_useful, known_source_only=known_source_only, min_chars=min_chars)
            return

        if not files and mode not in ("resume", "refine"):
            print("Nada para indexar (todos os arquivos estão atualizados).")
            return

        await _do_index(files, mode=mode, limit=limit)

    finally:
        if INDEX_LOCK_FILE.exists():
            INDEX_LOCK_FILE.unlink()

def cmd_vault_scan():
    """List vault files to be indexed."""
    print("Verificando discovery do vault...")
    files = _collect_files("scan")
    vault_files = [f for f in files if f[1].startswith("vault/")]
    
    for _, rel, size in vault_files:
        print(f"  [VAULT] {rel} ({size/1024:.1f} KB)")
    
    print(f"\nTotal de arquivos do vault encontrados: {len(vault_files)}")

def main():
    asyncio.run(cmd_index())


if __name__ == "__main__":
    main()

from __future__ import annotations

import fnmatch
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── OS-aware defaults ────────────────────────────────────────────
_default_workspace = os.path.expanduser("~/.local/share/kryonix/brain")
_default_vault = "/home/rocha/.local/share/kryonix/kryonix-vault"
_default_storage_subdir = "storage"
_default_export_subdir = "exports"
_default_refine_subdir = "storage"

# ── Paths ────────────────────────────────────────────────────────
WORKSPACE_ROOT = Path(os.getenv("LIGHTRAG_WORKSPACE_ROOT", _default_workspace))
PROJECT_DIR = Path(os.getenv("KRYONIX_REPO_ROOT", "/etc/kryonix"))
VAULT_DIR = Path(os.getenv("LIGHTRAG_VAULT_DIR", _default_vault))
WORKING_DIR = Path(os.getenv("LIGHTRAG_WORKING_DIR", str(VAULT_DIR / _default_storage_subdir)))
OBSIDIAN_EXPORT_DIR = Path(os.getenv("LIGHTRAG_OBSIDIAN_EXPORT_DIR", str(VAULT_DIR / _default_export_subdir)))
REFINE_STATE_FILE = Path(os.getenv("LIGHTRAG_REFINE_STATE_FILE", str(VAULT_DIR / _default_refine_subdir / "refine_state.json")))
REFINE_REPORT_FILE = Path(os.getenv("LIGHTRAG_REFINE_REPORT_FILE", str(VAULT_DIR / _default_refine_subdir / "refine_report.json")))

# ── Ollama ───────────────────────────────────────────────────────
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "0")

# ── Ingestion Queue ──────────────────────────────────────────────
# Pipeline de ingestão controlada: propose → queue → approve → indexar
INGEST_QUEUE_DIR = Path(os.getenv("LIGHTRAG_INGEST_QUEUE_DIR", str(WORKING_DIR.parent / "ingest_queue")))

# ── Indexing Sources ─────────────────────────────────────────────
INDEX_REPO = os.getenv("LIGHTRAG_INDEX_REPO", "true").lower() == "true"
INDEX_VAULT = os.getenv("LIGHTRAG_INDEX_VAULT", "true").lower() == "true"

VAULT_INCLUDE_DIRS = [
    "00-System",
    "01-MOCs",
    "02-Areas",
    "03-Projetos",
    "06-Playbooks",
    "07-Prompts",
]

VAULT_EXCLUDE_DIRS = [
    "11-LightRAG/rag_storage",
    ".backups",
    ".obsidian",
    ".git",
    ".sync",
    "node_modules"
]

# ── Provider ─────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LIGHTRAG_LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# ── Language & Prompt ─────────────────────────────────────────────
RESPONSE_LANGUAGE = os.getenv("RESPONSE_LANGUAGE", "pt-BR")
ANSWER_SYSTEM_PROMPT = os.getenv("ANSWER_SYSTEM_PROMPT", """
Você é um assistente técnico local do projeto Kryonix.
Responda sempre em português do Brasil.
Use somente o contexto recuperado quando a ferramenta exigir RAG.
Não invente.
Se o contexto for insuficiente, diga isso em português.
Preserve nomes técnicos, comandos, paths, arquivos, módulos Nix e código no original.
""").strip()

# ── Profiles ─────────────────────────────────────────────────────
# safe     → qwen2.5-coder:3b  — RTX 4060 8GB, first-run stable
# balanced → qwen2.5-coder:7b  — faster extraction, slightly heavier
# query    → qwen2.5-coder:7b  — used for interactive queries after indexing
PROFILES: dict[str, dict] = {
    "safe": {
        "llm_model": "qwen2.5-coder:3b",
        "llm_model_max_async": 1,
        "max_parallel_insert": 1,
        "embedding_batch_num": 1,
        "chunk_token_size": 350,
        "chunk_overlap_token_size": 50,
        "index_batch_size": 1,
    },
    "balanced": {
        "llm_model": "qwen2.5-coder:7b",
        "llm_model_max_async": 1,
        "max_parallel_insert": 1,
        "embedding_batch_num": 1,
        "chunk_token_size": 500,
        "chunk_overlap_token_size": 50,
        "index_batch_size": 1,
    },
    "query": {
        "llm_model": "qwen2.5-coder:7b",
        "llm_model_max_async": 2,
        "max_parallel_insert": 1,
        "embedding_batch_num": 2,
        "chunk_token_size": 500,
        "chunk_overlap_token_size": 50,
        "index_batch_size": 1,
    },
    "quality": {
        "llm_model": "qwen2.5-coder:7b",
        "llm_model_max_async": 1,
        "max_parallel_insert": 1,
        "embedding_batch_num": 1,
        "chunk_token_size": 500,
        "chunk_overlap_token_size": 50,
        "index_batch_size": 1,
    },
}

def _apply_profile(name: str | None) -> dict:
    """Return profile settings, falling back to 'safe'."""
    return PROFILES.get(name or "", PROFILES["safe"])

# Active profile — resolved at import time from env
_active_profile_name = os.getenv("LIGHTRAG_PROFILE_NAME", "balanced")
_p = _apply_profile(_active_profile_name)

LLM_MODEL               = os.getenv("LIGHTRAG_LLM_MODEL",            _p["llm_model"])
EMBEDDING_MODEL         = os.getenv("LIGHTRAG_EMBED_MODEL",           "nomic-embed-text:latest")
LLM_MODEL_MAX_ASYNC     = int(os.getenv("LIGHTRAG_LLM_MODEL_MAX_ASYNC",  str(_p["llm_model_max_async"])))
MAX_PARALLEL_INSERT     = int(os.getenv("LIGHTRAG_MAX_PARALLEL_INSERT",   str(_p["max_parallel_insert"])))
EMBEDDING_BATCH_NUM     = int(os.getenv("LIGHTRAG_EMBEDDING_BATCH_NUM",   str(_p["embedding_batch_num"])))
CHUNK_TOKEN_SIZE        = int(os.getenv("LIGHTRAG_CHUNK_TOKEN_SIZE",      str(_p["chunk_token_size"])))
CHUNK_OVERLAP_TOKEN_SIZE= int(os.getenv("LIGHTRAG_CHUNK_OVERLAP_TOKEN_SIZE", str(_p["chunk_overlap_token_size"])))
INDEX_BATCH_SIZE        = int(os.getenv("LIGHTRAG_INDEX_BATCH_SIZE",      str(_p["index_batch_size"])))
INDEX_HEARTBEAT_SECONDS = int(os.getenv("LIGHTRAG_INDEX_HEARTBEAT_SECONDS", "15"))
MAX_FILE_SIZE_FIRST_RUN_KB = int(os.getenv("LIGHTRAG_MAX_FILE_SIZE_FIRST_RUN_KB", "250"))

PROFILE = os.getenv("LIGHTRAG_PROFILE", "")
VERBOSE = os.getenv("LIGHTRAG_VERBOSE", "0") == "1"

# ── State files ───────────────────────────────────────────────────
FAILED_INDEX_FILE       = Path(os.getenv("LIGHTRAG_FAILED_INDEX_FILE",
                               str(VAULT_DIR / "11-LightRAG" / "failed_index_files.json")))
SKIPPED_LARGE_FILES_FILE= Path(os.getenv("LIGHTRAG_SKIPPED_LARGE_FILES_FILE",
                               str(VAULT_DIR / "11-LightRAG" / "skipped_large_files.json")))
INDEX_MANIFEST_FILE     = Path(os.getenv("LIGHTRAG_INDEX_MANIFEST_FILE",
                               str(VAULT_DIR / "11-LightRAG" / ".index_manifest.json")))
INDEX_LOCK_FILE         = Path(os.getenv("LIGHTRAG_INDEX_LOCK_FILE",
                               str(VAULT_DIR / "11-LightRAG" / ".index.lock")))

WORKING_DIR.mkdir(parents=True, exist_ok=True)
OBSIDIAN_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Backward-compatible aliases ───────────────────────────────────
STORAGE_DIR   = WORKING_DIR
VAULT_PATH    = OBSIDIAN_EXPORT_DIR
MANIFEST_PATH = INDEX_MANIFEST_FILE
SCOPE_MODE    = os.getenv("LIGHTRAG_SCOPE_MODE", "code_docs_config")

# ── Discovery rules ───────────────────────────────────────────────
INCLUDE_EXTENSIONS: dict[str, list[str]] = {
    "code_docs_config": [
        "*.nix", "*.py", "*.md", "*.toml", "*.yaml", "*.yml",
        "*.json", "*.sh", "*.rs", "*.go", "*.ts", "*.tsx",
        "*.js", "*.jsx", "*.conf", "*.cfg",
    ],
    "docs_only": ["*.md"],
}

EXCLUDE_DIRS: set[str] = {
    ".git", "node_modules", ".next", "dist", "build", "target",
    "__pycache__", ".venv", "rag_storage", "obsidian-export",
}

EXCLUDE_EXTS: set[str] = {
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp", "*.ico",
    "*.pdf", "*.zip", "*.7z", "*.tar", "*.gz",
    "*.exe", "*.dll", "*.so", "*.bin",
    "*.lock",
}

EXCLUDE_FILES: set[str] = {
    ".env",
    "flake.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "Cargo.lock",
    "uv.lock",
}

# Entire path prefixes that must be excluded
_EXCLUDE_PATH_PREFIXES: tuple[str, ...] = (
    "packages/kryonix-brain-lightrag/",
)


def should_exclude_path(rel_path: str) -> bool:
    norm = rel_path.replace("\\", "/")
    parts = norm.split("/")
    filename = parts[-1]

    # Special case for vault files (since they are prefixed with vault/ in indexer)
    if norm.startswith("vault/"):
        vault_rel = norm[6:]
        # Exclude internal vault dirs
        for ex in VAULT_EXCLUDE_DIRS:
            if vault_rel.startswith(ex):
                return True
        # Only allow markdown in vault
        if not filename.endswith(".md"):
            return True
        return False

    # Repo logic
    # Exclude entire subtrees by prefix
    for prefix in _EXCLUDE_PATH_PREFIXES:
        if norm.startswith(prefix) or norm == prefix.rstrip("/"):
            return True

    if any(part in EXCLUDE_DIRS for part in parts):
        return True

    if filename in EXCLUDE_FILES or filename.startswith(".env."):
        return True

    if any(fnmatch.fnmatch(filename, ext) for ext in EXCLUDE_EXTS):
        return True

    return False

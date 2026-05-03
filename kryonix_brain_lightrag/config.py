from __future__ import annotations

import fnmatch
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── OS-aware defaults ────────────────────────────────────────────
_default_brain_home = "/home/rocha/.local/share/kryonix/kryonix-vault"
_default_workspace = "/etc/kryonix"
_default_vault = f"{_default_brain_home}/vault"
_default_working_dir = f"{_default_brain_home}/storage"
_default_export_subdir = f"{_default_brain_home}/exports"
_default_refine_subdir = _default_working_dir

# ── Paths ────────────────────────────────────────────────────────
WORKSPACE_ROOT = Path(os.getenv("LIGHTRAG_WORKSPACE_ROOT", _default_workspace))
PROJECT_DIR = Path(os.getenv("KRYONIX_REPO_ROOT", "/etc/kryonix"))
BRAIN_HOME = Path(os.getenv("KRYONIX_BRAIN_HOME", _default_brain_home))
VAULT_DIR = Path(os.getenv("LIGHTRAG_VAULT_DIR", _default_vault))
VAULT_PROPOSAL_DIR = VAULT_DIR / "00-inbox/ai-proposals"
WORKING_DIR = Path(os.getenv("LIGHTRAG_WORKING_DIR", _default_working_dir))
OBSIDIAN_EXPORT_DIR = Path(os.getenv("LIGHTRAG_OBSIDIAN_EXPORT_DIR", _default_export_subdir))
REFINE_STATE_FILE = Path(os.getenv("LIGHTRAG_REFINE_STATE_FILE", str(WORKING_DIR / "refine_state.json")))
REFINE_REPORT_FILE = Path(os.getenv("LIGHTRAG_REFINE_REPORT_FILE", str(WORKING_DIR / "refine_report.json")))

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
    "01-Canonical",
    "01-MOCs",
    "02-Areas",
    "03-Projetos",
    "06-Playbooks",
    "07-Prompts",
]

VAULT_EXCLUDE_DIRS = [
    "storage",
    ".backups",
    ".obsidian",
    ".git",
    ".ssh",
    ".sync",
    "node_modules",
    "ai-rejected",
]

# ── Provider ─────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LIGHTRAG_LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# ── Language & Prompt ─────────────────────────────────────────────
RESPONSE_LANGUAGE = os.getenv("RESPONSE_LANGUAGE", "pt-BR")
ANSWER_SYSTEM_PROMPT = os.getenv("ANSWER_SYSTEM_PROMPT", """
Você é o Kryonix Brain, assistente técnico local do ecossistema Kryonix.

Responda sempre em português do Brasil, de forma técnica, objetiva e útil.

REGRAS DE GROUNDING:
1. Responda prioritariamente com base no CONTEXTO recuperado do Vault, repositório ou índice.
2. Não invente arquivos, comandos, paths, módulos, serviços, funções, endpoints, relações ou etapas que não apareçam no contexto.
3. Se o contexto for insuficiente para responder com segurança, diga claramente:
   "Não encontrei grounding suficiente no Vault/índice atual para responder com segurança."
4. Quando houver fontes, use-as explicitamente para sustentar a resposta.
5. Preserve nomes técnicos, comandos shell, paths, módulos Nix, arquivos, serviços systemd e nomes de pacotes exatamente como aparecem.
6. Se a pergunta envolver um tema específico, responda apenas sobre esse tema. Não puxe exemplos genéricos de fora.
7. Se houver conflito entre contexto recuperado e conhecimento geral, priorize o contexto recuperado.
8. Se a resposta exigir inferência, deixe claro que é uma inferência.
9. Não transforme uma resposta técnica em explicação genérica.
10. Não cite tecnologias, áreas ou funcionalidades que não estejam no contexto recuperado, exceto quando o usuário pedir comparação ou sugestão externa.

FORMATO:
- Comece com a resposta direta.
- Depois detalhe os pontos técnicos.
- Quando possível, inclua comandos ou paths verificáveis.
- Se estiver em modo explain/debug, inclua fontes, chunks, scores e nível de confiança.
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
        "llm_model": "llama3.1:8b",
        "llm_model_max_async": 2,
        "max_parallel_insert": 1,
        "embedding_batch_num": 2,
        "chunk_token_size": 500,
        "chunk_overlap_token_size": 50,
        "index_batch_size": 1,
    },
    "quality": {
        "llm_model": "llama3.1:8b",
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
                               str(WORKING_DIR / "failed_index_files.json")))
SKIPPED_LARGE_FILES_FILE= Path(os.getenv("LIGHTRAG_SKIPPED_LARGE_FILES_FILE",
                               str(WORKING_DIR / "skipped_large_files.json")))
INDEX_MANIFEST_FILE     = Path(os.getenv("LIGHTRAG_INDEX_MANIFEST_FILE",
                               str(WORKING_DIR / ".index_manifest.json")))
INDEX_LOCK_FILE         = Path(os.getenv("LIGHTRAG_INDEX_LOCK_FILE",
                               str(WORKING_DIR / ".index.lock")))
VAULT_CURATE_REPORT     = WORKING_DIR / "vault_curate_report.json"
VAULT_SYNC_LOG          = WORKING_DIR / "vault_sync.log"

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
    ".git", ".ssh", ".gnupg", "node_modules", ".next", "dist", "build", "target", "result", "backups",
    "__pycache__", ".venv", "rag_storage", "obsidian-export", ".direnv", ".cache",
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

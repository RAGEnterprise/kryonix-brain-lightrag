from __future__ import annotations

import os
import re
import logging
import asyncio
from typing import Any

import numpy as np
from ollama import AsyncClient
from lightrag.utils import wrap_embedding_func_with_attrs


from . import config

OLLAMA_BASE_URL = config.OLLAMA_BASE_URL
LIGHTRAG_LLM_MODEL = config.LLM_MODEL
LIGHTRAG_EMBED_MODEL = config.EMBEDDING_MODEL
PROFILE = config.PROFILE
VERBOSE = config.VERBOSE


def _client() -> AsyncClient:
    return AsyncClient(host=OLLAMA_BASE_URL)


logger = logging.getLogger("lightrag.llm")

def _message_content(response: Any) -> str:
    if isinstance(response, dict):
        return str(response.get("message", {}).get("content", ""))

    message = getattr(response, "message", None)
    if isinstance(message, dict):
        return str(message.get("content", ""))

    return str(getattr(message, "content", ""))


def validate_extraction(content: str) -> bool:
    """
    Validates that the extraction output follows the LightRAG schema.
    ENTITIES: ("entity"<name>, <type>, <description>) -> 3 fields
    RELATIONS: ("relationship"<src>, <tgt>, <desc>, <keywords>, <weight>) -> 5 fields
    """
    lines = content.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check RELATION
        if '("relationship"<' in line:
            # Count commas inside parentheses to ensure exactly 5 fields (4 commas)
            # Standard: ("relationship"<source_id>, <target_id>, <description>, <keywords>, <weight>)
            match = re.search(r'\("relationship"<(.*?)>\)', line)
            if match:
                inner = match.group(1)
                # This is a bit naive because descriptions can have commas. 
                # LightRAG's own error "found 4/5 fields" implies it counts something specific.
                # Usually it's the number of comma-separated values.
                # Let's count top-level commas (not inside quotes if possible, but Ollama output varies)
                commas = line.count(",")
                if commas < 4:
                    return False
        
        # Check ENTITY
        if '("entity"<' in line:
            commas = line.count(",")
            if commas < 2:
                return False
                
    # Mandatory completion delimiter for LightRAG
    if "<|COMPLETE|>" not in content:
        return False

    return True


async def llm_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, str]] | None = None,
    **kwargs: Any,
) -> str:
    """
    LightRAG LLM adapter with strict validation and retry logic.
    """
    max_retries = 2
    retry_count = 0
    current_prompt = prompt
    
    is_extraction = "entity" in prompt.lower() and "relationship" in prompt.lower()
    
    while retry_count <= max_retries:
        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for message in history_messages or []:
            role = message.get("role", "user")
            content = message.get("content", "")
            if content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": current_prompt})

        try:
            response = await _client().chat(
                model=LIGHTRAG_LLM_MODEL,
                messages=messages,
                options={
                    "num_ctx": 4096,
                    "temperature": 0.1 if retry_count == 0 else 0.05, # Reduce temp on retry
                    "num_predict": int(kwargs.get("num_predict", 1024)),
                },
                keep_alive="5m",
            )
            
            content = _message_content(response)
            
            if not is_extraction:
                return content
            
            if validate_extraction(content):
                if retry_count > 0 and VERBOSE:
                    print(f"  [LLM] Extração bem-sucedida na tentativa {retry_count}")
                return content
            
            # Validation failed
            retry_count += 1
            if retry_count <= max_retries:
                if VERBOSE:
                    print(f"  [LLM] Erro de formato detectado (tentativa {retry_count}/{max_retries+1}). Repetindo...")
                    print(f"  [DEBUG] Saída bruta: {content[:200]}...")
                
                # Enhance prompt for retry
                strict_instruction = (
                    "\n\nREGRA ESTRITA DE FORMATAÇÃO:\n"
                    "1. Cada ENTITY deve ter EXATAMENTE 3 campos: (\"entity\"<nome>, <tipo>, <descrição>)\n"
                    "2. Cada RELATION deve ter EXATAMENTE 5 campos: (\"relationship\"<origem>, <destino>, <descrição>, <palavras-chave>, <peso>)\n"
                    "3. NÃO omita o peso (ex: use 1.0) ou palavras-chave.\n"
                    "4. Certifique-se de que TODOS os 5 campos estão presentes para cada linha de relacionamento.\n"
                    "5. Produza apenas as tuplas solicitadas, uma por linha."
                )
                current_prompt = prompt + strict_instruction
            else:
                if VERBOSE:
                    print(f"  [AVISO] A extração falhou após {max_retries} tentativas. Revertendo para saída parcial.")
                    print(f"  [DEBUG] Saída final com falha: {content}")
                
                # Fallback: keep only entities if relations are broken? 
                # LightRAG might handle partial lines, but it logs errors.
                # We return the content anyway, but we've logged the failure.
                return content
                
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            retry_count += 1
            if retry_count > max_retries:
                raise
            await asyncio.sleep(1) # Backoff
            
    return ""


@wrap_embedding_func_with_attrs(
    embedding_dim=768,
    max_token_size=2048,
    model_name="nomic-embed-text",
)
async def embedding_func(texts: list[str] | str) -> np.ndarray:
    """
    LightRAG embedding adapter.

    Uses only local Ollama with nomic-embed-text.
    """
    if isinstance(texts, str):
        texts = [texts]

    if PROFILE == "first-run" and len(texts) > 1:
        if VERBOSE:
            print(f"  [EMBED] Perfil first-run ativo, limitando textos a 1 (era {len(texts)})")
        texts = [texts[0]]

    client = _client()


    try:
        response = await client.embed(
            model=LIGHTRAG_EMBED_MODEL,
            input=texts,
            keep_alive="5m",
        )

        if isinstance(response, dict):
            embeddings = response.get("embeddings", [])
        else:
            embeddings = getattr(response, "embeddings", [])

        return np.array(embeddings, dtype=np.float32)

    except Exception:
        vectors: list[list[float]] = []

        for text in texts:
            response = await client.embeddings(
                model=LIGHTRAG_EMBED_MODEL,
                prompt=text,
                keep_alive="5m",
            )

            if isinstance(response, dict):
                vector = response.get("embedding", [])
            else:
                vector = getattr(response, "embedding", [])

            vectors.append(vector)

        return np.array(vectors, dtype=np.float32)


# Compatibility aliases.
embed_func = embedding_func
ollama_llm_func = llm_func
ollama_embedding_func = embedding_func
local_llm_complete = llm_func
local_embedding = embedding_func

from __future__ import annotations

import os
import re
import logging
import asyncio
from typing import Any, AsyncGenerator
from contextvars import ContextVar

# Global Context for tracking the backend used in the current request
USED_BACKEND = ContextVar("used_backend", default="unknown")
PROVIDER_OVERRIDE = ContextVar("provider_override", default=None)
METRICS = ContextVar("llm_metrics", default={})

import numpy as np
import time
from ollama import AsyncClient
from lightrag.utils import wrap_embedding_func_with_attrs


import httpx
from . import config

OLLAMA_BASE_URL = config.OLLAMA_BASE_URL
LLAMA_CPP_BASE_URL = config.LLAMA_CPP_BASE_URL
LLAMA_CPP_TIMEOUT = config.LLAMA_CPP_TIMEOUT
LLM_PROVIDER = config.LLM_PROVIDER
LIGHTRAG_LLM_MODEL = config.LLM_MODEL
LIGHTRAG_EMBED_MODEL = config.EMBEDDING_MODEL
PROFILE = config.PROFILE
VERBOSE = config.VERBOSE


def _client() -> AsyncClient:
    return AsyncClient(host=OLLAMA_BASE_URL)


logger = logging.getLogger("lightrag.llm")

async def _ollama_generate(messages: list[dict[str, str]], **kwargs: Any) -> str:
    """Raw call to Ollama."""
    response = await _client().chat(
        model=LIGHTRAG_LLM_MODEL,
        messages=messages,
        options={
            "num_ctx": 16384,
            "temperature": kwargs.get("temperature", 0.1),
            "num_predict": int(kwargs.get("num_predict", 1024)),
        },
        keep_alive="5m",
    )
    
    # Capture metrics
    m = {}
    if isinstance(response, dict):
        m["prompt_tokens"] = response.get("prompt_eval_count", 0)
        m["completion_tokens"] = response.get("eval_count", 0)
        # Ollama durations are in nanoseconds
        m["total_duration_ms"] = response.get("total_duration", 0) / 1e6
        m["eval_duration_ms"] = response.get("eval_duration", 0) / 1e6
    else:
        m["prompt_tokens"] = getattr(response, "prompt_eval_count", 0)
        m["completion_tokens"] = getattr(response, "eval_count", 0)
        m["total_duration_ms"] = getattr(response, "total_duration", 0) / 1e6
        m["eval_duration_ms"] = getattr(response, "eval_duration", 0) / 1e6

    # Calculate TPS (Tokens Per Second)
    if m.get("eval_duration_ms", 0) > 0:
        m["tps"] = float((m["completion_tokens"] / m["eval_duration_ms"]) * 1000)
    
    # Final cast for all metrics to be safe for JSON
    m = {k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v) for k, v in m.items()}
    
    METRICS.set(m)
    return _message_content(response)


async def _llama_cpp_generate(messages: list[dict[str, str]], **kwargs: Any) -> str:
    """Raw call to llama.cpp (OpenAI-compatible /v1/chat/completions)."""
    url = f"{LLAMA_CPP_BASE_URL}/v1/chat/completions"
    payload = {
        "model": LIGHTRAG_LLM_MODEL,
        "messages": messages,
        "temperature": kwargs.get("temperature", 0.1),
        "max_tokens": int(kwargs.get("num_predict", 1024)),
    }
    
    async with httpx.AsyncClient(timeout=LLAMA_CPP_TIMEOUT) as client:
        start_time = time.monotonic()
        response = await client.post(url, json=payload)
        end_time = time.monotonic()
        
        response.raise_for_status()
        data = response.json()
        
        # Capture metrics
        usage = data.get("usage", {})
        m = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_duration_ms": (end_time - start_time) * 1000,
        }
        # Use native timings if available for more accurate generation TPS
        timings = data.get("timings", {})
        if "predicted_ms" in timings:
            m["eval_duration_ms"] = float(timings["predicted_ms"])
            if m["eval_duration_ms"] > 0:
                m["tps"] = float((m["completion_tokens"] / m["eval_duration_ms"]) * 1000)
        elif m["total_duration_ms"] > 0:
            m["tps"] = float((m["completion_tokens"] / m["total_duration_ms"]) * 1000)
        
        # Final cast for all metrics to be safe for JSON
        m = {k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v) for k, v in m.items()}
        
        METRICS.set(m)
        return data["choices"][0]["message"]["content"]


def _message_content(response: Any) -> str:
    if isinstance(response, dict):
        return str(response.get("message", {}).get("content", ""))

    message = getattr(response, "message", None)
    if isinstance(message, dict):
        return str(message.get("content", ""))

    return str(getattr(message, "content", ""))


def validate_extraction(content: str) -> bool:
    """
    Validates that the extraction output follows the Kryonix LightRAG schema.
    Format: entity<|#|>name<|#|>type<|#|>description
    Format: relation<|#|>src<|#|>tgt<|#|>keywords<|#|>description
    """
    lines = content.splitlines()
    has_entity = False
    has_relation = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check RELATION
        if "relation<|#|>" in line or "relationship<|#|>" in line:
            parts = line.split("<|#|>")
            if len(parts) >= 4: # relation, src, tgt, keywords, [desc]
                has_relation = True
        
        # Check ENTITY
        if "entity<|#|>" in line:
            parts = line.split("<|#|>")
            if len(parts) >= 3: # entity, name, type, [desc]
                has_entity = True
                
    # Mandatory completion delimiter for LightRAG
    if "<|COMPLETE|>" not in content:
        return False

    # We want at least one entity to consider it a success
    return has_entity


async def llm_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, str]] | None = None,
    provider_override: str | None = None,
    **kwargs: Any,
) -> str:
    """
    LightRAG LLM adapter with strict validation, retry logic and provider routing.
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

        # Routing Logic
        provider = (PROVIDER_OVERRIDE.get() or provider_override or LLM_PROVIDER).lower()
        content = ""
        used_backend = "unknown"
        
        try:
            temp = 0.1 if retry_count == 0 else 0.05
            
            if provider == "llama_cpp":
                try:
                    content = await _llama_cpp_generate(messages, temperature=temp, **kwargs)
                    used_backend = "llama_cpp"
                except Exception as e:
                    logger.warning(f"llama_cpp provider failed: {e}")
                    raise # Let it be caught by the outer retry block or re-raised
            
            elif provider == "auto":
                try:
                    content = await _llama_cpp_generate(messages, temperature=temp, **kwargs)
                    used_backend = "llama_cpp"
                except Exception as e:
                    logger.warning(f"llama_cpp (auto) failed, falling back to ollama: {e}")
                    content = await _ollama_generate(messages, temperature=temp, **kwargs)
                    used_backend = "ollama (fallback)"
            
            else: # default: ollama
                content = await _ollama_generate(messages, temperature=temp, **kwargs)
                used_backend = "ollama"

            if VERBOSE:
                print(f"  [LLM] Backend: {used_backend}")

            USED_BACKEND.set(used_backend)

            if not is_extraction:
                return content
            
            if validate_extraction(content):
                if retry_count > 0 and VERBOSE:
                    print(f"  [LLM] Extração bem-sucedida na tentativa {retry_count} ({used_backend})")
                return content
            
            # Validation failed
            retry_count += 1
            if retry_count <= max_retries:
                if VERBOSE:
                    print(f"  [LLM] Erro de formato detectado (tentativa {retry_count}/{max_retries+1}). Repetindo...")
                
                strict_instruction = (
                    "\n\nREGRA ESTRITA DE FORMATAÇÃO:\n"
                    "1. Use o delimitador <|#|> entre campos.\n"
                    "2. Cada ENTITY deve ser: entity<|#|>nome<|#|>tipo<|#|>descrição\n"
                    "3. Cada RELATION deve ser: relation<|#|>origem<|#|>destino<|#|>palavras-chave<|#|>descrição\n"
                    "4. Finalize SEMPRE com a tag <|COMPLETE|> em uma nova linha.\n"
                    "5. Produza apenas as linhas solicitadas, uma por linha."
                )
                current_prompt = prompt + strict_instruction
            else:
                if VERBOSE:
                    print(f"  [AVISO] A extração falhou após {max_retries} tentativas. Revertendo para saída parcial.")
                return content
                
        except Exception as e:
            logger.error(f"LLM Error ({provider}): {e}")
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

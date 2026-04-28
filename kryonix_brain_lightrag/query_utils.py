import re

def normalize_query_for_retrieval(user_query: str) -> str:
    """
    Remove language instructions from the query to improve RAG retrieval.
    Example: 'Responda em português: hyprland' -> 'hyprland'
    """
    prefixes = [
        r"responda em português do brasil",
        r"responda em português",
        r"responda apenas em pt-br",
        r"responda em pt-br",
        r"explique em português do brasil",
        r"explique em português",
        r"explique em pt-br",
        r"use pt-br",
        r"em português do brasil:",
        r"em português do brasil",
        r"em português:",
    ]
    
    normalized = user_query.strip()
    
    # Apply each prefix removal (case insensitive)
    for prefix in prefixes:
        # Match prefix at the start, followed by optional punctuation/whitespace
        pattern = re.compile(f"^{prefix}[:\s,-]*", re.IGNORECASE)
        normalized = pattern.sub("", normalized).strip()
    
    return normalized

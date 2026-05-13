import re

_LANGUAGE_PREFIXES = [
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

_KNOWN_TYPO_REPLACEMENTS = [
    (r"\bseaarch\b", "search"),
    (r"\bseach\b", "search"),
    (r"\bserach\b", "search"),
    (r"\baskk\b", "ask"),
    (r"\bdiferena\b", "diferença"),
    (r"\bdifereça\b", "diferença"),
]


def normalize_query_details(user_query: str) -> dict:
    """Return retrieval-safe query text plus auditable normalization metadata."""
    original = user_query.strip()
    normalized = original
    corrections: list[dict[str, str]] = []

    for prefix in _LANGUAGE_PREFIXES:
        pattern = re.compile(fr"^{prefix}[:\s,-]*", re.IGNORECASE)
        updated = pattern.sub("", normalized).strip()
        if updated != normalized:
            corrections.append({
                "from": normalized,
                "to": updated,
                "reason": "language_prefix",
            })
        normalized = updated

    for pattern_text, replacement in _KNOWN_TYPO_REPLACEMENTS:
        pattern = re.compile(pattern_text, re.IGNORECASE)

        def replace_match(match: re.Match) -> str:
            value = match.group(0)
            corrections.append({
                "from": value,
                "to": replacement,
                "reason": "known_typo",
            })
            return replacement

        normalized = pattern.sub(replace_match, normalized)

    normalized = re.sub(r"\s+", " ", normalized).strip()
    return {
        "query_original": original,
        "query_normalized": normalized,
        "corrections_applied": corrections,
    }


def normalize_query_for_retrieval(user_query: str) -> str:
    """
    Remove language instructions from the query to improve RAG retrieval.
    Example: 'Responda em português: hyprland' -> 'hyprland'
    """
    return normalize_query_details(user_query)["query_normalized"]

import pytest
from kryonix_brain_lightrag.query_utils import (
    normalize_query_details,
    normalize_query_for_retrieval,
)

def test_normalize_prefixes():
    assert normalize_query_for_retrieval("Responda em português: hyprland") == "hyprland"
    assert normalize_query_for_retrieval("Explique em português: kryonix") == "kryonix"
    assert normalize_query_for_retrieval("use pt-br hyprland") == "hyprland"
    assert normalize_query_for_retrieval("em português do brasil: ragos cli") == "ragos cli"
    assert normalize_query_for_retrieval("Responda apenas em pt-br: nixos") == "NixOS"

def test_normalize_case_insensitive():
    assert normalize_query_for_retrieval("RESPONDA EM PORTUGUÊS: hyprland") == "hyprland"
    assert normalize_query_for_retrieval("Em Português do Brasil: Test") == "Test"

def test_normalize_no_prefix():
    assert normalize_query_for_retrieval("hyprland configuration") == "hyprland configuration"

def test_normalize_multiple_whitespace():
    assert normalize_query_for_retrieval("Responda em português:    hyprland") == "hyprland"


def test_normalize_known_ptbr_typos():
    details = normalize_query_details("qual diferena tem entre o ask e seaarch")
    assert details["query_original"] == "qual diferena tem entre o ask e seaarch"
    assert details["query_normalized"] == "qual diferença tem entre o ask e search"
    assert {"from": "diferena", "to": "diferença", "reason": "known_typo"} in details["corrections_applied"]
    assert {"from": "seaarch", "to": "search", "reason": "known_typo"} in details["corrections_applied"]


def test_normalize_search_and_ask_variants():
    assert normalize_query_for_retrieval("seach serach askk difereça") == "search search ask diferença"


# ── Path-safety regressions (#34) ────────────────────────────────────────────

def test_normalize_preserves_path_with_cag():
    """cag inside a file path must NOT be uppercased to CAG."""
    path = "/var/lib/kryonix/brain/cag/manifest.json"
    assert normalize_query_for_retrieval(path) == path


def test_normalize_preserves_path_with_rag():
    """rag inside a file path must NOT be uppercased to RAG."""
    path = "/var/lib/kryonix/brain/rag/storage"
    assert normalize_query_for_retrieval(path) == path


def test_normalize_does_not_corrupt_cli_flags():
    """CLI flags like --json must survive normalization unchanged."""
    query = "kryonix brain search --json"
    assert normalize_query_for_retrieval(query) == query


def test_normalize_standalone_cag_still_uppercased():
    """cag as a standalone word in a question must still be normalized to CAG."""
    result = normalize_query_for_retrieval("como usar o cag do kryonix")
    assert "CAG" in result

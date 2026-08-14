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


def test_pkgs_nix_attr_preserved():
    """pkgs.X is canonical Nix syntax — must never be corrupted by normalization."""
    assert normalize_query_for_retrieval("como instalar pkgs.hello no NixOS") == "como instalar pkgs.hello no NixOS"
    assert normalize_query_for_retrieval("pkgs.git pkgs.python3") == "pkgs.git pkgs.python3"
    assert normalize_query_for_retrieval("use pkgs em flake.nix") == "use pkgs em flake.nix"
    assert normalize_query_for_retrieval("pkg.hello deve ser pkgs.hello") == "pkg.hello deve ser pkgs.hello"

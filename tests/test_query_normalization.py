import pytest
from kryonix_brain_lightrag.query_utils import normalize_query_for_retrieval

def test_normalize_prefixes():
    assert normalize_query_for_retrieval("Responda em português: hyprland") == "hyprland"
    assert normalize_query_for_retrieval("Explique em português: kryonix") == "kryonix"
    assert normalize_query_for_retrieval("use pt-br hyprland") == "hyprland"
    assert normalize_query_for_retrieval("em português do brasil: ragos cli") == "ragos cli"
    assert normalize_query_for_retrieval("Responda apenas em pt-br: nixos") == "nixos"

def test_normalize_case_insensitive():
    assert normalize_query_for_retrieval("RESPONDA EM PORTUGUÊS: hyprland") == "hyprland"
    assert normalize_query_for_retrieval("Em Português do Brasil: Test") == "Test"

def test_normalize_no_prefix():
    assert normalize_query_for_retrieval("hyprland configuration") == "hyprland configuration"

def test_normalize_multiple_whitespace():
    assert normalize_query_for_retrieval("Responda em português:    hyprland") == "hyprland"

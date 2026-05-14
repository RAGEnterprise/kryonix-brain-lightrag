import pytest
from kryonix_brain_lightrag.rag import (
    assess_intent_coverage,
    build_grounding_metadata
)

def test_assess_intent_coverage_comparison():
    query = "qual a diferença entre ask e search"
    # Case 1: Both terms covered
    chunks = [{"content": "o comando ask sintetiza...", "file_path": "a.md"}, 
              {"content": "o comando search recupera...", "file_path": "b.md"}]
    res = assess_intent_coverage(query, chunks)
    assert res["intent_coverage"] == "full"
    assert res["answerability_score"] == 1.0
    assert "ask" in res["covered_terms"]
    assert "search" in res["covered_terms"]

    # Case 2: Partial coverage
    chunks = [{"content": "o comando ask sintetiza...", "file_path": "a.md"}]
    res = assess_intent_coverage(query, chunks)
    assert res["intent_coverage"] == "partial"
    assert res["answerability_score"] == 0.5
    assert "search" in res["missing_terms"]

def test_assess_intent_coverage_standard():
    query = "como configurar o nixos"
    chunks = [{"content": "configuração do nixos...", "file_path": "nixos.md"}]
    res = assess_intent_coverage(query, chunks)
    assert res["intent_coverage"] == "full"
    assert res["answerability_score"] == 0.85

def test_build_grounding_metadata_logic():
    # Case 1: Contradiction case (High retrieval, Low answerability)
    # This happens when chunks are similar but don't contain the answer
    meta = build_grounding_metadata(
        retrieval_score=0.95,
        answerability_score=0.3,
        intent_coverage="none"
    )
    assert meta["grounding_label"] == "Baixa"
    assert meta["answerability_reason"] != ""
    assert "Similaridade alta" in meta["answerability_reason"]

    # Case 2: Perfect grounding
    meta = build_grounding_metadata(
        retrieval_score=0.9,
        answerability_score=0.9,
        intent_coverage="full"
    )
    assert meta["grounding_label"] == "Alta"

    # Case 3: Medium grounding due to retrieval score
    meta = build_grounding_metadata(
        retrieval_score=0.5,
        answerability_score=0.9,
        intent_coverage="full"
    )
    assert meta["grounding_label"] == "Média"

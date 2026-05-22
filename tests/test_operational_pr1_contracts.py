from kryonix_brain_lightrag.api import _missing_manifest_payload
from kryonix_brain_lightrag.rag import (
    assess_intent_coverage,
    build_grounding_metadata,
)


def test_missing_manifest_payload_is_actionable():
    payload = _missing_manifest_payload(
        FileNotFoundError("No manifest found at /var/lib/kryonix/brain/cag/manifest.json")
    )

    assert payload["status"] == "missing_manifest"
    assert payload["error_code"] == "CAG_MANIFEST_MISSING"
    assert payload["ok"] is False
    assert payload["manifest_path"] == "/var/lib/kryonix/brain/cag/manifest.json"
    assert "kryonix brain cag status" in payload["recommended_commands"]
    assert "kryonix brain cag build" in payload["recommended_commands"]


def test_not_answerable_grounding_never_reports_high():
    grounding = build_grounding_metadata(
        retrieval_score=0.95,
        intent_coverage="none",
        answerability="not_answerable",
    )

    assert grounding["grounding_label"] == "Baixa"
    assert grounding["answerability"] == "not_answerable"
    assert grounding["retrieval_score"] == 0.95


def test_comparative_ask_search_coverage_requires_both_terms():
    chunks = [
        {
            "file_path": "docs/CLI.md",
            "content": "kryonix brain ask pergunta; kryonix brain search pergunta",
        }
    ]

    coverage = assess_intent_coverage("qual diferença tem entre ask e search", chunks)

    assert coverage["intent_coverage"] == "full"
    assert coverage["answerability"] == "answerable"
    assert "ask" in coverage["covered_terms"]
    assert "search" in coverage["covered_terms"]

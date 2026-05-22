"""Testa que o handler /cag/ask trata manifest ausente sem propagar HTTP 500."""
import pytest
from unittest.mock import patch
from kryonix_brain_lightrag.api import _missing_manifest_payload, cag_ask, CagQueryRequest


@pytest.mark.asyncio
async def test_cag_ask_missing_manifest_returns_structured_payload():
    """cag_ask() com FileNotFoundError deve retornar payload estruturado, nunca re-raise."""
    req = CagQueryRequest(query="qual diferença entre ask e search")

    with patch(
        "kryonix_brain_lightrag.cag.ask",
        side_effect=FileNotFoundError(
            "No manifest found at /var/lib/kryonix/brain/cag/manifest.json"
        ),
    ):
        result = await cag_ask(req, api_key="test-key")

    assert isinstance(result, dict), "handler deve retornar dict, não levantar exceção"
    assert result["status"] == "missing_manifest"
    assert result["error_code"] == "CAG_MANIFEST_MISSING"
    assert result["ok"] is False
    assert result["manifest_path"] == "/var/lib/kryonix/brain/cag/manifest.json"
    assert "kryonix brain cag build" in result["recommended_commands"]


@pytest.mark.asyncio
async def test_cag_route_missing_manifest_returns_structured_payload():
    """cag_route() com FileNotFoundError deve retornar payload estruturado, nunca re-raise."""
    from kryonix_brain_lightrag.api import cag_route

    req = CagQueryRequest(query="qual comando usar?")

    with patch(
        "kryonix_brain_lightrag.cag.route",
        side_effect=FileNotFoundError(
            "No manifest found at /var/lib/kryonix/brain/cag/manifest.json"
        ),
    ):
        result = await cag_route(req, api_key="test-key")

    assert result["status"] == "missing_manifest"
    assert result["error_code"] == "CAG_MANIFEST_MISSING"
    assert result["ok"] is False

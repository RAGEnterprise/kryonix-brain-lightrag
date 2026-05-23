"""Regression tests for ragton/kryonix#35.

/cag/ask and /cag/route must never propagate FileNotFoundError as HTTP 500
when manifest.json is absent. Instead they must return a structured dict
with status='missing_manifest' and actionable recommended_commands.

These tests call the async handler functions directly — no TestClient or
httpx needed — so they run in CI without extra dev-deps.
"""
import json
import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# /cag/ask
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cag_ask_no_exception_on_missing_manifest():
    """Handler must return a dict, not raise, when manifest.json is absent."""
    from kryonix_brain_lightrag.api import cag_ask, CagQueryRequest

    req = CagQueryRequest(query="qual diferença entre ask e search")

    with patch("kryonix_brain_lightrag.cag.ask") as mock_ask:
        mock_ask.side_effect = FileNotFoundError(
            "No manifest found at /var/lib/kryonix/brain/cag/manifest.json"
        )
        # Calling the handler directly bypasses FastAPI Depends validation.
        result = await cag_ask(req, api_key="test-key")

    assert isinstance(result, dict), "Must return dict, not raise"


@pytest.mark.asyncio
async def test_cag_ask_missing_manifest_status_field():
    """Response status must be 'missing_manifest', not an HTTP 500 detail."""
    from kryonix_brain_lightrag.api import cag_ask, CagQueryRequest

    req = CagQueryRequest(query="test")

    with patch("kryonix_brain_lightrag.cag.ask") as mock_ask:
        mock_ask.side_effect = FileNotFoundError(
            "No manifest found at /var/lib/kryonix/brain/cag/manifest.json"
        )
        result = await cag_ask(req, api_key="test-key")

    assert result.get("status") == "missing_manifest"


@pytest.mark.asyncio
async def test_cag_ask_missing_manifest_includes_path():
    """Error payload must include the exact manifest_path."""
    from kryonix_brain_lightrag.api import cag_ask, CagQueryRequest

    req = CagQueryRequest(query="test")
    manifest_path = "/var/lib/kryonix/brain/cag/manifest.json"

    with patch("kryonix_brain_lightrag.cag.ask") as mock_ask:
        mock_ask.side_effect = FileNotFoundError(
            f"No manifest found at {manifest_path}"
        )
        result = await cag_ask(req, api_key="test-key")

    assert manifest_path in result.get("manifest_path", "")


@pytest.mark.asyncio
async def test_cag_ask_missing_manifest_includes_build_command():
    """recommended_commands must contain 'kryonix brain cag build'."""
    from kryonix_brain_lightrag.api import cag_ask, CagQueryRequest

    req = CagQueryRequest(query="test")

    with patch("kryonix_brain_lightrag.cag.ask") as mock_ask:
        mock_ask.side_effect = FileNotFoundError(
            "No manifest found at /var/lib/kryonix/brain/cag/manifest.json"
        )
        result = await cag_ask(req, api_key="test-key")

    commands = result.get("recommended_commands", [])
    assert any("cag build" in cmd for cmd in commands), (
        f"'kryonix brain cag build' not found in recommended_commands: {commands}"
    )


@pytest.mark.asyncio
async def test_cag_ask_no_secret_leak_on_missing_manifest():
    """Error payload must not expose secrets or sensitive env vars (issue #31)."""
    from kryonix_brain_lightrag.api import cag_ask, CagQueryRequest

    req = CagQueryRequest(query="test")

    with patch("kryonix_brain_lightrag.cag.ask") as mock_ask:
        mock_ask.side_effect = FileNotFoundError(
            "No manifest found at /var/lib/kryonix/brain/cag/manifest.json"
        )
        result = await cag_ask(req, api_key="test-key")

    serialized = json.dumps(result).lower()
    sensitive_terms = [
        "kryonix_brain_api_key",
        "kryonix_brain_key",
        "neo4j_auth",
        "password",
        "secret",
    ]
    for term in sensitive_terms:
        assert term not in serialized, (
            f"Sensitive term '{term}' found in error payload"
        )


# ---------------------------------------------------------------------------
# /cag/route  (same invariants must hold)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cag_route_no_exception_on_missing_manifest():
    """cag_route handler must also return dict instead of raising."""
    from kryonix_brain_lightrag.api import cag_route, CagQueryRequest

    req = CagQueryRequest(query="test route")

    with patch("kryonix_brain_lightrag.cag.route") as mock_route:
        mock_route.side_effect = FileNotFoundError(
            "No manifest found at /var/lib/kryonix/brain/cag/manifest.json"
        )
        result = await cag_route(req, api_key="test-key")

    assert isinstance(result, dict)
    assert result.get("status") == "missing_manifest"
    commands = result.get("recommended_commands", [])
    assert any("cag build" in cmd for cmd in commands)

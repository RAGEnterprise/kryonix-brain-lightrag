"""Regression tests for CAG missing-manifest handling (issue #35).

Unit tests (no HTTP) verify _missing_manifest_payload structure.
HTTP-level tests verify /cag/ask and /cag/route return 424, not 500,
when the manifest file is absent.

HTTP tests are skipped when httpx is not installed.
"""
import pytest
from unittest.mock import patch
from kryonix_brain_lightrag.api import _missing_manifest_payload


# ── Unit tests ─────────────────────────────────────────────────────────────────

def test_missing_manifest_payload_with_known_path():
    payload = _missing_manifest_payload(
        FileNotFoundError("No manifest found at /var/lib/kryonix/brain/cag/manifest.json")
    )
    assert payload["status"] == "missing_manifest"
    assert payload["manifest_path"] == "/var/lib/kryonix/brain/cag/manifest.json"
    assert "kryonix brain cag build" in payload["recommended_commands"]
    assert "kryonix brain cag status" in payload["recommended_commands"]
    assert "message" in payload


def test_missing_manifest_payload_without_known_path():
    payload = _missing_manifest_payload(FileNotFoundError("Manifest not found"))
    assert payload["status"] == "missing_manifest"
    assert payload["manifest_path"] == ""
    assert "message" in payload


# ── HTTP-level tests (require httpx) ───────────────────────────────────────────

@pytest.fixture(scope="module")
def api_client():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from kryonix_brain_lightrag.api import app, get_api_key

    async def _bypass():
        return "test-key"

    app.dependency_overrides[get_api_key] = _bypass
    client = TestClient(app)
    yield client
    app.dependency_overrides.pop(get_api_key, None)


def test_cag_ask_returns_424_on_missing_manifest(api_client):
    """POST /cag/ask must return 424 (not 500) when manifest is absent."""
    with patch("kryonix_brain_lightrag.cag.ask") as mock_ask:
        mock_ask.side_effect = FileNotFoundError(
            "No manifest found at /var/lib/kryonix/brain/cag/manifest.json"
        )
        response = api_client.post(
            "/cag/ask",
            json={"query": "qual diferença entre ask e search"},
            headers={"X-API-Key": "test-key"},
        )
    assert response.status_code == 424
    body = response.json()
    assert body["detail"]["status"] == "missing_manifest"
    assert body["detail"]["manifest_path"] == "/var/lib/kryonix/brain/cag/manifest.json"
    assert "kryonix brain cag build" in body["detail"]["recommended_commands"]
    assert "kryonix brain cag status" in body["detail"]["recommended_commands"]


def test_cag_route_returns_424_on_missing_manifest(api_client):
    """POST /cag/route must return 424 (not 500) when manifest is absent."""
    with patch("kryonix_brain_lightrag.cag.route") as mock_route:
        mock_route.side_effect = FileNotFoundError(
            "No manifest found at /var/lib/kryonix/brain/cag/manifest.json"
        )
        response = api_client.post(
            "/cag/route",
            json={"query": "test"},
            headers={"X-API-Key": "test-key"},
        )
    assert response.status_code == 424
    body = response.json()
    assert body["detail"]["status"] == "missing_manifest"


def test_cag_status_returns_200_when_manifest_missing(api_client):
    """GET /cag/status always returns 200 so CLI won't treat it as a network error."""
    with patch("kryonix_brain_lightrag.cag.status") as mock_status:
        mock_status.return_value = {
            "status": "missing",
            "message": "No manifest found at /var/lib/kryonix/brain/cag/manifest.json",
        }
        response = api_client.get(
            "/cag/status",
            headers={"X-API-Key": "test-key"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "missing_manifest"
    assert "kryonix brain cag build" in body["recommended_commands"]

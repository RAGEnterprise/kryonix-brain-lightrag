"""
Regression tests: /cag/ask e /cag/route nao devem retornar HTTP 500 quando
o manifest CAG esta ausente. Cobre kryonix#35 e faz parte de kryonix#31.
"""
import os
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient


_MANIFEST_PATH = "/var/lib/kryonix/brain/cag/manifest.json"
_MISSING_ERROR = FileNotFoundError(f"No manifest found at {_MANIFEST_PATH}")


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch):
    monkeypatch.setenv("KRYONIX_BRAIN_API_KEY", "test-key")


@pytest.fixture
def client():
    from kryonix_brain_lightrag.api import app
    return TestClient(app)


_HEADERS = {"X-API-Key": "test-key"}


def test_cag_ask_missing_manifest_not_500(client):
    """/cag/ask nao retorna HTTP 500 quando manifest esta ausente."""
    with patch("kryonix_brain_lightrag.cag.ask", side_effect=_MISSING_ERROR):
        resp = client.post("/cag/ask", json={"query": "teste"}, headers=_HEADERS)
    assert resp.status_code != 500, (
        f"Expected non-500 status, got {resp.status_code}: {resp.text}"
    )


def test_cag_ask_missing_manifest_structured_payload(client):
    """/cag/ask retorna payload estruturado com status, manifest_path e recommended_commands."""
    with patch("kryonix_brain_lightrag.cag.ask", side_effect=_MISSING_ERROR):
        resp = client.post("/cag/ask", json={"query": "teste"}, headers=_HEADERS)
    body = resp.json()
    assert body.get("status") == "missing_manifest"
    assert body.get("manifest_path") == _MANIFEST_PATH
    assert isinstance(body.get("recommended_commands"), list)
    assert any("cag build" in cmd for cmd in body["recommended_commands"])


def test_cag_route_missing_manifest_not_500(client):
    """/cag/route tambem retorna payload estruturado quando manifest esta ausente."""
    with patch("kryonix_brain_lightrag.cag.route", side_effect=_MISSING_ERROR):
        resp = client.post("/cag/route", json={"query": "teste"}, headers=_HEADERS)
    assert resp.status_code != 500
    body = resp.json()
    assert body.get("status") == "missing_manifest"


def test_cag_ask_missing_manifest_no_secrets_leaked(client):
    """Payload de erro nao deve conter termos sensiveis."""
    with patch("kryonix_brain_lightrag.cag.ask", side_effect=_MISSING_ERROR):
        resp = client.post("/cag/ask", json={"query": "teste"}, headers=_HEADERS)
    body_text = resp.text.lower()
    for term in ("api_key", "neo4j_auth", "password", "token=", "secret"):
        assert term not in body_text, (
            f"Sensitive term '{term}' found in error response payload"
        )

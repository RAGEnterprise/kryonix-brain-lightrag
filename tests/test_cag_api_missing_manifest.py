"""Regression tests for issue #35: /cag/ask and /cag/route must return HTTP 200
with a structured payload when the CAG manifest is absent, never HTTP 500.
"""
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

_TEST_KEY = "ci-test-key"


@pytest.fixture
def client():
    with patch.dict(os.environ, {"KRYONIX_BRAIN_API_KEY": _TEST_KEY}):
        from kryonix_brain_lightrag.api import app
        yield TestClient(app)


def test_cag_ask_missing_manifest_returns_structured_200(client):
    """Regression #35: /cag/ask sem manifest → 200 estruturado, não HTTP 500."""
    exc = FileNotFoundError(
        "No manifest found at /var/lib/kryonix/brain/cag/manifest.json"
    )
    with patch("kryonix_brain_lightrag.cag.ask", side_effect=exc):
        response = client.post(
            "/cag/ask",
            json={"query": "qual diferença entre ask e search"},
            headers={"X-API-Key": _TEST_KEY},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "missing_manifest"
    assert data["manifest_path"] == "/var/lib/kryonix/brain/cag/manifest.json"
    assert "kryonix brain cag build" in data["recommended_commands"]
    assert "kryonix brain cag status" in data["recommended_commands"]


def test_cag_route_missing_manifest_returns_structured_200(client):
    """Regression #35: /cag/route sem manifest → 200 estruturado, não HTTP 500."""
    exc = FileNotFoundError(
        "No manifest found at /var/lib/kryonix/brain/cag/manifest.json"
    )
    with patch("kryonix_brain_lightrag.cag.route", side_effect=exc):
        response = client.post(
            "/cag/route",
            json={"query": "rotear esta pergunta"},
            headers={"X-API-Key": _TEST_KEY},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "missing_manifest"
    assert "recommended_commands" in data


def test_cag_ask_real_error_still_returns_500(client):
    """Outros erros internos devem continuar retornando HTTP 500."""
    with patch(
        "kryonix_brain_lightrag.cag.ask",
        side_effect=RuntimeError("unexpected internal failure"),
    ):
        response = client.post(
            "/cag/ask",
            json={"query": "teste"},
            headers={"X-API-Key": _TEST_KEY},
        )

    assert response.status_code == 500

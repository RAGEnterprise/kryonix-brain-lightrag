import pytest
import os
from pathlib import Path
from kryonix_brain_lightrag.obsidian_cli import _safe_path, VAULT_ROOT

def test_safe_path_within_vault():
    path = "test.md"
    safe = _safe_path(path)
    assert str(safe).startswith(str(VAULT_ROOT.resolve()))

def test_safe_path_traversal_blocked():
    with pytest.raises(PermissionError):
        _safe_path("../../etc/passwd")

def test_safe_path_absolute_within_vault():
    abs_path = VAULT_ROOT / "sub" / "note.md"
    # Ensure parent exists for resolve() if needed, but _safe_path handles it
    safe = _safe_path(str(abs_path))
    assert str(safe) == str(abs_path.resolve())

def test_safe_path_absolute_outside_vault():
    with pytest.raises(PermissionError):
        # Usando um path absoluto real do linux fora do vault
        _safe_path("/etc/shadow")

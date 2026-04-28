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
        _safe_path("../../windows/system32/cmd.exe")

def test_safe_path_absolute_within_vault():
    abs_path = VAULT_ROOT / "sub" / "note.md"
    safe = _safe_path(str(abs_path))
    assert safe == abs_path.resolve()

def test_safe_path_absolute_outside_vault():
    with pytest.raises(PermissionError):
        _safe_path("C:/Users/aguia/Documents/secret.txt")

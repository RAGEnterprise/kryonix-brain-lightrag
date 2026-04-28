import subprocess
import sys
import pytest

def test_cli_help():
    # Just check if the flags are present in help
    cmd = [sys.executable, "-m", "kryonix_brain_lightrag.cli", "index", "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert "--only-useful" in result.stdout
    assert "--min-chars" in result.stdout
    assert "--reset-refine-state" in result.stdout
    assert "--report" in result.stdout

def test_cli_report_non_existent():
    # Should not crash even if no report exists
    cmd = [sys.executable, "-m", "kryonix_brain_lightrag.cli", "index", "--report"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert "Relat\u00f3rio n\u00e3o encontrado" in result.stdout or "LIGHTRAG REFINE REPORT" in result.stdout

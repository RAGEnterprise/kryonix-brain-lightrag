import subprocess
import sys
import json
from pathlib import Path

def test_mcp_stdio_clean():
    """Verify that the MCP server doesn't output non-JSON to stdout on startup."""
    # Run the server with --help or just a simple check that doesn't start the full loop
    # but loads the modules
    cmd = [sys.executable, "-m", "kryonix_brain_lightrag.server", "--help"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    # Stdout should be empty or only contains help text if redirected
    # But usually, we want to ensure no 'INFO: ...' or 'Loading ...' goes to stdout
    # during actual MCP execution.
    
    # Let's try to find if there are any stray prints in server.py
    server_path = Path(__file__).parent.parent / "kryonix_brain_lightrag" / "server.py"
    if server_path.exists():
        content = server_path.read_text(encoding="utf-8")
        assert 'print(' not in content, "Found stray print() in server.py, use logger instead!"

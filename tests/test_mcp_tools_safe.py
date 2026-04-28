import pytest
from kryonix_brain_lightrag.server import list_tools

@pytest.mark.asyncio
async def test_no_dangerous_tools():
    tools = await list_tools()
    tool_names = [t.name for t in tools]
    
    dangerous = ["reset", "first-run", "delete", "full_index", "shell"]
    for d in dangerous:
        assert d not in tool_names, f"Dangerous tool {d} should not be exposed via MCP"

@pytest.mark.asyncio
async def test_required_tools_present():
    tools = await list_tools()
    tool_names = [t.name for t in tools]
    
    required = [
        "rag_search", "rag_ask", "obsidian_read", "obsidian_write", 
        "brain_search", "brain_answer", "brain_capture"
    ]
    for r in required:
        assert r in tool_names, f"Required tool {r} is missing from MCP"

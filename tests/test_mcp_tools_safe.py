import pytest
from kryonix_brain_lightrag.server import list_tools

@pytest.mark.asyncio
async def test_no_dangerous_tools():
    tools = await list_tools()
    tool_names = [t.name for t in tools]
    
    # Adicionando ferramentas que agora são proibidas
    dangerous = [
        "reset", "first-run", "delete", "full_index", "shell",
        "obsidian_write", "obsidian_append", "obsidian_create_moc",
        "brain_sync", "brain_capture", "vault_index"
    ]
    for d in dangerous:
        assert d not in tool_names, f"Dangerous or restricted tool {d} should not be exposed via MCP"

@pytest.mark.asyncio
async def test_required_tools_present():
    tools = await list_tools()
    tool_names = [t.name for t in tools]
    
    # Novas ferramentas seguras
    required = [
        "rag_search", "obsidian_read", "obsidian_search",
        "brain_learn_propose", "brain_note_propose", "brain_events_log"
    ]
    for r in required:
        assert r in tool_names, f"Required tool {r} is missing from MCP"

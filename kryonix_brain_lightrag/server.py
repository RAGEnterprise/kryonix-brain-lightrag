import asyncio
import json
import sys
import os
from pathlib import Path

# --- MCP Stdio Silence ---
# Redirect all stdout to stderr immediately to avoid corrupting the MCP JSON-RPC stream.
# The MCP stdio transport uses the original stdout for communication.
original_stdout = sys.stdout
sys.stdout = sys.stderr

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from . import rag as rag_mod
from . import config
from . import obsidian_cli
from .index import cmd_repair_vdb, cmd_index

app = Server("kryonix-brain")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        # ── RAG & Knowledge Tools ───────────────────────────────────
        Tool(
            name="rag_search",
            description="Search the LightRAG knowledge graph. Returns a synthesized answer with citations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "mode": {"type": "string", "enum": ["hybrid", "naive", "local", "global"], "default": "hybrid"},
                    "lang": {"type": "string", "default": "pt-BR"}
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="rag_stats",
            description="Get knowledge graph statistics (entities, relations, documents).",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="rag_health",
            description="Check RAG health, consistency and security constraints.",
            inputSchema={"type": "object", "properties": {}},
        ),
        
        # ── Safe Learning & Proposal Tools ──────────────────────────
        Tool(
            name="brain_learn_propose",
            description="Propose new content to be learned/indexed by the Brain. Requires human approval.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The information to learn"},
                    "source": {"type": "string", "description": "Source URL or file path"},
                    "reason": {"type": "string", "description": "Why this is useful"}
                },
                "required": ["content", "source", "reason"],
            },
        ),
        Tool(
            name="brain_note_propose",
            description="Propose a new note for the Obsidian vault. Note will be placed in ai-proposals inbox.",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Note title (slugified)"},
                    "content": {"type": "string", "description": "Markdown content"},
                    "source": {"type": "string", "description": "Information source"},
                    "reason": {"type": "string", "description": "Why create this note?"}
                },
                "required": ["title", "content", "source", "reason"],
            },
        ),
        Tool(
            name="brain_events_log",
            description="Record a technical event or interaction log for future reference.",
            inputSchema={
                "type": "object",
                "properties": {
                    "event": {"type": "string", "description": "Event description"},
                    "metadata": {"type": "object", "description": "Additional context"}
                },
                "required": ["event"],
            },
        ),

        # ── Maintenance & Integrity Tools (Safe) ────────────────────
        Tool(
            name="graph_repair_dry_run",
            description="Run a diagnostic repair on the knowledge graph without modifying files.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="rag_repair_vdb_dry_run",
            description="Check if VDB needs reconstruction without actually running it.",
            inputSchema={"type": "object", "properties": {}},
        ),

        # ── Read-Only Obsidian Tools ────────────────────────────────
        Tool(
            name="obsidian_search",
            description="Search for notes in the Obsidian vault.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search term"}
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="obsidian_read",
            description="Read a note from the Obsidian vault.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to vault root"}
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="obsidian_status",
            description="Get vault metadata and count of notes.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        # --- RAG & Search ---
        if name == "rag_search":
            query = arguments.get("query", "")
            mode = arguments.get("mode", "hybrid")
            lang = arguments.get("lang", "pt-BR")
            result = await rag_mod.query(query, mode=mode, lang=lang)
            return [TextContent(type="text", text=result)]

        elif name == "rag_stats":
            info = await rag_mod.stats()
            return [TextContent(type="text", text=json.dumps(info, indent=2))]

        elif name == "rag_health":
            from .cli import cmd_doctor
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            with redirect_stdout(f):
                await cmd_doctor(None)
            return [TextContent(type="text", text=f.getvalue())]

        # --- Proposals & Events ---
        elif name == "brain_learn_propose":
            import httpx
            api_key = os.getenv("KRYONIX_BRAIN_KEY")
            port = os.getenv("KRYONIX_BRAIN_PORT", "8000")
            url = f"http://127.0.0.1:{port}/ingest/propose"
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url, 
                    json={
                        "content": arguments["content"],
                        "source": arguments["source"],
                        "reason": arguments["reason"]
                    },
                    headers={"X-API-Key": api_key}
                )
                return [TextContent(type="text", text=json.dumps(resp.json(), indent=2))]

        elif name == "brain_note_propose":
            import httpx
            api_key = os.getenv("KRYONIX_BRAIN_KEY")
            port = os.getenv("KRYONIX_BRAIN_PORT", "8000")
            url = f"http://127.0.0.1:{port}/notes/propose"
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url, 
                    json={
                        "title": arguments["title"],
                        "content": arguments["content"],
                        "source": arguments["source"],
                        "reason": arguments["reason"]
                    },
                    headers={"X-API-Key": api_key}
                )
                return [TextContent(type="text", text=json.dumps(resp.json(), indent=2))]

        elif name == "brain_events_log":
            import httpx
            api_key = os.getenv("KRYONIX_BRAIN_KEY")
            port = os.getenv("KRYONIX_BRAIN_PORT", "8000")
            url = f"http://127.0.0.1:{port}/events/log"
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url, 
                    json={
                        "event": arguments["event"],
                        "metadata": arguments.get("metadata", {})
                    },
                    headers={"X-API-Key": api_key}
                )
                return [TextContent(type="text", text=json.dumps(resp.json(), indent=2))]

        # --- Maintenance (Safe) ---
        elif name == "graph_repair_dry_run":
            import io
            from contextlib import redirect_stdout
            from .index import cmd_repair_graph
            f = io.StringIO()
            with redirect_stdout(f):
                await cmd_repair_graph(dry_run=True)
            return [TextContent(type="text", text=f.getvalue())]

        elif name == "rag_repair_vdb_dry_run":
            import io
            from contextlib import redirect_stdout
            from .index import cmd_repair_vdb
            f = io.StringIO()
            with redirect_stdout(f):
                await cmd_repair_vdb(dry_run=True)
            return [TextContent(type="text", text=f.getvalue())]

        # --- Read-Only Obsidian ---
        elif name == "obsidian_status":
            res = obsidian_cli.obsidian_status()
            return [TextContent(type="text", text=json.dumps(res, indent=2))]

        elif name == "obsidian_search":
            res = obsidian_cli.obsidian_search_notes(arguments["query"])
            return [TextContent(type="text", text=json.dumps(res, indent=2))]

        elif name == "obsidian_read":
            res = obsidian_cli.obsidian_read_note(arguments["path"])
            return [TextContent(type="text", text=res)]

        else:
            return [TextContent(type="text", text=f"Tool '{name}' is restricted or unknown.")]

    except Exception as e:
        return [TextContent(type="text", text=f"Error executing {name}: {str(e)}")]

def main():
    import anyio
    from io import TextIOWrapper

    async def run():
        # The MCP stdio transport expects anyio.AsyncFile objects.
        # We re-wrap the underlying binary streams to ensure UTF-8 and proper async handling.
        async_stdin = anyio.wrap_file(TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace"))
        async_stdout = anyio.wrap_file(TextIOWrapper(original_stdout.buffer, encoding="utf-8"))

        async with stdio_server(async_stdin, async_stdout) as (read_stream, write_stream):
            await app.run(
                read_stream, write_stream,
                app.create_initialization_options(),
            )
    
    try:
        asyncio.run(run())
    finally:
        sys.stdout = original_stdout

if __name__ == "__main__":
    main()

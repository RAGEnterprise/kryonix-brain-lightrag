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
from mcp.types import Tool, TextContent, Resource

from . import rag as rag_mod
from . import config
from . import obsidian_cli
from .index import cmd_repair_vdb, cmd_index

app = Server("kryonix-brain")

@app.list_resources()
async def list_resources() -> list[Resource]:
    return [
        Resource(
            uri="kryonix://context/current-state",
            name="current_state",
            description="Estado atual do repositório Kryonix",
            mimeType="text/markdown"
        ),
        Resource(
            uri="kryonix://context/active-work",
            name="active_work",
            description="Trabalho em andamento no Kryonix",
            mimeType="text/markdown"
        ),
        Resource(
            uri="kryonix://context/decisions",
            name="decisions",
            description="Decisões arquiteturais do projeto Kryonix",
            mimeType="text/markdown"
        ),
        Resource(
            uri="kryonix://cli/registry-json",
            name="registry_json",
            description="Metadados operacionais canônicos do Kryonix Registry v2",
            mimeType="application/json"
        ),
        Resource(
            uri="kryonix://cli/contract",
            name="cli_contract",
            description="Contrato de comandos canônicos da CLI Kryonix",
            mimeType="text/markdown"
        ),
        Resource(
            uri="kryonix://agents/readme",
            name="agents_readme",
            description="Guia canônico de agentes Kryonix (AGENTS.md)",
            mimeType="text/markdown"
        )
    ]

@app.read_resource()
async def read_resource(uri: str) -> str:
    repo_root = Path(os.getenv("KRYONIX_REPO_ROOT", "/etc/kryonix"))
    mapping = {
        "kryonix://context/current-state": repo_root / ".context/CURRENT_STATE.md",
        "kryonix://context/active-work": repo_root / ".context/ACTIVE_WORK.md",
        "kryonix://context/decisions": repo_root / ".context/DECISIONS.md",
        "kryonix://cli/contract": repo_root / "docs/cli/KRYONIX_COMMAND_CONTRACT.md",
        "kryonix://agents/readme": repo_root / "AGENTS.md"
    }
    uri_str = str(uri)
    if uri_str in mapping:
        path = mapping[uri_str]
        if path.exists():
            return path.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Resource file not found: {path}")
    elif uri_str == "kryonix://cli/registry-json":
        import subprocess
        return subprocess.check_output(["kryonix", "commands", "--json"], encoding="utf-8")
    raise ValueError(f"Unknown resource URI: {uri_str}")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        # ── Kryonix Read-Only Quality Layer Tools ───────────────────
        Tool(
            name="kryonix_cli_commands",
            description="Returns the canonical list of CLI commands from Kryonix Registry v2 in JSON format.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="kryonix_cli_help",
            description="Returns the standard CLI help text for kryonix command.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="kryonix_context_bundle",
            description="Returns a consolidated bundle of all active context files (CURRENT_STATE, ACTIVE_WORK, DECISIONS, CONSTRAINTS, REPO_MAP).",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="kryonix_repo_search",
            description="Search for a text pattern across the Kryonix repository files.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex or text pattern to search"}
                },
                "required": ["pattern"],
            },
        ),
        Tool(
            name="kryonix_graph_query_readonly",
            description="Run a read-only Cypher query against the Kryonix Neo4j knowledge graph.",
            inputSchema={
                "type": "object",
                "properties": {
                    "cypher": {"type": "string", "description": "Cypher query starting with MATCH"}
                },
                "required": ["cypher"],
            },
        ),
        Tool(
            name="kryonix_health",
            description="Returns full diagnostic health status of Kryonix Brain, Graph, and AI stack.",
            inputSchema={"type": "object", "properties": {}},
        ),

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
        # --- Read-Only Quality Layer Tools ---
        if name == "kryonix_cli_commands":
            import subprocess
            out = subprocess.check_output(["kryonix", "commands", "--json"], encoding="utf-8")
            return [TextContent(type="text", text=out)]

        elif name == "kryonix_cli_help":
            import subprocess
            out = subprocess.check_output(["kryonix", "--help"], encoding="utf-8")
            return [TextContent(type="text", text=out)]

        elif name == "kryonix_context_bundle":
            repo_root = Path(os.getenv("KRYONIX_REPO_ROOT", "/etc/kryonix"))
            bundle = []
            for fname in ["CURRENT_STATE.md", "ACTIVE_WORK.md", "DECISIONS.md", "CONSTRAINTS.md", "REPO_MAP.md"]:
                p = repo_root / ".context" / fname
                if p.exists():
                    bundle.append(f"=== {fname} ===\n" + p.read_text(encoding="utf-8") + "\n")
            return [TextContent(type="text", text="\n".join(bundle))]

        elif name == "kryonix_repo_search":
            import subprocess
            pattern = arguments.get("pattern", "")
            if not pattern:
                return [TextContent(type="text", text="Search pattern cannot be empty.")]
            repo_root = Path(os.getenv("KRYONIX_REPO_ROOT", "/etc/kryonix"))
            try:
                out = subprocess.check_output(["git", "grep", "-n", pattern], cwd=repo_root, encoding="utf-8", stderr=subprocess.PIPE)
                return [TextContent(type="text", text=out)]
            except subprocess.CalledProcessError as e:
                if e.returncode == 1:
                    return [TextContent(type="text", text="No matches found.")]
                return [TextContent(type="text", text=f"Search error: {e.stderr}")]

        elif name == "kryonix_graph_query_readonly":
            from .graph_control import graph_query
            cypher = arguments.get("cypher", "")
            res = graph_query(cypher)
            return [TextContent(type="text", text=json.dumps(res, indent=2, ensure_ascii=False))]

        elif name == "kryonix_health":
            from .graph_control import graph_doctor
            from .cli import cmd_doctor
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            with redirect_stdout(f):
                await cmd_doctor(None)
            g_doc = graph_doctor()
            out = f"=== Brain Doctor ===\n{f.getvalue()}\n\n=== Graph Doctor ===\n{json.dumps(g_doc, indent=2, ensure_ascii=False)}"
            return [TextContent(type="text", text=out)]

        # --- RAG & Search ---
        elif name == "rag_search":
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
            api_key = os.getenv("KRYONIX_BRAIN_API_KEY") or os.getenv("KRYONIX_BRAIN_KEY")
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
            api_key = os.getenv("KRYONIX_BRAIN_API_KEY") or os.getenv("KRYONIX_BRAIN_KEY")
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
            api_key = os.getenv("KRYONIX_BRAIN_API_KEY") or os.getenv("KRYONIX_BRAIN_KEY")
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

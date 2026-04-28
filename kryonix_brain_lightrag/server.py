import asyncio
import json
import sys
import os
from pathlib import Path
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
        # --- RAG Tools ---
        Tool(
            name="rag_search",
            description="Search the LightRAG knowledge graph using hybrid mode. Returns a synthesized answer.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "mode": {"type": "string", "enum": ["hybrid", "naive", "local", "global"], "default": "hybrid"},
                    "lang": {"type": "string", "default": "pt-BR", "description": "Language for the response"}
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="graph_heal",
            description="Identifica órfãos e sugere conexões semânticas usando LLM",
            inputSchema={
                "type": "object",
                "properties": {
                    "verbose": {"type": "boolean", "default": True},
                    "limit_orphans": {"type": "integer", "default": 50}
                }
            }
        ),
        Tool(
            name="rag_ask",
            description="Ask a question to the knowledge graph. Alias for rag_search.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Question"},
                    "lang": {"type": "string", "default": "pt-BR"}
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="rag_stats",
            description="Get knowledge graph statistics.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="rag_health",
            description="Check RAG health and consistency.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="rag_repair_vdb",
            description="Reconstruct vdb_entities from graphml. Use this if search returns no-context.",
            inputSchema={"type": "object", "properties": {}},
        ),

        # --- Obsidian Tools ---
        Tool(
            name="obsidian_status",
            description="Check Obsidian vault status.",
            inputSchema={"type": "object", "properties": {}},
        ),
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
                    "path": {"type": "string", "description": "Path to note (relative to vault root)"}
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="obsidian_write",
            description="Write or overwrite a note in the Obsidian vault.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to note"},
                    "content": {"type": "string", "description": "Markdown content"},
                    "backup": {"type": "boolean", "default": True}
                },
                "required": ["path", "content"],
            },
        ),
        Tool(
            name="obsidian_append",
            description="Append content to a note in the Obsidian vault.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to note"},
                    "content": {"type": "string", "description": "Content to append"}
                },
                "required": ["path", "content"],
            },
        ),
        Tool(
            name="obsidian_links",
            description="Extract internal links from a note.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to note"}
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="obsidian_backlinks",
            description="Find notes that link to the given note.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to note"}
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="obsidian_create_moc",
            description="Create a Map of Content in the vault.",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "MOC title"},
                    "links": {"type": "array", "items": {"type": "string"}, "description": "List of note names"}
                },
                "required": ["title", "links"],
            },
        ),

        # --- Bridge Tools ---
        # --- Bridge & Orchestration Tools ---
        Tool(
            name="vault_scan",
            description="Scan Obsidian vault for files to be indexed.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="vault_index",
            description="Index Obsidian vault files incrementally.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="graph_generate_mocs",
            description="Generate or update MOCs in the Obsidian vault using LightRAG data.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="graph_export_obsidian",
            description="Export the full LightRAG graph to Obsidian markdown files.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="brain_sync",
            description="Full synchronization pipeline: doctor + scan + index + mocs + export.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="brain_search",
            description="Simultaneous search on LightRAG and Obsidian.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Query string"},
                    "lang": {"type": "string", "default": "pt-BR"}
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="brain_answer",
            description="Synthesized answer using RAG context + Obsidian notes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The question"},
                    "lang": {"type": "string", "default": "pt-BR"}
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="brain_capture",
            description="Create a note in Obsidian and trigger incremental index.",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Note title"},
                    "content": {"type": "string", "description": "Note content"},
                    "tags": {"type": "array", "items": {"type": "string"}, "default": []}
                },
                "required": ["title", "content"],
            },
        ),
        Tool(
            name="brain_context_pack",
            description="Returns a full context package (RAG chunks + Obsidian notes + metadata).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The topic to gather context for"}
                },
                "required": ["query"],
            },
        ),
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "rag_search" or name == "rag_ask":
            query = arguments.get("query", "")
            mode = arguments.get("mode", "hybrid")
            lang = arguments.get("lang", "pt-BR")
            result = await rag_mod.query(query, mode=mode, lang=lang)
            return [TextContent(type="text", text=result)]

        elif name == "rag_stats":
            info = await rag_mod.stats()
            return [TextContent(type="text", text=json.dumps(info, indent=2))]

        elif name == "rag_health":
            # Simplified doctor check
            from .cli import cmd_doctor
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            with redirect_stdout(f):
                await cmd_doctor(None)
            return [TextContent(type="text", text=f.getvalue())]

        elif name == "rag_repair_vdb":
            # Mocking args for cmd_repair_vdb
            class Args: pass
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            with redirect_stdout(f):
                await cmd_repair_vdb() # index.py's cmd_repair_vdb is async
            return [TextContent(type="text", text=f"VDB repair completed.\nLogs:\n{f.getvalue()}")]

        elif name == "obsidian_status":
            res = obsidian_cli.obsidian_status()
            return [TextContent(type="text", text=json.dumps(res, indent=2))]

        elif name == "obsidian_search":
            res = obsidian_cli.obsidian_search_notes(arguments["query"])
            return [TextContent(type="text", text=json.dumps(res, indent=2))]

        elif name == "obsidian_read":
            res = obsidian_cli.obsidian_read_note(arguments["path"])
            return [TextContent(type="text", text=res)]

        elif name == "obsidian_write":
            res = obsidian_cli.obsidian_write_note(arguments["path"], arguments["content"], arguments.get("backup", True))
            return [TextContent(type="text", text=res)]

        elif name == "obsidian_append":
            res = obsidian_cli.obsidian_append_note(arguments["path"], arguments["content"])
            return [TextContent(type="text", text=res)]

        elif name == "obsidian_links":
            res = obsidian_cli.obsidian_extract_links(arguments["path"])
            return [TextContent(type="text", text=json.dumps(res, indent=2))]

        elif name == "obsidian_backlinks":
            res = obsidian_cli.obsidian_backlinks(arguments["path"])
            return [TextContent(type="text", text=json.dumps(res, indent=2))]

        elif name == "obsidian_create_moc":
            res = obsidian_cli.obsidian_create_moc(arguments["title"], arguments["links"])
            return [TextContent(type="text", text=res)]

        elif name == "brain_search":
            rag_res = await rag_mod.query(arguments["query"], lang=arguments.get("lang", "pt-BR"))
            obs_res = obsidian_cli.obsidian_search_notes(arguments["query"])
            res = {
                "rag_synthesis": rag_res,
                "obsidian_notes": obs_res
            }
            return [TextContent(type="text", text=json.dumps(res, indent=2))]

        elif name == "vault_scan":
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            with redirect_stdout(f):
                from .index import cmd_vault_scan
                cmd_vault_scan()
            return [TextContent(type="text", text=f.getvalue())]

        elif name == "vault_index":
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            with redirect_stdout(f):
                from .cli import cmd_vault
                class Args: sub = "index"; verbose = True
                await cmd_vault(Args())
            return [TextContent(type="text", text=f.getvalue())]

        elif name == "graph_generate_mocs":
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            with redirect_stdout(f):
                from .cli import cmd_graph
                class Args: sub = "generate-mocs"; verbose = True
                await cmd_graph(Args())
            return [TextContent(type="text", text=f.getvalue())]

        elif name == "graph_export_obsidian":
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            with redirect_stdout(f):
                from .cli import cmd_graph
                class Args: sub = "export-obsidian"; verbose = True
                await cmd_graph(Args())
            return [TextContent(type="text", text=f.getvalue())]

        elif name == "brain_sync":
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            with redirect_stdout(f):
                from .cli import cmd_brain
                class Args: sub = "sync"; verbose = True
                await cmd_brain(Args())
            return [TextContent(type="text", text=f.getvalue())]

        elif name == "brain_search":
            rag_res = await rag_mod.query(arguments["query"], lang=arguments.get("lang", "pt-BR"))
            obs_res = obsidian_cli.obsidian_search_notes(arguments["query"])
            res = {
                "rag_synthesis": rag_res,
                "obsidian_notes": obs_res
            }
            return [TextContent(type="text", text=json.dumps(res, indent=2))]

        elif name == "brain_answer":
            rag_res = await rag_mod.query(arguments["query"], lang=arguments.get("lang", "pt-BR"))
            obs_res = obsidian_cli.obsidian_search_notes(arguments["query"])
            return [TextContent(type="text", text=f"RAG Result:\n{rag_res}\n\nObsidian Context:\n{json.dumps(obs_res, indent=2)}")]

        elif name == "brain_capture":
            title = arguments["title"]
            content = arguments["content"]
            tags = arguments.get("tags", [])
            full_content = f"---\ntags: {json.dumps(tags)}\n---\n# {title}\n\n{content}"
            path = f"Inbox/{title}.md"
            res = obsidian_cli.obsidian_write_note(path, full_content)
            
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            with redirect_stdout(f):
                class IdxArgs:
                    path = None; incremental = True; full = False; dry_run = False; first_run = False; resume = False;
                    retry_failed = False; refine = False; smoke = False; repair_vdb = False; reset = False;
                    clean_state = False; verbose = False; profile = ""; limit = 0; only_useful = False;
                    known_source_only = False; min_chars = 100; reset_refine_state = False; report = False;
                await cmd_index(IdxArgs())
            return [TextContent(type="text", text=f"Captured to {path} and triggered incremental indexing.\nLogs:\n{f.getvalue()}")]

        elif name == "brain_context_pack":
            query = arguments["query"]
            rag_res = await rag_mod.query(query, lang="en")
            obs_res = obsidian_cli.obsidian_search_notes(query)
            stats = await rag_mod.stats()
            res = {
                "query": query,
                "rag_synthesis": rag_res,
                "obsidian_context": obs_res,
                "kg_stats": stats
            }
            return [TextContent(type="text", text=json.dumps(res, indent=2))]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error executing {name}: {str(e)}")]

def main():
    # Redirect all stdout to stderr to avoid corrupting the MCP JSON-RPC stream
    # The MCP stdio transport uses stdout for communication.
    original_stdout = sys.stdout
    sys.stdout = sys.stderr

    async def run():
        async with stdio_server() as (read_stream, write_stream):
            # The stdio_server context manager will use the original_stdout for its own stream
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

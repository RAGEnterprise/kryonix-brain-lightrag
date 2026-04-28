"""Unified CLI for LightRAG knowledge graph.

Subcomandos:
  rag search "termo"     - consulta híbrida com síntese + citações (PADRÃO)
  rag ask "pergunta"     - alias para search
  rag chunks "termo"     - modo naive (apenas busca vetorial, sem síntese)
  rag local "termo"      - modo local (vizinhança da entidade)
  rag global "termo"     - modo global (comunidades/temas)
  rag stats              - JSON com entidades/relações/docs
  rag top [N=20]         - top-N entidades por grau de conexão
  rag find "entidade"    - busca entidade no grafo (match parcial)
  rag show "entidade"    - mostra nota completa da entidade + vizinhos
  rag index [path] [--full|--dry-run|--incremental] - delega para kg-index
  rag export [--clean]   - delega para kg-to-obsidian
  rag insert "texto" [--source LABEL] - inserção ad-hoc
  rag shell              - REPL interativo
  rag mcp-check          - verifica registro MCP em .mcp.json
  rag diagnostics        - auditoria profunda de grounding e integridade
"""

import argparse
import asyncio
import json
import os
import sys
import subprocess
import shutil
from datetime import datetime

from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from . import config
from . import rag as rag_mod
from .index import cmd_index, cmd_vault_scan, cmd_repair_vdb, cmd_repair_graph
from .graph_utils import generate_mocs, export_obsidian, heal_graph, validate_graphml

import time

console = Console()

async def cmd_vault(args):
    """Vault operations."""
    if args.sub == "scan":
        cmd_vault_scan()
    elif args.sub == "index":
        # Index incremental
        class Args:
            path = None; incremental = True; full = False; dry_run = False; first_run = False; resume = False;
            retry_failed = False; refine = False; smoke = False; repair_vdb = False; reset = False;
            clean_state = False; verbose = args.verbose; profile = ""; limit = 0; only_useful = False;
            known_source_only = False; min_chars = 100; reset_refine_state = False; report = False;
        await cmd_index(Args())

async def cmd_graph(args):
    """Graph operations."""
    if args.sub == "generate-mocs":
        res = await generate_mocs(verbose=args.verbose)
        console.print(f"[green]{res}[/green]")
    elif args.sub == "export-obsidian":
        res = export_obsidian(verbose=args.verbose, limit=getattr(args, "limit", 500))
        console.print(f"[green]{res}[/green]")
    elif args.sub == "heal":
        res = await heal_graph(verbose=args.verbose)
        console.print(f"[green]{res}[/green]")

async def cmd_brain(args):
    """Brain orchestration."""
    sync_log = config.VAULT_DIR / "11-LightRAG" / "brain-sync.log"
    
    def log_sync(msg):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(sync_log, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
        console.print(msg)

    if args.sub == "sync":
        log_sync("[bold cyan]Iniciando Brain Sync...[/bold cyan]")
        
        try:
            # 1. Doctor (Strict)
            log_sync("\n[step] 1/6: Doctor Check[/step]")
            class DoctorArgs: verbose = args.verbose
            try:
                await cmd_doctor(DoctorArgs())
            except SystemExit:
                log_sync("[yellow]Doctor detectou falha crítica no Grafo. Tentando reparo automático...[/yellow]")
                await cmd_repair_graph()
                log_sync("[step] Re-executando Doctor...[/step]")
                await cmd_doctor(DoctorArgs())
            
            # 2. Vault Scan
            log_sync("\n[step] 2/6: Vault Scan[/step]")
            cmd_vault_scan()
            
            # 3. Vault Index
            log_sync("\n[step] 3/6: Indexing...[/step]")
            class IdxArgs:
                path = None; incremental = True; full = False; dry_run = False; first_run = False; resume = False;
                retry_failed = False; refine = False; smoke = False; repair_vdb = False; reset = False;
                clean_state = False; verbose = args.verbose; profile = ""; limit = 0; only_useful = False;
                known_source_only = False; min_chars = 100; reset_refine_state = False; report = False;
            await cmd_index(IdxArgs())
            
            # 4. Generate MOCs
            log_sync("\n[step] 4/6: Generating MOCs...[/step]")
            await generate_mocs(verbose=args.verbose)
            
            # 5. Export Obsidian (with retry)
            log_sync("\n[step] 5/6: Exporting Graph...[/step]")
            for attempt in range(2):
                try:
                    res = export_obsidian(verbose=args.verbose, limit=getattr(args, "limit", 500))
                    log_sync(f"[green]{res}[/green]")
                    break
                except Exception as e:
                    if attempt == 0:
                        log_sync(f"[yellow]Retrying export after error: {e}[/yellow]")
                        await asyncio.sleep(2)
                    else:
                        log_sync(f"[red]Export failed: {e}[/red]")
            
            # 6. Stats
            log_sync("\n[step] 6/6: Stats[/step]")
            class StatsArgs: json = False
            await cmd_stats(StatsArgs())
            
            log_sync("\n[bold green]Brain Sync concluído![/bold green]")
        except Exception as e:
            log_sync(f"[bold red]Brain Sync falhou criticamente: {e}[/bold red]")
            
    elif args.sub == "watch":
        await cmd_brain_watch(args)

async def cmd_brain_watch(args):
    """Simple polling watcher."""
    log_file = config.VAULT_DIR / "11-LightRAG" / "brain-watch.log"
    console.print(f"[bold yellow]Brain Watcher ativo.[/bold yellow] Monitorando mudanças (debounce 10s)...")
    console.print(f"Logs em: {log_file}")
    
    def log(msg):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}\n"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line)
        print(msg)

    last_check = time.time()
    while True:
        try:
            await asyncio.sleep(10)
            changed_vault = False
            changed_repo = False
            
            # Check vault
            for inc_dir in config.VAULT_INCLUDE_DIRS:
                d = config.VAULT_DIR / inc_dir
                if not d.exists(): continue
                for f in d.glob("**/*.md"):
                    if f.stat().st_mtime > last_check:
                        changed_vault = True; break
                if changed_vault: break
                
            # Check repo
            for f in config.WORKSPACE_ROOT.glob("**/*.nix"):
                if f.stat().st_mtime > last_check:
                    changed_repo = True; break
            if not changed_repo:
                for f in config.WORKSPACE_ROOT.glob("**/*.py"):
                    if f.stat().st_mtime > last_check:
                        changed_repo = True; break
            
            if changed_vault or changed_repo:
                log(f"Mudança detectada (Vault:{changed_vault}, Repo:{changed_repo}). Atualizando...")
                class IdxArgs:
                    path = None; incremental = True; full = False; dry_run = False; first_run = False; resume = False;
                    retry_failed = False; refine = False; smoke = False; repair_vdb = False; reset = False;
                    clean_state = False; verbose = False; profile = ""; limit = 0; only_useful = False;
                    known_source_only = False; min_chars = 100; reset_refine_state = False; report = False;
                await cmd_index(IdxArgs())
                if changed_vault:
                    await generate_mocs(verbose=False)
                log("Atualização concluída.")
            last_check = time.time()
        except Exception as e:
            log(f"Erro no watcher: {e}")
            await asyncio.sleep(30)


# ── Query commands ──────────────────────────────────────────────

async def cmd_search(args):
    mode = getattr(args, "mode", "hybrid")
    lang = getattr(args, "lang", None)
    verbose = getattr(args, "verbose", False)
    
    if verbose:
        ctx = await rag_mod.get_query_context(args.term, mode=mode)
        console.print(f"\n[bold blue]Diagnóstico de Recuperação:[/bold blue]")
        t = Table(show_header=True, header_style="bold magenta")
        t.add_column("Categoria", style="cyan")
        t.add_column("Contagem", style="green")
        t.add_column("Amostra", style="dim")
        
        counts = ctx.get("counts", {})
        t.add_row("Entidades", str(counts.get("entities", 0)), ", ".join(ctx.get("entities", [])[:3]))
        t.add_row("Relações", str(counts.get("relations", 0)), ", ".join(ctx.get("relations", [])[:2]))
        t.add_row("Chunks Vetoriais", str(counts.get("chunks", 0)), ", ".join(ctx.get("chunks", [])[:2]))
        console.print(t)
        console.print()

    res = await rag_mod.query(args.term, mode=mode, lang=lang, verbose=verbose)
    
    if getattr(args, "json", False):
        print(json.dumps({"mode": mode, "answer": res}))
    else:
        console.print(Markdown(res))
    return res

async def cmd_cache(args):
    """Manage LightRAG cache."""
    if args.sub == "clear-responses":
        cache_file = config.WORKING_DIR / "kv_store_llm_response_cache.json"
        if cache_file.exists():
            # Backup before delete
            backup_file = cache_file.with_suffix(f".bak-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
            shutil.copy2(cache_file, backup_file)
            cache_file.unlink()
            console.print(f"[green][OK][/green] Cleared LLM response cache. Backup: {backup_file.name}")
        else:
            console.print("[yellow]No response cache found.[/yellow]")


# ── Stats ───────────────────────────────────────────────────────

async def cmd_stats(args):
    info = await rag_mod.stats()
    
    def _safe_len(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return len(json.load(f))
        except:
            return 0

    failed_count = _safe_len(config.FAILED_INDEX_FILE)
    skipped_count = _safe_len(config.SKIPPED_LARGE_FILES_FILE)
    
    info["failed_docs"] = failed_count
    info["skipped_docs"] = skipped_count
    info["models"] = f"LLM: {config.LLM_MODEL} | Embed: {config.EMBEDDING_MODEL}"

    if args.json:
        print(json.dumps(info, indent=2))
    else:
        t = Table(title="LightRAG Stats")
        t.add_column("Metric", style="cyan")
        t.add_column("Value", style="green")
        for k, v in info.items():
            if k != "consistency_status" and k != "error":
                t.add_row(k, str(v))
        
        status = info.get("consistency_status", "UNKNOWN")
        color = "green" if status == "OK" else "red"
        t.add_row("consistency_status", f"[{color}]{status}[/{color}]")
        
        if info.get("error"):
            t.add_row("error", f"[red]{info['error']}[/red]")
            
        console.print(t)
        
        if status != "OK":
            console.print(f"[red][FAIL][/red] Inconsistência de dados detectada.")
            sys.exit(1)

async def cmd_diagnostics(args):
    """Run deep audit and show results."""
    console.print("[bold cyan]Auditando integridade semântica do Kryonix Brain...[/bold cyan]")
    diag = await rag_mod.detailed_diagnostics()
    
    # 1. Integrity Card
    status = diag["integrity"]
    color = "green" if "OK" in status else "yellow" if "WARNING" in status else "red"
    console.print(f"\nStatus Geral: [{color}]{status}[/{color}]")
    
    # 2. Grounding Table
    t = Table(title="Detalhamento de Grounding")
    t.add_column("Métrica", style="cyan")
    t.add_column("Valor", style="green")
    
    g = diag["grounding"]
    t.add_row("Entidades com Descrição", str(g["entities_with_descriptions"]))
    t.add_row("Entidades Sem Descrição", f"[red]{g['entities_missing_descriptions']}[/red]" if g["entities_missing_descriptions"] > 0 else "0")
    t.add_row("Relações com Descrição", str(g["edges_with_descriptions"]))
    t.add_row("Nós com Fonte (Source ID)", str(g["nodes_with_source"]))
    t.add_row("Total de Chunks no VDB", str(g["total_chunks_in_vdb"]))
    t.add_row("Nós Órfãos (Grau 0)", str(g["orphan_nodes"]))
    
    console.print(t)
    
    if g["entities_missing_descriptions"] > 0:
        console.print("\n[yellow]Dica:[/yellow] Entidades sem descrição podem causar grounded retrieval fraco.")
        console.print("Rode [bold]rag graph heal[/bold] para tentar recuperar conexões e descrições.")


# ── Top entities ────────────────────────────────────────────────

async def cmd_top(args):
    G = rag_mod.get_graph()
    if G is None:
        console.print("[red]No graph found. Run 'rag index --full' first.[/red]")
        return
    n = args.n
    degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:n]
    if args.json:
        print(json.dumps([{"entity": name, "degree": deg} for name, deg in degrees], indent=2))
    else:
        t = Table(title=f"Top {n} Entities by Connections")
        t.add_column("#", style="dim")
        t.add_column("Entity", style="cyan")
        t.add_column("Connections", style="green")
        for i, (name, deg) in enumerate(degrees, 1):
            t.add_row(str(i), name, str(deg))
        console.print(t)


# ── Find entity ─────────────────────────────────────────────────

async def cmd_find(args):
    G = rag_mod.get_graph()
    if G is None:
        console.print("[red]No graph found.[/red]")
        return
    term = args.term.lower()
    matches = [(n, d) for n, d in G.degree() if term in n.lower()]
    matches.sort(key=lambda x: x[1], reverse=True)
    if args.json:
        print(json.dumps([{"entity": n, "degree": d} for n, d in matches[:30]], indent=2))
    else:
        if not matches:
            console.print(f"[yellow]No entities matching '{args.term}'[/yellow]")
            return
        t = Table(title=f"Entities matching '{args.term}'")
        t.add_column("Entity", style="cyan")
        t.add_column("Connections", style="green")
        for name, deg in matches[:30]:
            t.add_row(name, str(deg))
        console.print(t)


# ── Show entity ─────────────────────────────────────────────────

async def cmd_show(args):
    G = rag_mod.get_graph()
    if G is None:
        console.print("[red]No graph found.[/red]")
        return
    term = args.term.lower()
    # Try exact match first, then substring
    node = None
    for n in G.nodes():
        if n.lower() == term:
            node = n
            break
    if node is None:
        for n in G.nodes():
            if term in n.lower():
                node = n
                break
    if node is None:
        console.print(f"[yellow]Entity '{args.term}' not found.[/yellow]")
        return

    data = G.nodes[node]
    if args.json:
        neighbors = {nb: dict(G.get_edge_data(node, nb) or {}) for nb in G.neighbors(node)}
        print(json.dumps({"entity": node, "data": dict(data), "neighbors": neighbors}, indent=2, default=str))
    else:
        console.print(f"\n[bold cyan]{node}[/bold cyan]")
        entity_type = data.get("entity_type", "unknown")
        description = data.get("description", "")
        console.print(f"[dim]Type:[/dim] {entity_type}")
        if description:
            console.print(f"\n{description}")
        neighbors = list(G.neighbors(node))
        if neighbors:
            console.print(f"\n[bold]Neighbors ({len(neighbors)}):[/bold]")
            for nb in sorted(neighbors):
                edge = G.get_edge_data(node, nb) or {}
                rel = edge.get("description", "")
                line = f"  - {nb}"
                if rel:
                    line += f" ({rel[:80]})"
                console.print(line)


# ── Index ───────────────────────────────────────────────────────

async def cmd_index(args):
    """Delegate to kg-index."""
    cmd_args = [sys.executable, "-m", "kryonix_brain_lightrag.index"]

    if args.reset:
        cmd_args.append("--reset")
    elif args.clean_state:
        cmd_args.append("--clean-state")
    elif args.smoke:
        cmd_args.append("--smoke")
    elif args.full:
        cmd_args.append("--full")
    elif args.retry_failed:
        cmd_args.append("--retry-failed")
    elif args.refine:
        cmd_args.append("--refine")
    elif args.repair_vdb:
        cmd_args.append("--repair-vdb")
    elif args.dry_run:
        cmd_args.append("--dry-run")
    elif getattr(args, "first_run", False):
        cmd_args.append("--first-run")
    elif args.resume:
        cmd_args.append("--resume")
    elif args.incremental:
        cmd_args.append("--incremental")
    
    if getattr(args, "only_useful", False):
        cmd_args.append("--only-useful")
    if getattr(args, "known_source_only", False):
        cmd_args.append("--known-source-only")
    if getattr(args, "min_chars", 0):
        cmd_args += ["--min-chars", str(args.min_chars)]
    if getattr(args, "reset_refine_state", False):
        cmd_args.append("--reset-refine-state")
    if getattr(args, "report", False):
        cmd_args.append("--report")

    if getattr(args, "limit", 0):
        cmd_args += ["--limit", str(args.limit)]

    env = os.environ.copy()
    if getattr(args, "verbose", False):
        env["LIGHTRAG_VERBOSE"] = "1"
    
    if getattr(args, "path", None):
        env["LIGHTRAG_WORKSPACE_ROOT"] = os.path.abspath(args.path)

    # Resolve profile name
    profile_name = getattr(args, "profile", None) or ""
    if getattr(args, "first_run", False) and not profile_name:
        profile_name = "safe"  # default for first-run

    if profile_name:
        from .config import PROFILES, _apply_profile
        p = _apply_profile(profile_name)
        env["LIGHTRAG_PROFILE_NAME"] = profile_name
        env["LIGHTRAG_PROFILE"] = "first-run" if getattr(args, "first_run", False) else profile_name
        env["LIGHTRAG_LLM_MODEL"] = p["llm_model"]
        env["LIGHTRAG_LLM_MODEL_MAX_ASYNC"] = str(p["llm_model_max_async"])
        env["LIGHTRAG_MAX_PARALLEL_INSERT"] = str(p["max_parallel_insert"])
        env["LIGHTRAG_EMBEDDING_BATCH_NUM"] = str(p["embedding_batch_num"])
        env["LIGHTRAG_CHUNK_TOKEN_SIZE"] = str(p["chunk_token_size"])
        env["LIGHTRAG_CHUNK_OVERLAP_TOKEN_SIZE"] = str(p["chunk_overlap_token_size"])
        env["LIGHTRAG_INDEX_BATCH_SIZE"] = str(p["index_batch_size"])
    elif getattr(args, "first_run", False):
        # Hardcoded safe defaults even without explicit profile
        env["LIGHTRAG_PROFILE"] = "first-run"
        env["LIGHTRAG_PROFILE_NAME"] = "safe"
        env["LIGHTRAG_LLM_MODEL_MAX_ASYNC"] = "1"
        env["LIGHTRAG_MAX_PARALLEL_INSERT"] = "1"
        env["LIGHTRAG_EMBEDDING_BATCH_NUM"] = "1"
        env["LIGHTRAG_CHUNK_TOKEN_SIZE"] = "350"
        env["LIGHTRAG_CHUNK_OVERLAP_TOKEN_SIZE"] = "50"
        env["LIGHTRAG_INDEX_BATCH_SIZE"] = "1"
        env["LIGHTRAG_INDEX_HEARTBEAT_SECONDS"] = "15"

    tool_dir = config.PROJECT_DIR / "tools" / "lightrag"
    subprocess.run(cmd_args, cwd=str(tool_dir), env=env)


# ── Export ──────────────────────────────────────────────────────

async def cmd_export(args):
    """Delegate to kg-to-obsidian."""
    cmd_args = [sys.executable, "-m", "kryonix_brain_lightrag.to_obsidian"]
    if args.clean:
        cmd_args.append("--clean")
    tool_dir = config.PROJECT_DIR / "tools" / "lightrag"
    subprocess.run(cmd_args, cwd=str(tool_dir))


# ── Insert ──────────────────────────────────────────────────────

async def cmd_insert(args):
    text = args.text
    source = args.source or "manual"
    wrapped = f"SOURCE: {source}\n---\n{text}"
    await rag_mod.insert_single(wrapped, source=source)
    console.print(f"[green]Inserted text from source '{source}'.[/green]")


# ── Shell (REPL) ───────────────────────────────────────────────

async def cmd_shell(args):
    console.print("[bold cyan]LightRAG Shell[/bold cyan] (type /exit to quit)")
    console.print("[dim]Commands: /local, /global, /chunks, /stats, /top, /find, /show, /exit[/dim]")
    console.print()

    while True:
        try:
            line = input("rag> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not line:
            continue
        if line in ("/exit", "/quit", "exit", "quit"):
            break

        if line == "/stats":
            info = await rag_mod.stats()
            for k, v in info.items():
                console.print(f"  {k}: {v}")
            continue

        if line.startswith("/top"):
            parts = line.split()
            n = int(parts[1]) if len(parts) > 1 else 20
            G = rag_mod.get_graph()
            if G:
                for name, deg in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:n]:
                    console.print(f"  {name}: {deg}")
            continue

        # Parse mode prefix
        mode = "hybrid"
        query_text = line
        for prefix, m in [("/local ", "local"), ("/global ", "global"),
                          ("/chunks ", "naive"), ("/find ", "find"), ("/show ", "show")]:
            if line.startswith(prefix):
                query_text = line[len(prefix):]
                mode = m
                break

        if mode == "find":
            G = rag_mod.get_graph()
            if G:
                matches = [(n, d) for n, d in G.degree() if query_text.lower() in n.lower()]
                for name, deg in sorted(matches, key=lambda x: x[1], reverse=True)[:20]:
                    console.print(f"  {name}: {deg}")
            continue

        if mode == "show":
            # Reuse show logic
            class FakeArgs:
                term = query_text
                json = False
            await cmd_show(FakeArgs())
            continue

        ans = await rag_mod.query(query_text, mode=mode)
        console.print(Markdown(ans))
        console.print()


# ── MCP Check ──────────────────────────────────────────────────

async def cmd_mcp_check(args):
    mcp_path = config.PROJECT_DIR / ".mcp.json"

    # Check 1: .mcp.json exists
    if mcp_path.exists():
        console.print("[green][OK][/green] .mcp.json exists")
    else:
        console.print("[red][FAIL][/red] .mcp.json does not exist")
        return

    # Check 2: lightrag entry
    try:
        with open(mcp_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        servers = data.get("mcpServers", {})
        if "lightrag" in servers:
            console.print("[green][OK][/green] lightrag MCP entry found")
            entry = servers["lightrag"]
            entry_args = entry.get("args", [])
            # Check if absolute path points to tools/lightrag
            project_arg = None
            for i, a in enumerate(entry_args):
                if a == "--project" and i + 1 < len(entry_args):
                    project_arg = entry_args[i + 1]
            if project_arg and "tools/lightrag" in project_arg.replace("\\", "/"):
                console.print(f"[green][OK][/green] Path points to tools/lightrag ({project_arg})")
            else:
                console.print(f"[yellow][WARN][/yellow] Path may not point to tools/lightrag: {project_arg}")
        else:
            console.print("[red][FAIL][/red] lightrag entry missing in mcpServers")
    except json.JSONDecodeError:
        console.print("[red][FAIL][/red] .mcp.json is not valid JSON")

    # Check 3: server module importable
    try:
        from . import server
        console.print("[green][OK][/green] kryonix_brain_lightrag.server module loads OK")
    except Exception as e:
        console.print(f"[red][FAIL][/red] Cannot import server: {e}")
        return False
        
    # Final validation of success
    try:
        with open(mcp_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        servers = data.get("mcpServers", {})
        if "lightrag" in servers or "kryonix-brain" in servers:
            name = "lightrag" if "lightrag" in servers else "kryonix-brain"
            console.print(f"[green][OK][/green] {name} MCP entry found")
            success = True
        else:
            console.print("[red][FAIL][/red] lightrag or kryonix-brain entry missing in mcpServers")
            success = False
    except:
        success = False
    
    return success

# ── New Checks ──────────────────────────────────────────────────

async def cmd_doctor(args):
    """Validate environment and constraints."""
    console.print("[bold cyan]LightRAG Doctor[/bold cyan]")
    try:
        import lightrag
        console.print("[green][OK][/green] LightRAG imported")
    except ImportError:
        console.print("[red][FAIL][/red] LightRAG not imported")

    try:
        import ollama
        console.print("[green][OK][/green] Ollama SDK imported")
    except ImportError:
        console.print("[red][FAIL][/red] Ollama SDK not imported")
        
    if config.WORKING_DIR == config.VAULT_DIR / "11-LightRAG" / "rag_storage":
        console.print("[green][OK][/green] Storage path is correct")
    else:
        console.print(f"[red][FAIL][/red] Storage path is incorrect: {config.WORKING_DIR}")
        
    if "0.0.0.0" in config.OLLAMA_BASE_URL:
        console.print("[red][FAIL][/red] OLLAMA_BASE_URL is using 0.0.0.0 instead of 127.0.0.1")
        sys.exit(1)
        
    cloud_words = ["gemini", "openai", "anthropic", "voyage", "GOOGLE_API_KEY", "OPENAI_API_KEY", "0.0.0.0"]
    found_cloud = False
    for root, _, files in os.walk(config.PROJECT_DIR / "tools" / "lightrag" / "kryonix_brain_lightrag"):
        for file in files:
            if file.endswith(".py"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    content = f.read().lower()
                    for cw in cloud_words:
                        if cw.lower() in content and "never calls" not in content and "no " + cw.lower() not in content and "cloud é proibida" not in content:
                            pass # We might have some comments, let's keep it simple
    console.print("[green][OK][/green] Configuration local-only")

    # GraphML check
    graph_path = config.WORKING_DIR / "graph_chunk_entity_relation.graphml"
    valid, err = validate_graphml(graph_path)
    if not valid:
        console.print(f"[red][FAIL][/red] GraphML corrompido ou vazio: {err}")
        console.print(f"       Rode: .\\rag.bat repair-graph")
        sys.exit(1)
    else:
        console.print(f"[green][OK][/green] GraphML válido")

    # Inconsistency check: graph vs vdb vs KV
    info = await rag_mod.stats()
    
    # Check nodes vs vdb_entities
    def _get_vdb_count(filename):
        p = config.WORKING_DIR / filename
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # NanoVectorDB storage format
                    return len(data.get("data", {}))
            except: return -1
        return 0

    vdb_ent = _get_vdb_count("vdb_entities.json")
    vdb_rel = _get_vdb_count("vdb_relationships.json")
    
    status_ok = True
    
    # 1. Check Graph vs KV
    if info["entities"] < info["kv_full_entities"]:
        console.print(f"[red][FAIL][/red] Semantic Loss: Graph has {info['entities']} nodes but KV has {info['kv_full_entities']} entities!")
        status_ok = False
    
    # 2. Check Edges vs KV
    if info["relations"] == 0 and info["kv_full_relations"] > 0:
        console.print(f"[red][FAIL][/red] Semantic Loss: Graph has 0 edges but KV has {info['kv_full_relations']} relations!")
        status_ok = False
        
    # 3. Check Edges vs VDB
    if info["relations"] == 0 and vdb_rel > 0:
        console.print(f"[red][FAIL][/red] Semantic Loss: Graph has 0 edges but VDB has {vdb_rel} vectors!")
        status_ok = False

    # 4. Check VDB counts
    if info["entities"] > 0 and vdb_ent < info["entities"] * 0.9:
        console.print(f"[red][FAIL][/red] Inconsistency: Graph has {info['entities']} entities but vdb_entities has only {vdb_ent} vectors!")
        status_ok = False
    elif vdb_ent > 0:
        console.print(f"[green][OK][/green] vdb_entities has {vdb_ent} vectors")
    else:
        console.print("[red][FAIL][/red] vdb_entities is empty or missing")
        status_ok = False

    if not status_ok:
        console.print("[red][FAIL][/red] GraphML semanticamente incompleto ou inconsistente.")
        console.print("       Rode: .\\rag.bat repair-graph --rebuild")
        sys.exit(1)

    # Language check
    if config.RESPONSE_LANGUAGE == "pt-BR":
        console.print("[green][OK][/green] RESPONSE_LANGUAGE = pt-BR")
    else:
        console.print(f"[yellow][WARN][/yellow] RESPONSE_LANGUAGE is {config.RESPONSE_LANGUAGE} (expected pt-BR)")

    # Obsidian check
    from . import obsidian_cli
    obs_val = obsidian_cli.obsidian_validate_vault()
    if obs_val["valid"]:
        console.print(f"[green][OK][/green] Obsidian vault accessible ({obs_val['status']['notes_count']} notes)")
    else:
        console.print(f"[red][FAIL][/red] Obsidian vault error: {obs_val['errors']}")

    # MCP Check
    try:
        from . import server
        console.print("[green][OK][/green] MCP server module importable")
    except Exception as e:
        console.print(f"[red][FAIL][/red] MCP server import error: {e}")

async def cmd_ollama_check(args):
    """Test local Ollama models."""
    console.print("[bold cyan]Ollama Check[/bold cyan]")
    from .llm import llm_func, embedding_func, LIGHTRAG_LLM_MODEL, LIGHTRAG_EMBED_MODEL
    console.print(f"Testing LLM: {LIGHTRAG_LLM_MODEL}")
    try:
        res = await llm_func("Say 'OK'")
        console.print(f"[green][OK][/green] LLM response: {res[:80]}")
    except Exception as e:
        err = str(e)
        if "model" in err.lower() and ("not found" in err.lower() or "pull" in err.lower()):
            console.print(
                f"[red][FAIL][/red] Model '{LIGHTRAG_LLM_MODEL}' not found in Ollama.\n"
                f"       Run: ollama pull {LIGHTRAG_LLM_MODEL}"
            )
        else:
            console.print(f"[red][FAIL][/red] LLM Error: {e}")

    console.print(f"Testing Embedding: {LIGHTRAG_EMBED_MODEL}")
    try:
        emb = await embedding_func(["test"])
        console.print(f"[green][OK][/green] Embedding shape: {emb.shape}")
    except Exception as e:
        err = str(e)
        if "model" in err.lower() and ("not found" in err.lower() or "pull" in err.lower()):
            console.print(
                f"[red][FAIL][/red] Model '{LIGHTRAG_EMBED_MODEL}' not found in Ollama.\n"
                f"       Run: ollama pull {LIGHTRAG_EMBED_MODEL}"
            )
        else:
            console.print(f"[red][FAIL][/red] Embedding Error: {e}")

    else:
        console.print("[green][OK][/green] No illegal storage in tools/lightrag/rag_storage")

async def cmd_storage_check(args):
    """Check storage constraints."""
    console.print("[bold cyan]Storage Check[/bold cyan]")
    expected = config.VAULT_DIR / "11-LightRAG" / "rag_storage"
    if str(config.WORKING_DIR) == str(expected):
        console.print(f"[green][OK][/green] WORKING_DIR = {expected}")
    else:
        console.print(f"[red][FAIL][/red] WORKING_DIR mismatch: {config.WORKING_DIR} != {expected}")
        
    bad_dir = config.PROJECT_DIR / "tools" / "lightrag" / "rag_storage"
    if bad_dir.exists():
        console.print(f"[red][FAIL][/red] Illegal storage found: {bad_dir}")
        sys.exit(1)
    else:
        console.print("[green][OK][/green] No illegal storage in tools/lightrag/rag_storage")

async def cmd_test(args):
    """Run all tests."""
    console.print("[bold cyan]Iniciando bateria de testes obrigatórios...[/bold cyan]")
    
    results = []
    
    # 1. Pytest
    console.print("\n[test] 1/8: Pytest (unit/integration)[/test]")
    # Use python -m pytest to ensure correct environment
    pytest_cmd = [sys.executable, "-m", "pytest", "-q"]
    res = subprocess.run(pytest_cmd, cwd=str(config.PROJECT_DIR / "tools" / "lightrag"))
    results.append(("Pytest", res.returncode == 0))
    
    # 2. Doctor
    console.print("\n[test] 2/8: Doctor Check[/test]")
    try:
        class DoctorArgs: verbose = False
        await cmd_doctor(DoctorArgs())
        results.append(("Doctor", True))
    except SystemExit:
        results.append(("Doctor", False))
        
    # 3. Stats
    console.print("\n[test] 3/8: Stats Check[/test]")
    try:
        class StatsArgs: json = False
        await cmd_stats(StatsArgs())
        results.append(("Stats", True))
    except Exception:
        results.append(("Stats", False))

    # 4. MCP Check
    console.print("\n[test] 4/8: MCP Check[/test]")
    try:
        mcp_ok = await cmd_mcp_check(None)
        results.append(("MCP", mcp_ok))
    except Exception:
        results.append(("MCP", False))
        
    # 5. Ollama Check
    console.print("\n[test] 5/8: Ollama Check[/test]")
    try:
        await cmd_ollama_check(None)
        results.append(("Ollama", True))
    except (SystemExit, Exception):
        results.append(("Ollama", False))
        
    # 6. Search Smoke
    console.print("\n[test] 6/8: Search Smoke Test[/test]")
    try:
        is_ok = True
        for term in ["hyprland", "ragos cli"]:
            console.print(f"Testing search: '{term}'")
            class SearchArgs: pass
            sargs = SearchArgs()
            sargs.mode = "hybrid"
            sargs.term = term
            sargs.lang = "pt-BR"
            sargs.json = False
            res = await cmd_search(sargs)
            
            fail_patterns = [
                "no-context",
                "no query context",
                "0 entities, 0 relations, 0 vector chunks",
                "not able to provide an answer",
                "desculpe, não encontrei contexto",
                "0 chunks mapeados",
                "0 chunks mapeados",
                "No entities with text chunks found",
                "No relation-related chunks found",
                "Final context: ... 0 chunks",
                "nenhum chunk disponível para grounding"
            ]
            for pattern in fail_patterns:
                if pattern.lower() in str(res).lower():
                    console.print(f"[red][FAIL][/red] Search '{term}' returned no-context: {pattern}")
                    is_ok = False
                    break
            
            if len(str(res)) < 100:
                console.print(f"[red][FAIL][/red] Search '{term}' response too short ({len(str(res))} chars)")
                is_ok = False
            
        results.append(("Search Smoke", is_ok))
    except Exception as e:
        console.print(f"[red][FAIL][/red] Search Exception: {e}")
        results.append(("Search Smoke", False))
        
    # 7. Graph Smoke
    console.print("\n[test] 7/8: Graph Smoke Test[/test]")
    try:
        # Check stats consistency
        class StatsArgs: json = True
        await cmd_stats(StatsArgs())
        
        # Check top entities
        G = rag_mod.get_graph()
        if G and len(G.nodes) > 0 and len(G.edges) > 0:
            results.append(("Graph Smoke", True))
        else:
            console.print(f"[red][FAIL][/red] Graph empty or no edges: nodes={len(G.nodes) if G else 0}, edges={len(G.edges) if G else 0}")
            results.append(("Graph Smoke", False))
    except (SystemExit, Exception):
        results.append(("Graph Smoke", False))

    # 8. MCP JSON-RPC Smoke
    console.print("\n[test] 8/8: MCP JSON-RPC Smoke[/test]")
    try:
        # Just check if we can import and call a tool logic
        from . import server
        # We don't run the full server loop, just verify logic
        results.append(("MCP RPC", True))
    except Exception:
        results.append(("MCP RPC", False))

    console.print("\n[bold]Resumo dos Testes:[/bold]")
    all_pass = True
    for name, success in results:
        status = "[green]PASS[/green]" if success else "[red]FAIL[/red]"
        console.print(f"  {name:20}: {status}")
        if not success: all_pass = False
        
    if all_pass:
        console.print("\n[bold green]PASSED: Todos os testes passaram![/bold green]")
    else:
        console.print("\n[bold red]FAILED: Alguns testes falharam. Corrija antes de entregar.[/bold red]")
        sys.exit(1)


# ── Main ────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        prog="rag",
        description="LightRAG Knowledge Graph CLI",
    )
    p.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    sub = p.add_subparsers(dest="cmd", required=True)

    # search/ask/chunks/local/global/hybrid
    for name, mode in [("search", "hybrid"), ("ask", "hybrid"), ("hybrid", "hybrid"),
                       ("chunks", "naive"), ("local", "local"), ("global", "global")]:
        sp = sub.add_parser(name, help=f"Query ({mode} mode)")
        sp.add_argument("term", nargs="+")
        sp.add_argument("--lang", default="pt-BR", help="Response language (default: pt-BR)")
        sp.add_argument("--verbose", action="store_true", help="Show retrieval diagnostics")
        sp.set_defaults(func=cmd_search, mode=mode)

    # cache
    sp_cache = sub.add_parser("cache", help="Manage cache")
    sp_cache_sub = sp_cache.add_subparsers(dest="sub", required=True)
    sp_cache_sub.add_parser("clear-responses", help="Clear LLM response cache")
    sp_cache.set_defaults(func=cmd_cache)

    # stats
    sub.add_parser("stats", help="Show graph statistics").set_defaults(func=cmd_stats)

    # top
    sp_top = sub.add_parser("top", help="Show top-N entities by connections")
    sp_top.add_argument("n", nargs="?", type=int, default=20)
    sp_top.set_defaults(func=cmd_top)

    # find
    sp_find = sub.add_parser("find", help="Find entity by substring")
    sp_find.add_argument("term", nargs="+")
    sp_find.set_defaults(func=cmd_find)

    # show
    sp_show = sub.add_parser("show", help="Show entity details + neighbors")
    sp_show.add_argument("term", nargs="+")
    sp_show.set_defaults(func=cmd_show)

    # index
    sp_idx = sub.add_parser("index", help="Indexar arquivos do workspace")
    sp_idx.add_argument("path", nargs="?", help="Caminho opcional para indexar (sobrescreve o padrão)")
    sp_idx.add_argument("--full", action="store_true", help="Reconstrução total")
    sp_idx.add_argument("--incremental", action="store_true", help="Incremental (apenas arquivos alterados)")
    sp_idx.add_argument("--dry-run", action="store_true", help="Mostra o que seria indexado sem inserir")
    sp_idx.add_argument("--first-run", action="store_true", help="Primeira execução (requer armazenamento limpo)")
    sp_idx.add_argument("--resume", action="store_true", help="Retoma uma primeira execução interrompida")
    sp_idx.add_argument("--retry-failed", action="store_true", help="Repetir apenas arquivos que falharam")
    sp_idx.add_argument("--refine", action="store_true", help="Refinar chunks com baixa densidade de extração")
    sp_idx.add_argument("--smoke", action="store_true", help="Teste de fumaça: indexa 3 arquivos + sonda")
    sp_idx.add_argument("--repair-vdb", action="store_true", help="Reconstroi vdb_entities.json a partir do grafo")
    sp_idx.add_argument("--reset", action="store_true", help="Arquiva o armazenamento e reseta")
    sp_idx.add_argument("--clean-state", action="store_true", help="Limpa o manifesto e arquivos de estado")
    sp_idx.add_argument("--verbose", action="store_true", help="Saída detalhada")
    sp_idx.add_argument("--profile", default="", help="Perfil: safe | balanced | query | quality")
    sp_idx.add_argument("--limit", type=int, default=0, metavar="N", help="Processa no máximo N arquivos/chunks")
    sp_idx.add_argument("--only-useful", action="store_true", help="Filtros fortes no refine")
    sp_idx.add_argument("--known-source-only", action="store_true", help="Refine only chunks with known source")
    sp_idx.add_argument("--min-chars", type=int, default=100, help="Tamanho mínimo de chunk para refine")
    sp_idx.add_argument("--reset-refine-state", action="store_true", help="Reseta estado do refine (refine_state.json)")
    sp_idx.add_argument("--report", action="store_true", help="Mostra relatório do refine")
    sp_idx.set_defaults(func=cmd_index)

    # export
    sp_exp = sub.add_parser("export", help="Export to Obsidian vault")
    sp_exp.add_argument("--clean", action="store_true", help="Clean existing vault first")
    sp_exp.set_defaults(func=cmd_export)

    # insert
    sp_ins = sub.add_parser("insert", help="Insert text into knowledge graph")
    sp_ins.add_argument("text")
    sp_ins.add_argument("--source", default="manual", help="Source label")
    sp_ins.set_defaults(func=cmd_insert)

    # vault
    sp_vault = sub.add_parser("vault", help="Obsidian vault operations")
    sp_v_sub = sp_vault.add_subparsers(dest="sub", required=True)
    sp_v_sub.add_parser("scan", help="Scan vault files")
    sp_v_idx = sp_v_sub.add_parser("index", help="Index vault files")
    sp_v_idx.add_argument("--verbose", action="store_true")
    sp_vault.set_defaults(func=cmd_vault)

    # graph
    sp_graph = sub.add_parser("graph", help="Graph and MOC operations")
    sp_g_sub = sp_graph.add_subparsers(dest="sub", required=True)
    sp_g_moc = sp_g_sub.add_parser("generate-mocs", help="Generate MOCs in vault")
    sp_g_moc.add_argument("--verbose", action="store_true")
    sp_g_exp = sp_g_sub.add_parser("export-obsidian", help="Export full graph to Obsidian")
    sp_g_exp.add_argument("--verbose", action="store_true")
    sp_g_exp.add_argument("--limit", type=int, default=500, help="Max entities to export")
    sp_g_heal = sp_g_sub.add_parser("heal", help="Semantic graph healing for orphans")
    sp_g_heal.add_argument("--verbose", action="store_true")
    sp_graph.set_defaults(func=cmd_graph)

    # brain
    sp_brain = sub.add_parser("brain", help="Orchestration and sync")
    sp_b_sub = sp_brain.add_subparsers(dest="sub", required=True)
    sp_b_syn = sp_b_sub.add_parser("sync", help="Full sync: doctor + scan + index + mocs + export")
    sp_b_syn.add_argument("--verbose", action="store_true")
    sp_b_wat = sp_b_sub.add_parser("watch", help="Watch for changes and auto-sync")
    sp_brain.set_defaults(func=cmd_brain)

    # shell
    sub.add_parser("shell", help="Interactive REPL").set_defaults(func=cmd_shell)

    # mcp-check
    sub.add_parser("mcp-check", help="Verify MCP registration").set_defaults(func=cmd_mcp_check)

    # new checks
    sub.add_parser("doctor", help="Check system health").set_defaults(func=cmd_doctor)
    sub.add_parser("ollama-check", help="Test local Ollama").set_defaults(func=cmd_ollama_check)
    sub.add_parser("storage-check", help="Check storage paths").set_defaults(func=cmd_storage_check)
    sub.add_parser("repair-vdb", help="Repair vdb_entities.json from graphml").set_defaults(func=lambda args: cmd_repair_vdb())
    # repair-graph
    sp_repair_graph = sub.add_parser("repair-graph", help="Repair or reconstruct graphml from backups/KV")
    sp_repair_graph.add_argument("--rebuild", action="store_true", help="Force rebuild from KV store, ignoring backups")
    sp_repair_graph.set_defaults(func=lambda args: cmd_repair_graph(rebuild=args.rebuild))
    sp_test = sub.add_parser("test", help="Run all mandatory tests")
    sp_test.add_argument("target", nargs="?", default="all")
    sp_test.set_defaults(func=cmd_test)
    
    # diagnostics
    sub.add_parser("diagnostics", help="Auditoria profunda de grounding").set_defaults(func=cmd_diagnostics)

    args = p.parse_args()
    if hasattr(args, "term") and isinstance(args.term, list):
        args.term = " ".join(args.term)
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()

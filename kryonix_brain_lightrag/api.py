import os
import json
import logging
import uuid
import re
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from . import rag as rag_mod
from .utils import SecretScanner

# Configuração de log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kryonix-brain-api")

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Depends(api_key_header)):
    expected_key = os.getenv("KRYONIX_BRAIN_KEY")
    if expected_key and api_key != expected_key:
        raise HTTPException(status_code=403, detail="Acesso negado: API Key inválida")
    return api_key

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

# Metrics
SEARCH_LATENCY = Histogram("kryonix_brain_search_latency_seconds", "Latency of search requests")
SEARCH_COUNT = Counter("kryonix_brain_search_total", "Total search requests")
INGEST_COUNT = Counter("kryonix_brain_ingest_proposals_total", "Total ingestion proposals", ["source"])

app = FastAPI(title="Kryonix Brain API")

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

class SearchRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    lang: str = "pt-BR"
    no_cache: bool = False
    debug: bool = False
    explain: bool = False

class SearchResponse(BaseModel):
    status: str = "success"
    answer: str
    grounding: dict = {}
    sources: list = []
    warnings: list = []

class IngestProposeRequest(BaseModel):
    content: str = Field(..., description="Conteúdo a ser ingerido no grafo LightRAG")
    source: str = Field(..., description="Origem do conteúdo (ex.: 'manual', 'vault/note.md', 'web')")
    reason: str = Field("", description="Motivo da ingestão")

class NoteProposeRequest(BaseModel):
    title: str = Field(..., description="Título da nota (será slugificado)")
    content: str = Field(..., description="Conteúdo da nota em Markdown")
    source: str = Field(..., description="Origem da informação")
    reason: str = Field(..., description="Motivo da criação da nota")

class EventLogRequest(BaseModel):
    event: str = Field(..., description="Descrição do evento")
    metadata: dict = Field(default_factory=dict, description="Metadados adicionais")

class IngestApproveRequest(BaseModel):
    item_id: str = Field(..., description="ID do item na fila de ingestão")

@app.get("/health")
async def health():
    from .config import WORKING_DIR
    return {"status": "ok", "storage": str(WORKING_DIR)}

@app.get("/stats")
async def stats(api_key: str = Depends(get_api_key)):
    try:
        info = await rag_mod.stats()
        return info
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, api_key: str = Depends(get_api_key)):
    try:
        t0 = datetime.now().timestamp()
        SEARCH_COUNT.inc()
        
        request_id = str(uuid.uuid4())[:8]
        logger.info(f"[{request_id}] Searching: {req.query} (mode={req.mode}, lang={req.lang})")
        
        res = await rag_mod.query(
            req.query, 
            mode=req.mode, 
            lang=req.lang, 
            no_cache=req.no_cache,
            verbose=req.debug,
            explain=req.explain
        )
        
        latency = datetime.now().timestamp() - t0
        SEARCH_LATENCY.observe(latency)
        
        # Ensure grounding dict exists for tracing metadata
        if "grounding" not in res:
            res["grounding"] = {}
            
        res["grounding"]["trace_id"] = request_id
        res["grounding"]["latency_sec"] = round(latency, 3)
        res["grounding"]["confidence"] = res.get("confidence", "Unknown")
        res["grounding"]["max_score"] = res.get("max_score", 0.0)
        
        return SearchResponse(**res)
    except Exception as e:
        logger.error(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ── Ingestion Pipeline ───────────────────────────────────────────
# Fluxo: propose → queue → approve/reject
# Nenhum conteúdo entra no grafo sem aprovação explícita.

def _get_queue_dir() -> Path:
    from .config import INGEST_QUEUE_DIR
    INGEST_QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    return INGEST_QUEUE_DIR

def _get_approved_dir() -> Path:
    d = _get_queue_dir() / "approved"
    d.mkdir(parents=True, exist_ok=True)
    return d

@app.post("/ingest/propose")
async def ingest_propose(req: IngestProposeRequest, api_key: str = Depends(get_api_key)):
    """Propõe conteúdo para ingestão. Não indexa — fica na fila até approve."""
    
    # 1. Secret Scanning & Redaction
    sanitized_content, findings = SecretScanner.scan_and_redact(req.content)
    
    INGEST_COUNT.labels(source=req.source).inc()
    
    item_id = str(uuid.uuid4())[:8]
    item = {
        "id": item_id,
        "content": sanitized_content,
        "source": req.source,
        "reason": req.reason,
        "status": "pending",
        "proposed_at": datetime.now(timezone.utc).isoformat(),
        "security_findings": findings
    }
    path = _get_queue_dir() / f"{item_id}.json"
    path.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
    
    msg = f"Ingest proposed: {item_id} from {req.source}"
    if findings:
        msg += f" (Security warnings: {', '.join(findings)})"
    logger.info(msg)
    
    return {
        "status": "queued", 
        "id": item_id, 
        "security_alerts": findings if findings else None
    }

@app.post("/notes/propose")
async def notes_propose(req: NoteProposeRequest, api_key: str = Depends(get_api_key)):
    """Propõe uma nota para o Vault (00-inbox/ai-proposals)."""
    from .obsidian_cli import obsidian_propose_note
    try:
        # Sanitize note content too
        sanitized_content, findings = SecretScanner.scan_and_redact(req.content)
        
        result = obsidian_propose_note(
            title=req.title,
            content=sanitized_content,
            source=req.source,
            reason=req.reason
        )
        return {"status": "proposed", "message": result, "security_alerts": findings if findings else None}
    except Exception as e:
        logger.error(f"Error proposing note: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/events/log")
async def events_log(req: EventLogRequest, api_key: str = Depends(get_api_key)):
    """Registra um evento de interação no log do sistema."""
    from .config import BRAIN_HOME
    log_dir = BRAIN_HOME / "vault/90-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize metadata for safety
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": req.event,
        "metadata": req.metadata
    }
    
    log_file = log_dir / f"events-{datetime.now().strftime('%Y-%m')}.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\\n")
        
    return {"status": "logged"}

@app.get("/ingest/queue")
async def ingest_queue(api_key: str = Depends(get_api_key)):
    """Lista itens pendentes na fila de ingestão."""
    queue_dir = _get_queue_dir()
    items = []
    for f in sorted(queue_dir.glob("*.json")):
        if f.name == "approved":
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("status") == "pending":
                items.append(data)
        except Exception:
            pass
    return {"status": "ok", "count": len(items), "items": items}

@app.post("/ingest/approve/{item_id}")
async def ingest_approve(item_id: str, api_key: str = Depends(get_api_key)):
    """Aprova e indexa um item da fila no grafo LightRAG."""
    path = _get_queue_dir() / f"{item_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Item {item_id} não encontrado na fila")
    
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "pending":
        raise HTTPException(status_code=400, detail=f"Item {item_id} não está pendente (status={data.get('status')})")
    
    # Indexar no LightRAG
    try:
        await rag_mod.insert_single(data["content"])
        logger.info(f"Ingest approved and indexed: {item_id}")
    except Exception as e:
        logger.error(f"Error indexing {item_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao indexar: {e}")
    
    # Mover para approved/
    data["status"] = "approved"
    data["approved_at"] = datetime.now(timezone.utc).isoformat()
    approved_path = _get_approved_dir() / f"{item_id}.json"
    approved_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    path.unlink()
    
    return {"status": "approved", "id": item_id}

@app.delete("/ingest/reject/{item_id}")
async def ingest_reject(item_id: str, api_key: str = Depends(get_api_key)):
    """Rejeita e remove um item da fila de ingestão."""
    path = _get_queue_dir() / f"{item_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Item {item_id} não encontrado na fila")
    
    path.unlink()
    logger.info(f"Ingest rejected: {item_id}")
    return {"status": "rejected", "id": item_id}

def main():
    import uvicorn
    
    host = os.getenv("KRYONIX_BRAIN_HOST", "0.0.0.0")
    port = int(os.getenv("KRYONIX_BRAIN_PORT", "8000"))
    
    logger.info(f"Starting Kryonix Brain API on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()


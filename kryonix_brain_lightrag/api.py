import asyncio
import json
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from . import rag as rag_mod

# Configuração de log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kryonix-brain-api")

app = FastAPI(title="Kryonix Brain API")

class SearchRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    lang: str = "pt-BR"

class SearchResponse(BaseModel):
    answer: str
    status: str = "success"

@app.get("/health")
async def health():
    from .config import WORKING_DIR
    return {"status": "ok", "storage": str(WORKING_DIR)}

@app.get("/stats")
async def stats():
    try:
        info = await rag_mod.stats()
        return info
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    try:
        logger.info(f"Searching: {req.query} (mode={req.mode}, lang={req.lang})")
        answer = await rag_mod.query(req.query, mode=req.mode, lang=req.lang)
        return SearchResponse(answer=answer)
    except Exception as e:
        logger.error(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    import uvicorn
    import os
    
    # Permitir configurar porta via ENV
    host = os.getenv("KRYONIX_BRAIN_HOST", "0.0.0.0")
    port = int(os.getenv("KRYONIX_BRAIN_PORT", "8000"))
    
    logger.info(f"Starting Kryonix Brain API on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()

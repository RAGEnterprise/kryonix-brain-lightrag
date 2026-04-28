import os
import logging
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from . import rag as rag_mod

# ConfiguraÃ§Ã£o de log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kryonix-brain-api")

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Depends(api_key_header)):
    expected_key = os.getenv("KRYONIX_BRAIN_KEY")
    if expected_key and api_key != expected_key:
        raise HTTPException(status_code=403, detail="Acesso negado: API Key invÃ¡lida")
    return api_key

app = FastAPI(title="Kryonix Brain API")

class SearchRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    lang: str = "pt-BR"
    no_cache: bool = False
    debug: bool = False

class SearchResponse(BaseModel):
    status: str = "success"
    answer: str
    grounding: dict = {}
    sources: list = []
    warnings: list = []

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
        logger.info(f"Searching: {req.query} (mode={req.mode}, lang={req.lang})")
        res = await rag_mod.query(
            req.query, 
            mode=req.mode, 
            lang=req.lang, 
            no_cache=req.no_cache,
            verbose=req.debug
        )
        return SearchResponse(**res)
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

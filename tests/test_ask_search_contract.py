import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

# Mocking the RAG engine and dependencies
@pytest.mark.asyncio
async def test_search_intent_skips_llm():
    """Garante que search NÃO chama llm_func."""
    from kryonix_brain_lightrag.rag import query
    
    # Mock do llm_func que falha se for chamado
    fail_mock = AsyncMock(side_effect=AssertionError("search must not call llm_func"))
    
    # Mock do aquery_data para não precisar de um motor real
    mock_data = {
        "status": "success",
        "data": {
            "entities": [{"entity_name": "Kryonix", "entity_type": "Project", "description": "AI Platform", "score": 0.9}],
            "relationships": []
        }
    }
    
    # Mocks para evitar acesso ao disco
    with patch("kryonix_brain_lightrag.rag.llm_func", fail_mock), \
         patch("kryonix_brain_lightrag.rag.get_rag_async") as mock_get_rag, \
         patch("kryonix_brain_lightrag.rag._manual_grounding", AsyncMock(return_value=[{"file_path": "test.md", "content": "test", "score": 0.9, "chunk_id": "123"}])), \
         patch("kryonix_brain_lightrag.rag.analyze_query_strategy", AsyncMock(return_value={"strategy": "balanced", "mode": "hybrid", "hops": 1, "top_k": 5})), \
         patch("kryonix_brain_lightrag.rag.expand_query_semantically", AsyncMock(return_value="expanded query")):
        
        mock_rag = AsyncMock()
        mock_rag.aquery_data.return_value = mock_data
        mock_get_rag.return_value = mock_rag
        
        # Executa a query com intent=search
        res = await query("teste", intent="search")
        
        # Verificações
        assert res["status"] == "success"
        assert "sources" in res
        assert res.get("generation_skipped") is True
        assert res.get("provider_used") is None
        
        # O mock do llm_func NÃO deve ter sido chamado
        fail_mock.assert_not_called()

@pytest.mark.asyncio
async def test_ask_intent_calls_llm():
    """Garante que ask chama llm_func normalmente."""
    from kryonix_brain_lightrag.rag import query
    
    # Mock do llm_func que retorna uma resposta fixa
    success_mock = AsyncMock(return_value="Resposta sintetizada")
    
    mock_data = {
        "status": "success",
        "data": {
            "entities": [{"entity_name": "Kryonix", "entity_type": "Project", "description": "AI Platform", "score": 0.9}],
            "relationships": []
        }
    }
    
    # Mocks para evitar acesso ao disco
    with patch("kryonix_brain_lightrag.rag.llm_func", success_mock), \
         patch("kryonix_brain_lightrag.rag.get_rag_async") as mock_get_rag, \
         patch("kryonix_brain_lightrag.rag._manual_grounding", AsyncMock(return_value=[{"file_path": "test.md", "content": "test", "score": 0.9, "chunk_id": "123"}])), \
         patch("kryonix_brain_lightrag.rag.analyze_query_strategy", AsyncMock(return_value={"strategy": "balanced", "mode": "hybrid", "hops": 1, "top_k": 5})), \
         patch("kryonix_brain_lightrag.rag.expand_query_semantically", AsyncMock(return_value="expanded query")):
        
        mock_rag = AsyncMock()
        mock_rag.aquery_data.return_value = mock_data
        mock_get_rag.return_value = mock_rag
        
        # Executa a query com intent=ask (padrão)
        res = await query("teste", intent="ask")
        
        # Verificações
        assert res["status"] == "success"
        assert res["answer"] == "Resposta sintetizada"
        assert res.get("generation_skipped") is False
        
        # O mock do llm_func DEVE ter sido chamado
        success_mock.assert_called_once()

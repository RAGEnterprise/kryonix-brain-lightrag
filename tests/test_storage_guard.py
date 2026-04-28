def test_storage_guard():
    from kryonix_brain_lightrag import config
    assert "tools\\lightrag\\rag_storage" not in str(config.WORKING_DIR)
    assert "11-LightRAG\\rag_storage" in str(config.WORKING_DIR)

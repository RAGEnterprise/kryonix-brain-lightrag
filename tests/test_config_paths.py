def test_config_paths():
    from kryonix_brain_lightrag import config
    assert "kryonix-vault" in str(config.VAULT_DIR)
    assert "rag_storage" in str(config.WORKING_DIR)
    assert config.WORKING_DIR.parent.name == "11-LightRAG"
    assert config.LLM_PROVIDER == "ollama"

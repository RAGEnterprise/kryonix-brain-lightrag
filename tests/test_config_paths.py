def test_config_paths():
    from kryonix_brain_lightrag import config
    assert "kryonix-vault" in str(config.VAULT_DIR)
    # No novo padrão, WORKING_DIR é .../storage dentro do home do brain
    assert "storage" in str(config.WORKING_DIR)
    assert config.WORKING_DIR.parent.name == "kryonix-vault"
    assert config.LLM_PROVIDER == "ollama"

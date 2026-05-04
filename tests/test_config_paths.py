def test_config_paths():
    from kryonix_brain_lightrag import config
    assert str(config.VAULT_DIR).endswith("/vault")
    # No novo padrão, WORKING_DIR é .../storage dentro do home do brain
    assert str(config.WORKING_DIR).endswith("/storage")
    assert config.WORKING_DIR.parent == config.BRAIN_HOME
    assert config.VAULT_DIR.parent == config.BRAIN_HOME
    assert config.LLM_PROVIDER == "ollama"

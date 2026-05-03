def test_storage_guard():
    from kryonix_brain_lightrag import config
    # Garante que não estamos usando o path local de dev dentro do pacote
    assert "packages" not in str(config.WORKING_DIR)
    # Garante que estamos no path correto do sistema
    assert "kryonix-vault/storage" in str(config.WORKING_DIR).replace("\\", "/")

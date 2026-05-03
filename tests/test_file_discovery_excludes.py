def test_file_discovery_excludes():
    from kryonix_brain_lightrag.config import should_exclude_path as _should_exclude
    assert _should_exclude(".env") == True
    assert _should_exclude(".env.local") == True
    assert _should_exclude("node_modules/test.js") == True
    assert _should_exclude("image.png") == True
    assert _should_exclude("app.exe") == True
    assert _should_exclude("src/main.py") == False

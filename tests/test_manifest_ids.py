def test_manifest_ids():
    from kryonix_brain_lightrag.rag import doc_id
    import hashlib
    rel_path = "test/file.py"
    digest = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:12]
    expected = f"doc-{digest}"
    assert doc_id(rel_path) == expected

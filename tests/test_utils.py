import pytest
from kryonix_brain_lightrag.rag import doc_id, slugify
from kryonix_brain_lightrag.utils import SecretScanner

def test_doc_id():
    # Correct SHA1 prefixes for reference
    assert doc_id("vault/note.md") == "doc-972abdd2511f"
    assert doc_id("repo/main.py") == "doc-111461a91b8d"
    assert len(doc_id("any")) == 16

def test_slugify():
    assert slugify("Minha Nota") == "minha-nota"
    assert slugify("Path/With\\Backslash") == "path-with-backslash"
    assert slugify("Special @!# Characters") == "special-characters"
    assert slugify("   Space Trim   ") == "space-trim"
    assert slugify("") == "unknown"

def test_secret_scanner():
    text = "Aqui está minha key: sk-1234567890abcdef"
    redacted, findings = SecretScanner.scan_and_redact(text)
    assert "[REDACTED]" in redacted
    assert "sk-1234567890abcdef" not in redacted
    assert "key" in findings

def test_secret_scanner_clean():
    text = "Texto limpo sem segredos."
    redacted, findings = SecretScanner.scan_and_redact(text)
    assert redacted == text
    assert len(findings) == 0

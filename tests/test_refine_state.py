import json
import pytest
from pathlib import Path
from kryonix_brain_lightrag.index import _load_json, _save_json

def test_save_load_json(tmp_path):
    test_file = tmp_path / "test.json"
    data = {"key": "value"}
    _save_json(test_file, data)
    
    loaded = _load_json(test_file)
    assert loaded == data

def test_load_non_existent():
    loaded = _load_json(Path("non_existent.json"))
    assert loaded == {}

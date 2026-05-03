import pytest
from pathlib import Path
import networkx as nx
from kryonix_brain_lightrag.graph_utils import validate_graphml, atomic_write_graphml

def test_validate_graphml_exists(tmp_path):
    p = tmp_path / "test.graphml"
    valid, err = validate_graphml(p)
    assert not valid
    assert "não existe" in err

def test_validate_graphml_empty(tmp_path):
    p = tmp_path / "test.graphml"
    p.write_text("")
    valid, err = validate_graphml(p)
    assert not valid
    assert "vazio" in err

def test_validate_graphml_invalid_xml(tmp_path):
    p = tmp_path / "test.graphml"
    p.write_text("<graphml><invalid")
    valid, err = validate_graphml(p)
    assert not valid
    assert "XML inválido" in err

def test_validate_graphml_no_nodes(tmp_path):
    p = tmp_path / "test.graphml"
    G = nx.Graph()
    nx.write_graphml(G, p)
    valid, err = validate_graphml(p)
    assert not valid
    assert "não possui nós" in err

def test_validate_graphml_ok(tmp_path):
    p = tmp_path / "test.graphml"
    G = nx.Graph()
    G.add_node("A")
    nx.write_graphml(G, p)
    valid, err = validate_graphml(p)
    assert valid
    assert err == "OK"

def test_atomic_write_graphml(tmp_path):
    p = tmp_path / "final.graphml"
    G = nx.Graph()
    G.add_node("A")
    
    atomic_write_graphml(G, p)
    assert p.exists()
    
    G2 = nx.read_graphml(p)
    assert "A" in G2.nodes
    
    # Update
    G.add_node("B")
    atomic_write_graphml(G, p)
    
    G3 = nx.read_graphml(p)
    assert "B" in G3.nodes
    
    # Verify backup exists
    backups = list(tmp_path.glob("*.bak-*.graphml"))
    assert len(backups) >= 1

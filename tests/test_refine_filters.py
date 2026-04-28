import pytest
from kryonix_brain_lightrag.index import _is_useful_content

def test_is_useful_content_too_short():
    useful, reason = _is_useful_content("abc", min_chars=10)
    assert not useful
    assert reason == "too_short"

def test_is_useful_content_no_letters():
    useful, reason = _is_useful_content("123 !@#", min_chars=5)
    assert not useful
    assert reason == "no_letters"

def test_is_useful_content_boilerplate_json():
    useful, reason = _is_useful_content('{{a}}{{b}}{{c}}{{d}}{{e}}', min_chars=5)
    assert not useful
    assert reason == "boilerplate_symbols"

def test_is_useful_content_ok():
    useful, reason = _is_useful_content("This is a technical documentation about NixOS and Kryonix.", min_chars=10)
    assert useful
    assert reason == "ok"

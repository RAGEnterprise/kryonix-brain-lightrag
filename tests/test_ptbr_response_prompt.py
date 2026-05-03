import pytest
from kryonix_brain_lightrag import config

def test_ptbr_prompt_content():
    prompt = config.ANSWER_SYSTEM_PROMPT
    assert "português do Brasil" in prompt
    assert "Kryonix" in prompt
    assert "Preserve nomes técnicos" in prompt
    assert "Não invente" in prompt

def test_language_setting():
    assert config.RESPONSE_LANGUAGE == "pt-BR"

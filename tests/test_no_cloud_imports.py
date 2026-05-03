import os
from pathlib import Path

def test_no_cloud_imports():
    from kryonix_brain_lightrag import config
    cloud_words = ["gemini", "openai", "anthropic", "voyage", "GOOGLE_API_KEY", "OPENAI_API_KEY"]
    src_dir = config.PROJECT_DIR / "tools" / "lightrag" / "kryonix_brain_lightrag"
    
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".py"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    content = f.read().lower()
                    for cw in cloud_words:
                        if cw.lower() in content and "never calls" not in content and "cloud_words" not in content:
                            assert False, f"Cloud word {cw} found in {file}"

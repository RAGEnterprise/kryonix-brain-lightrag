import json
import os

path = r"C:\Users\aguia\Documents\kryonix-vault\11-LightRAG\rag_storage\vdb_relationships.json"
if os.path.exists(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("data", [])
    print(f"Total relations in VDB: {len(items)}")
    print("First 5 relations:")
    for item in items[:5]:
        src = item.get("src_id")
        tgt = item.get("tgt_id")
        print(f"  - {src} -> {tgt}")
        print(f"    Content sample: {item.get('content', '')[:100]}")
else:
    print("File not found")

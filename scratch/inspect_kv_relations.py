import json
import os

path = r"C:\Users\aguia\Documents\kryonix-vault\11-LightRAG\rag_storage\kv_store_full_relations.json"
if os.path.exists(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Total entries in KV full_relations: {len(data)}")
    first_key = list(data.keys())[0]
    print(f"First key: {first_key}")
    print(f"Value sample: {json.dumps(data[first_key], indent=2)[:500]}")
else:
    print("File not found")

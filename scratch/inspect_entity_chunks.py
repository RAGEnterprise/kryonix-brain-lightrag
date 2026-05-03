import json
import os

path = r"C:\Users\aguia\Documents\kryonix-vault\11-LightRAG\rag_storage\kv_store_entity_chunks.json"
if os.path.exists(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Total entries in entity_chunks: {len(data)}")
    first_key = list(data.keys())[0]
    print(f"First key (Entity Name): {first_key}")
    print(f"Value (Chunk IDs): {data[first_key]}")
else:
    print("File not found")

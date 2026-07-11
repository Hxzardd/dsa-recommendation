import json
import os
from qdrant_client import QdrantClient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_storage_v2")
client = QdrantClient(path=QDRANT_PATH)

COLLECTION = "problems_v2"
OUT = os.path.join(BASE_DIR, "real_difficulty_map.json")

def main():
    difficulty_map = {}
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION, limit=500, offset=next_offset,
            with_payload=True, with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            slug = payload.get("title_slug")
            diff = payload.get("difficulty_score")
            if slug is not None and diff is not None:
                difficulty_map[slug] = float(diff)
        if next_offset is None:
            break
    with open(OUT, "w") as f:
        json.dump(difficulty_map, f)
    print(f"Exported {len(difficulty_map)} problems. Written to {OUT}")

if __name__ == "__main__":
    main()

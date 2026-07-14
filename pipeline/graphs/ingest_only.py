"""
ingest_only.py

Run this if run_rgcn_pipeline.py completed (final_cluster showed) but
Qdrant collections still don't exist -- means the ingest step silently
failed inside the pipeline. This runs it directly with full error output.

Run from pipeline/graphs/:
    python ingest_only.py
"""
import sys
from pathlib import Path

# Make sure db_env.py (at repo root) is importable
_here = Path(__file__).resolve()
for _p in [_here.parent, *_here.parents]:
    if (_p / "pyproject.toml").exists():
        sys.path.insert(0, str(_p))
        sys.path.insert(0, str(_p / "pipeline" / "graphs"))
        break

import db_env
import config as C
from qdrant_client import QdrantClient
import ingest_rgcn_to_qdrant as ing

print(f"Connecting to Qdrant at {db_env.QDRANT_URL}...")
print(f"API key set: {bool(db_env.QDRANT_API_KEY)}")

client = QdrantClient(url=C.QDRANT_URL, api_key=C.QDRANT_API_KEY, timeout=120)
client.get_collections()   # will raise immediately if auth fails -- no silent swallow
print("Connected OK")

print("Loading artifacts...")
ids, payloads, vecs = ing.from_artifacts_and_parquet()
print(f"Loaded {len(ids)} problems")

print(f"Ingesting -> {C.QDRANT_COLLECTION_RGCN}...")
ing.ingest_embeddings(client, C.QDRANT_COLLECTION_RGCN, ids, vecs["rgcn"], payloads)
print(f"Ingesting -> {C.QDRANT_COLLECTION_FULL}...")
ing.ingest_embeddings(client, C.QDRANT_COLLECTION_FULL, ids, vecs["full"], payloads)

print("\n[OK] Both collections written to Qdrant.")
print("Now run from repo root:")
print("  python create_qdrant_indexes.py")
print("  python check_qdrant.py")

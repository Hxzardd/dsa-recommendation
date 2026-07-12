"""
check_titles.py - run from repo root
Checks whether problem payloads actually have title/title_slug fields
"""
import sys
from pathlib import Path
for p in Path(__file__).resolve().parents:
    if (p / "pyproject.toml").exists():
        sys.path.insert(0, str(p))
        break

import db_env

def _stable_point_id(problem_id: str) -> int:
    try:
        import xxhash
        return xxhash.xxh64(problem_id).intdigest() & 0x7FFF_FFFF_FFFF_FFFF
    except ImportError:
        import hashlib
        return int(hashlib.sha256(problem_id.encode()).hexdigest(), 16) & 0x7FFF_FFFF_FFFF_FFFF

client = db_env.qdrant_client(timeout=10)

# Check what fields are actually stored in a payload
points, _ = client.scroll(
    collection_name="problems_full",
    limit=3,
    with_payload=True,
    with_vectors=False,
)
print("=== Full payload of first 3 points ===")
for p in points:
    print(p.payload)
    print()

# Check if retrieve() by stable hash ID works
print("=== retrieve() by stable hash ID ===")
pid = points[0].id  # this is the raw string problem_id
stable_id = _stable_point_id(str(pid))
print(f"problem_id string: {pid}")
print(f"stable hash int:   {stable_id}")
retrieved = client.retrieve(
    collection_name="problems_full",
    ids=[stable_id],
    with_payload=True,
    with_vectors=False,
)
print(f"retrieve() found: {len(retrieved)} points")
if retrieved:
    print(f"payload: {retrieved[0].payload}")
else:
    print("NOTHING FOUND -- the stable hash ID doesn't match how points were stored")
    print("Trying retrieve() with the raw string ID instead...")
    try:
        retrieved2 = client.retrieve(
            collection_name="problems_full",
            ids=[str(pid)],
            with_payload=True,
            with_vectors=False,
        )
        print(f"retrieve() with string ID found: {len(retrieved2)} points")
    except Exception as e:
        print(f"string ID also failed: {e}")

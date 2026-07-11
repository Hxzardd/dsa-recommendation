"""
Run this from repo root to check what's actually in your Qdrant collection
and why cold-start recommendations return empty.

    python check_qdrant.py
"""
import sys
from pathlib import Path

for p in Path(__file__).resolve().parents:
    if (p / "pyproject.toml").exists():
        sys.path.insert(0, str(p))
        break

import db_env

def main():
    client = db_env.qdrant_client(timeout=10)

    print("=== Collections ===")
    collections = client.get_collections().collections
    for c in collections:
        count = client.count(collection_name=c.name).count
        print(f"  {c.name}: {count} points")

    print()
    target = "problems_full"
    if not any(c.name == target for c in collections):
        print(f"[X] '{target}' collection does not exist.")
        print("    Run the offline pipeline first:")
        print("    python run_full_pipeline.py my_user_id --force-offline")
        return

    print(f"=== Sample from '{target}' ===")
    points, _ = client.scroll(
        collection_name=target,
        limit=3,
        with_payload=True,
        with_vectors=False,
    )
    for p in points:
        payload = p.payload or {}
        print(f"  problem_id : {payload.get('problem_id')}")
        print(f"  title      : {payload.get('title')}")
        print(f"  topic_tags : {payload.get('topic_tags')}")
        print()

    print("=== Starter concept check ===")
    STARTER = ["arrays", "strings", "hash_map", "sorting"]
    for topic in STARTER:
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        points, _ = client.scroll(
            collection_name=target,
            scroll_filter=Filter(must=[
                FieldCondition(key="topic_tags", match=MatchAny(any=[topic]))
            ]),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        count = client.count(
            collection_name=target,
            count_filter=Filter(must=[
                FieldCondition(key="topic_tags", match=MatchAny(any=[topic]))
            ])
        ).count
        print(f"  '{topic}': {count} problems tagged")

if __name__ == "__main__":
    main()

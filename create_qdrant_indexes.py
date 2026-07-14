"""
create_qdrant_indexes.py

Creates the required payload indexes on your Qdrant Cloud collections.
Run this ONCE -- Qdrant Cloud requires explicit keyword indexes before
filtered scrolls work (unlike local Docker Qdrant which is more permissive).

Without these indexes, every pool's topic_tags filter returns:
    "Index required but not found for topic_tags"
...and all 7 pools return zero candidates, giving empty recommendations.

Run:
    python create_qdrant_indexes.py
"""

import sys
from pathlib import Path

for p in Path(__file__).resolve().parents:
    if (p / "pyproject.toml").exists():
        sys.path.insert(0, str(p))
        break

import db_env
from qdrant_client.models import PayloadSchemaType

COLLECTIONS = ["problems_full", "problems_rgcn", "problems_question", "problems_solution"]

INDEXES = [
    ("topic_tags",       PayloadSchemaType.KEYWORD),
    ("difficulty_score", PayloadSchemaType.FLOAT),
    ("problem_id",       PayloadSchemaType.KEYWORD),
    ("title_slug",       PayloadSchemaType.KEYWORD),
]


def main():
    client = db_env.qdrant_client(timeout=30)

    for collection in COLLECTIONS:
        try:
            client.get_collection(collection)
        except Exception:
            print(f"[->] Skipping '{collection}' -- does not exist")
            continue

        print(f"\n[->] Creating indexes on '{collection}'...")
        for field, schema_type in INDEXES:
            try:
                client.create_payload_index(
                    collection_name=collection,
                    field_name=field,
                    field_schema=schema_type,
                )
                print(f"     OK  {field} ({schema_type})")
            except Exception as exc:
                if "already exists" in str(exc).lower():
                    print(f"     --  {field} already indexed")
                else:
                    print(f"     [!] {field} failed: {exc}")

    print("\n[OK] Done. Run check_qdrant.py again to confirm, then retry /recommend.")


if __name__ == "__main__":
    main()
"""
reset_qdrant.py

Deletes all three Qdrant collections (dsa_problems, problems_full,
problems_rgcn) from your .env-configured cloud cluster, then confirms
they're gone so the offline pipeline can re-ingest cleanly.

Run BEFORE re-running the offline pipeline:
    python reset_qdrant.py

Then re-run the full pipeline:
    python run_full_pipeline.py <user_id> --force-offline
"""

import sys
from pathlib import Path

for _p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
    if (_p / "pyproject.toml").exists():
        sys.path.insert(0, str(_p))
        break

import db_env

COLLECTIONS = ["dsa_problems", "problems_full", "problems_rgcn"]


def main():
    print(f"\nConnecting to Qdrant at {db_env.QDRANT_URL}...")
    client = db_env.qdrant_client(timeout=15)

    # Verify connection
    existing = {c.name for c in client.get_collections().collections}
    print(f"Existing collections: {sorted(existing) or '(none)'}")

    to_delete = [c for c in COLLECTIONS if c in existing]
    if not to_delete:
        print("\n[OK] Nothing to delete -- collections already empty.")
        print("     Ready to re-run the offline pipeline:")
        print("     python run_full_pipeline.py <user_id> --force-offline")
        return

    print(f"\nDeleting: {to_delete}")
    confirm = input("Type 'yes' to confirm permanent deletion: ").strip().lower()
    if confirm != "yes":
        print("Aborted.")
        return

    for name in to_delete:
        count = client.count(collection_name=name).count
        client.delete_collection(name)
        print(f"  [OK] Deleted '{name}' ({count} points)")

    # Verify gone
    remaining = {c.name for c in client.get_collections().collections}
    still_there = [c for c in COLLECTIONS if c in remaining]
    if still_there:
        print(f"\n[!] Still present after deletion: {still_there}")
        sys.exit(1)

    print(f"\n[OK] All collections deleted. Qdrant is clean.")
    print(f"\nNext steps:")
    print(f"  1. Regenerate graph artifacts (if not done):")
    print(f"       python regenerate_graph_artifacts.py")
    print(f"  2. Re-run the full offline pipeline:")
    print(f"       python run_full_pipeline.py <user_id> --force-offline")
    print(f"  3. Recreate Qdrant indexes after upload:")
    print(f"       python create_qdrant_indexes.py")


if __name__ == "__main__":
    main()

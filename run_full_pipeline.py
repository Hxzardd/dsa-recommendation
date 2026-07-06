"""
run_full_pipeline.py

ONE file that runs the whole thing, offline through online, as intended:

    python run_full_pipeline.py <user_id>

What it does, in order:
    1. Checks whether the offline pipeline has already been run (does
       'dsa_problems' and 'problems_full' exist in Qdrant with points in
       them?). If yes, skips straight to step 2 -- ingestion/embedding/RGCN
       training NEVER re-run just because this script is called again.
       If no, runs the three offline stages once, in order.
    2. Calls get_recommendations() -- the single online callable -- and
       prints the ranked top-10 result as JSON.

Flags:
    --force-offline   Re-run the offline pipeline even if Qdrant already
                       looks populated (use after adding new problems).
    --skip-offline     Never run the offline pipeline, even if empty (useful
                       if you know it's running separately / on a schedule).
    --k N              How many recommendations to return (default 10).
    --total-n N        How many raw candidates to draw before filtering
                       (default 30).

Examples:
    python run_full_pipeline.py my_user_id
    python run_full_pipeline.py my_user_id --force-offline
    python run_full_pipeline.py my_user_id --k 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Config -- adjust these two if your paths/URLs differ
# ---------------------------------------------------------------------------

QDRANT_URL = "http://localhost:6333"
MANIFEST_PATH = "data/1000_manifest_final.json"
REQUIRED_COLLECTIONS = ["dsa_problems", "problems_full"]

OFFLINE_STEPS = [
    ("Ingestion", [
        sys.executable, "pipeline/ingestion/ingest.py",
        "--input", MANIFEST_PATH,
    ]),
    ("Embeddings", [
        sys.executable, "pipeline/embeddings/embedder.py",
        "--resume", "--collection", "dsa_problems",
        "--qdrant-url", QDRANT_URL,
    ]),
    ("Graph build + RGCN train + Qdrant ingest", [
        sys.executable, "pipeline/graphs/run_rgcn_pipeline.py",
        "--graph-source", "normalized",
    ]),
]


# ---------------------------------------------------------------------------
# Step 1: offline pipeline, run-once-only
# ---------------------------------------------------------------------------

def offline_pipeline_already_done(qdrant_url: str) -> bool:
    """
    True if every required collection exists AND has at least one point.
    This is the check that keeps ingestion/embedding/RGCN training from
    re-running on every call -- only runs the offline steps when Qdrant
    genuinely looks empty/unset-up.
    """
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        print("[!] qdrant-client not installed -- cannot check offline state, "
              "assuming offline pipeline has NOT been run.")
        return False

    try:
        client = QdrantClient(url=qdrant_url, timeout=5)
        for name in REQUIRED_COLLECTIONS:
            info = client.get_collection(name)
            count = client.count(collection_name=name).count
            if count == 0:
                print(f"[->] Collection '{name}' exists but is empty.")
                return False
        return True
    except Exception as exc:
        print(f"[->] Offline pipeline check failed ({exc.__class__.__name__}) "
              f"-- assuming it hasn't been run yet.")
        return False


def run_offline_pipeline():
    """Run ingestion -> embeddings -> RGCN pipeline, once, in order. Stops on first failure."""
    print("\n" + "=" * 64)
    print("  OFFLINE PIPELINE -- running (this only happens once)")
    print("=" * 64)

    for i, (label, cmd) in enumerate(OFFLINE_STEPS, 1):
        print(f"\n[{i}/{len(OFFLINE_STEPS)}] {label}")
        print(f"    $ {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\n[X] FAILED at step {i}/{len(OFFLINE_STEPS)}: {label}")
            print(f"    Exit code: {result.returncode}")
            sys.exit(result.returncode)

    print("\n[OK] Offline pipeline complete.\n")


def ensure_offline_pipeline(force: bool, skip: bool):
    if skip:
        print("[->] --skip-offline set; assuming Qdrant is already populated.")
        return
    if force:
        run_offline_pipeline()
        return
    if offline_pipeline_already_done(QDRANT_URL):
        print("[OK] Offline pipeline already done -- Qdrant is populated. "
              "Skipping ingestion/embedding/training (use --force-offline "
              "to re-run anyway).")
        return
    run_offline_pipeline()


# ---------------------------------------------------------------------------
# Step 2: the online callable
# ---------------------------------------------------------------------------

def run_online_recommendation(user_id: str, k: int, total_n: int) -> dict:
    from qdrant_client import QdrantClient
    from pipeline.recommender.services.recommend import get_recommendations

    qdrant = QdrantClient(url=QDRANT_URL, timeout=10)

    # Shraddha's in-memory BKT/HLR stores -- in a real backend these are
    # imported as the same module-level dicts her submission_controller.py
    # already owns, e.g.:
    #   from controllers.submission_controller import user_mastery_store, user_hlr_store
    bkt_store: dict = {}
    hlr_store: dict = {}

    # db=None demonstrates the brand-new-user path (no Postgres session).
    # Pass a real session here once wired to your backend's db layer:
    #   db = get_db_session()
    db = None

    result = get_recommendations(
        user_id=user_id,
        db=db,
        redis=None,
        qdrant=qdrant,
        bkt_store=bkt_store,
        hlr_store=hlr_store,
        collection="problems_full",
        total_n=total_n,
        k=k,
    )
    return result.to_dict()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("user_id", nargs="?", default="test_user_001")
    parser.add_argument("--force-offline", action="store_true")
    parser.add_argument("--skip-offline", action="store_true")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--total-n", type=int, default=30)
    args = parser.parse_args()

    if args.force_offline and args.skip_offline:
        print("[X] --force-offline and --skip-offline are mutually exclusive.")
        sys.exit(1)

    ensure_offline_pipeline(force=args.force_offline, skip=args.skip_offline)

    print("\n" + "=" * 64)
    print(f"  ONLINE RECOMMENDATION -- user_id={args.user_id}")
    print("=" * 64)

    result = run_online_recommendation(args.user_id, k=args.k, total_n=args.total_n)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""
run_full_pipeline.py

ONE file that runs the whole thing, offline through online, taking a JSON
REQUEST and returning a JSON RESPONSE -- the same shape a backend
controller would send/receive if it called this as a subprocess or you
ported this logic into an HTTP handler later.

REQUEST (stdin, or --input-json <file>, or CLI flags for quick manual runs):
    {
      "user_id":      "string, required",
      "k":             10,              // optional, final slate size
      "total_n":       30,              // optional, raw candidates drawn
      "session_mode":  "practice"       // optional, "practice" | "learning"
    }

RESPONSE (stdout -- ALWAYS pure JSON, nothing else on stdout):
    {
      "user_id": "string",
      "session_mode": "practice" | "learning" | null,
      "recommendations": [
        {
          "problem_id":       "string",
          "title":            "string",
          "title_slug":       "string",
          "difficulty_score": 0.42,
          "topic_tags":       ["arrays", "hash_map"],
          "source":           "course_path",   // RecommendationLog.source enum
          "recommended_at":   "2026-07-07T12:00:00+00:00"
        },
        ...
      ]
    }

All progress/status output (offline pipeline steps, Neo4j connection
status, etc.) goes to STDERR, never stdout -- so stdout is always clean,
parseable JSON regardless of what happens during the run. This is what
lets another process pipe stdin -> this script -> stdout and get back
exactly one JSON object, no text to strip out first.

Usage:
    # JSON request via stdin
    echo '{"user_id": "u1", "k": 5}' | python run_full_pipeline.py

    # JSON request from a file
    python run_full_pipeline.py --input-json request.json

    # Quick manual run (builds the request dict from CLI flags)
    python run_full_pipeline.py u1 --k 5

Flags:
    --input-json FILE  Read the JSON request from this file instead of stdin/CLI.
    --force-offline     Re-run the offline pipeline even if Qdrant already
                        looks populated (use after adding new problems).
    --skip-offline      Never run the offline pipeline, even if empty.
    --no-neo4j          Skip Neo4j durable storage even if reachable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Config -- adjust these if your paths/URLs/credentials differ
# ---------------------------------------------------------------------------

QDRANT_URL = "http://localhost:6333"
MANIFEST_PATH = "data/1000_manifest_final.json"
REQUIRED_COLLECTIONS = ["dsa_problems", "problems_full"]

NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "your_password"   # change this to your actual Neo4j password

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


def _log(msg: str):
    """All status/progress output goes to stderr -- stdout is reserved for the final JSON response."""
    print(msg, file=sys.stderr)


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
        _log("[!] qdrant-client not installed -- cannot check offline state, "
             "assuming offline pipeline has NOT been run.")
        return False

    try:
        client = QdrantClient(url=qdrant_url, timeout=5)
        for name in REQUIRED_COLLECTIONS:
            info = client.get_collection(name)
            count = client.count(collection_name=name).count
            if count == 0:
                _log(f"[->] Collection '{name}' exists but is empty.")
                return False
        return True
    except Exception as exc:
        _log(f"[->] Offline pipeline check failed ({exc.__class__.__name__}) "
             f"-- assuming it hasn't been run yet.")
        return False


def run_offline_pipeline():
    """Run ingestion -> embeddings -> RGCN pipeline, once, in order. Stops on first failure."""
    _log("\n" + "=" * 64)
    _log("  OFFLINE PIPELINE -- running (this only happens once)")
    _log("=" * 64)

    for i, (label, cmd) in enumerate(OFFLINE_STEPS, 1):
        _log(f"\n[{i}/{len(OFFLINE_STEPS)}] {label}")
        _log(f"    $ {' '.join(cmd)}")
        result = subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr)
        if result.returncode != 0:
            _log(f"\n[X] FAILED at step {i}/{len(OFFLINE_STEPS)}: {label}")
            _log(f"    Exit code: {result.returncode}")
            sys.exit(result.returncode)

    _log("\n[OK] Offline pipeline complete.\n")


def ensure_offline_pipeline(force: bool, skip: bool):
    if skip:
        _log("[->] --skip-offline set; assuming Qdrant is already populated.")
        return
    if force:
        run_offline_pipeline()
        return
    if offline_pipeline_already_done(QDRANT_URL):
        _log("[OK] Offline pipeline already done -- Qdrant is populated. "
             "Skipping ingestion/embedding/training (use --force-offline "
             "to re-run anyway).")
        return
    run_offline_pipeline()


# ---------------------------------------------------------------------------
# Step 2: Neo4j -- automatic connection, graceful fallback if unavailable
# ---------------------------------------------------------------------------

def build_neo4j_store(disabled: bool):
    """
    Connects to Neo4j automatically so the user graph persists durably
    (past the Redis TTL, past this process ending) with zero manual setup
    per run. Falls back to a disabled store (Neo4jGraphStore(driver=None))
    -- NOT a crash -- if:
        - --no-neo4j was passed
        - the neo4j package isn't installed
        - Neo4j isn't reachable at NEO4J_URI

    A disabled store makes every UserGraphService call behave exactly as
    it did before Neo4j existed (Redis + Postgres only) -- this script
    never hard-requires Neo4j to run.
    """
    from pipeline.recommender.services.neo4j_graph_store import Neo4jGraphStore

    if disabled:
        _log("[->] --no-neo4j set; skipping durable graph storage.")
        return Neo4jGraphStore(driver=None)

    try:
        from neo4j import GraphDatabase
    except ImportError:
        _log("[!] neo4j package not installed -- durable graph storage "
             "disabled for this run. Install with: uv pip install neo4j")
        return Neo4jGraphStore(driver=None)

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        _log(f"[OK] Neo4j connected at {NEO4J_URI} -- user graph will "
             f"persist durably across runs.")
        return Neo4jGraphStore(driver)
    except Exception as exc:
        _log(f"[!] Neo4j unreachable at {NEO4J_URI} ({exc.__class__.__name__}) "
             f"-- durable graph storage disabled for this run. "
             f"Falling back to Redis+Postgres only.")
        return Neo4jGraphStore(driver=None)


# ---------------------------------------------------------------------------
# Step 3: title resolution (belt-and-suspenders re-resolve for anything
# get_recommendations() itself couldn't resolve)
# ---------------------------------------------------------------------------

def _stable_point_id(problem_id: str) -> int:
    """Must exactly match pipeline/graphs/ingest_rgcn_to_qdrant.py's function of the same name."""
    try:
        import xxhash
        return xxhash.xxh64(problem_id).intdigest() & 0x7FFF_FFFF_FFFF_FFFF
    except ImportError:
        import hashlib
        return int(hashlib.sha256(problem_id.encode()).hexdigest(), 16) & 0x7FFF_FFFF_FFFF_FFFF


def resolve_problem_ids(problem_ids: list, qdrant, collection: str = "problems_full") -> dict:
    if not problem_ids:
        return {}
    point_ids = [_stable_point_id(pid) for pid in problem_ids]
    id_to_pid = dict(zip(point_ids, problem_ids))
    try:
        points = qdrant.retrieve(collection_name=collection, ids=point_ids,
                                 with_payload=True, with_vectors=False)
    except Exception as exc:
        _log(f"[!] Could not resolve problem titles: {exc}")
        return {}
    resolved = {}
    for p in points:
        original_pid = id_to_pid.get(p.id)
        if original_pid is not None:
            resolved[original_pid] = p.payload or {}
    return resolved


def resolve_recommendations(result_dict: dict, qdrant, collection: str = "problems_full") -> dict:
    """Fills in title/title_slug wherever get_recommendations() itself left them empty."""
    recs = result_dict.get("recommendations", [])
    problem_ids = [r["problem_id"] for r in recs if "problem_id" in r and not r.get("title")]
    payloads = resolve_problem_ids(problem_ids, qdrant, collection=collection)

    enriched_recs = []
    for r in recs:
        pid = r.get("problem_id")
        enriched = dict(r)
        if not enriched.get("title"):
            payload = payloads.get(pid, {})
            enriched["title"] = payload.get("title", "(not found in Qdrant)")
            enriched["title_slug"] = payload.get("title_slug")
        enriched_recs.append(enriched)

    out = dict(result_dict)
    out["recommendations"] = enriched_recs
    return out


# ---------------------------------------------------------------------------
# Core: request dict -> response dict
# ---------------------------------------------------------------------------

def handle_request(request: dict) -> dict:
    """
    The actual JSON-in / JSON-out contract. Everything above this function
    is plumbing (offline pipeline, Neo4j connection); everything below is
    what a backend calling this as an HTTP handler or subprocess would
    actually invoke.

    request:  {"user_id": str, "k": int?, "total_n": int?, "session_mode": str?}
    returns:  {"user_id": str, "session_mode": str|None, "recommendations": [...]}
    """
    user_id = request.get("user_id")
    if not user_id:
        raise ValueError("request must include \"user_id\"")

    k = request.get("k", 10)
    total_n = request.get("total_n", 30)
    session_mode = request.get("session_mode")
    force_offline = request.get("force_offline", False)
    skip_offline = request.get("skip_offline", False)
    no_neo4j = request.get("no_neo4j", False)

    ensure_offline_pipeline(force=force_offline, skip=skip_offline)
    neo4j_store = build_neo4j_store(disabled=no_neo4j)

    from qdrant_client import QdrantClient
    from pipeline.recommender.services.recommend import get_recommendations

    qdrant = QdrantClient(url=QDRANT_URL, timeout=10)

    # Shraddha's in-memory BKT/HLR stores -- in a real backend these are
    # imported as the same module-level dicts her submission_controller.py
    # already owns, e.g.:
    #   from controllers.submission_controller import user_mastery_store, user_hlr_store
    bkt_store: dict = {}
    hlr_store: dict = {}

    # db=None demonstrates the brand-new-user path. Pass a real session
    # here once wired to your backend's db layer: db = get_db_session()
    db = None

    result = get_recommendations(
        user_id=user_id, db=db, redis=None, neo4j=neo4j_store, qdrant=qdrant,
        bkt_store=bkt_store, hlr_store=hlr_store,
        collection="problems_full", total_n=total_n, k=k,
    )
    response = result.to_dict()
    response = resolve_recommendations(response, qdrant, collection="problems_full")
    response["session_mode"] = session_mode
    return response


# ---------------------------------------------------------------------------
# Entry point -- reads a request from stdin / --input-json / CLI flags,
# writes exactly one JSON object to stdout.
# ---------------------------------------------------------------------------

def _load_request(args) -> dict:
    if args.input_json:
        with open(args.input_json, encoding="utf-8") as f:
            return json.load(f)

    if args.user_id:
        # quick manual run -- build the request dict from CLI flags
        req = {"user_id": args.user_id, "k": args.k, "total_n": args.total_n}
        if args.session_mode:
            req["session_mode"] = args.session_mode
        req["force_offline"] = args.force_offline
        req["skip_offline"] = args.skip_offline
        req["no_neo4j"] = args.no_neo4j
        return req

    if not sys.stdin.isatty():
        raw = sys.stdin.read().strip()
        if raw:
            return json.loads(raw)

    raise ValueError(
        "No request provided. Pass a user_id positionally, use --input-json "
        "FILE, or pipe a JSON request via stdin.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("user_id", nargs="?", default=None,
                       help="Quick manual run: user_id as a positional arg. "
                            "Omit this to read a JSON request from stdin or --input-json instead.")
    parser.add_argument("--input-json", default=None,
                       help="Path to a JSON file containing the request.")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--total-n", type=int, default=30)
    parser.add_argument("--session-mode", default=None, choices=["practice", "learning"])
    parser.add_argument("--force-offline", action="store_true")
    parser.add_argument("--skip-offline", action="store_true")
    parser.add_argument("--no-neo4j", action="store_true")
    args = parser.parse_args()

    if args.force_offline and args.skip_offline:
        _log("[X] --force-offline and --skip-offline are mutually exclusive.")
        sys.exit(1)

    try:
        request = _load_request(args)
        response = handle_request(request)
    except Exception as exc:
        # Errors also go out as JSON on stdout -- a caller parsing stdout
        # as JSON always gets valid JSON back, success or failure.
        print(json.dumps({"error": str(exc), "error_type": exc.__class__.__name__}))
        sys.exit(1)

    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
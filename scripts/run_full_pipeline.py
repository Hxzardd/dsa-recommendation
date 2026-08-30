"""
run_full_pipeline.py

ONE file that runs the whole thing, offline through online, taking a JSON
REQUEST and returning a JSON RESPONSE.

CLOUD-READY: every Qdrant and Neo4j credential is imported directly from
db_env.py (the one canonical .env-loading module at repo root) -- nothing
is hardcoded, nothing is re-parsed here. See db_env.py's docstring for the
full list of supported .env keys.

REQUEST (stdin, or --input-json <file>, or CLI flags for quick manual runs):
    {
      "user_id":      "string, required",
      "k":             10,
      "total_n":       30,
      "session_mode":  "practice"
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
          "source":           "course_path",
          "recommended_at":   "2026-07-08T12:00:00+00:00"
        }
      ]
    }

All progress/status output goes to STDERR, never stdout.

Usage:
    echo '{"user_id": "u1", "k": 5}' | python run_full_pipeline.py
    python run_full_pipeline.py --input-json request.json
    python run_full_pipeline.py u1 --k 5

Flags:
    --input-json FILE  Read the JSON request from this file instead of stdin/CLI.
    --force-offline     Re-run the offline pipeline even if Qdrant already looks populated.
    --skip-offline      Never run the offline pipeline, even if empty.
    --no-neo4j          Skip Neo4j durable storage even if reachable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Resolve the repo root (dir holding pyproject.toml) so `import db_env` works
# when this script is invoked from scripts/. Data paths below stay relative to
# the CWD -- run this from the repo root.
for _p in Path(__file__).resolve().parents:
    if (_p / "pyproject.toml").exists():
        sys.path.insert(0, str(_p))
        break

import db_env   # the ONE place .env gets loaded -- see db_env.py

QDRANT_URL     = db_env.QDRANT_URL
QDRANT_API_KEY = db_env.QDRANT_API_KEY
MANIFEST_PATH  = "data/1000_manifest_final.json"
REQUIRED_COLLECTIONS = ["dsa_problems", "problems_full"]


def _qdrant_offline_step_cmd():
    """Embedder subprocess call -- includes the API key flag if one is set."""
    cmd = [
        sys.executable, "pipeline/embeddings/embedder.py",
        "--resume", "--collection", "dsa_problems",
        "--qdrant-url", QDRANT_URL,
    ]
    if QDRANT_API_KEY:
        cmd += ["--qdrant-api-key", QDRANT_API_KEY]
    return cmd


# FIX: this pointed at "question-graph/data/problem_topic_edges_normalized.json",
# which never exists (that directory only ever had problem_topic_edges.json,
# not the _normalized variant) -- generate_topic_edges.py actually writes
# to data/problem_topic_edges_normalized.json (the same file bkt.py/hlr.py
# load at import time). The stale path meant this existence-check was
# ALWAYS False, so every --force-offline run silently re-generated and
# overwrote the topic edges from scratch even when a current, correct
# version (e.g. the taxonomy-reconciled one already in data/) existed.
TOPIC_EDGES_PATH = "data/problem_topic_edges_normalized.json"

OFFLINE_STEPS = [
    ("Ingestion", [
        sys.executable, "pipeline/ingestion/ingest.py",
        "--input", MANIFEST_PATH,
    ]),
]

# Only auto-generate the topic-edges file if it doesn't already exist --
# if you're using a curated version (e.g. the one in data/), this step is
# skipped entirely so it's never overwritten. Same "skip if already
# done" logic as the Qdrant collection check further down.
if not Path(TOPIC_EDGES_PATH).exists():
    OFFLINE_STEPS.append((
        "Generate problem-topic edges (for bkt.py/hlr.py)",
        [sys.executable, "pipeline/ingestion/generate_topic_edges.py"],
    ))

OFFLINE_STEPS += [
    ("Embeddings", _qdrant_offline_step_cmd()),
    ("Graph build + RGCN train + Qdrant ingest", [
        sys.executable, "pipeline/graphs/run_rgcn_pipeline.py",
        "--graph-source", "normalized",
        # build_graph.py / ingest_rgcn_to_qdrant.py import config.py, which
        # imports db_env.py -- no CLI flag needed, they pick up the same
        # credentials this process already loaded.
    ]),
]


def _log(msg: str):
    """All status/progress output goes to stderr -- stdout is reserved for the final JSON response."""
    print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# Step 1: offline pipeline, run-once-only
# ---------------------------------------------------------------------------

def offline_pipeline_already_done(qdrant_url: str) -> bool:
    try:
        client = db_env.qdrant_client(timeout=5)
    except ImportError:
        _log("[!] qdrant-client not installed -- cannot check offline state, "
             "assuming offline pipeline has NOT been run.")
        return False

    try:
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
        # subprocess.run() inherits this process's os.environ by default --
        # every credential db_env.py loaded from .env is automatically
        # visible to each offline-stage subprocess.
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
             "Skipping ingestion/embedding/training (use --force-offline to re-run anyway).")
        return
    run_offline_pipeline()


# ---------------------------------------------------------------------------
# Step 2: Neo4j -- automatic connection via db_env.py, graceful fallback
# ---------------------------------------------------------------------------

def build_neo4j_store(disabled: bool):
    from pipeline.recommender.services.neo4j_graph_store import Neo4jGraphStore

    if disabled:
        _log("[->] --no-neo4j set; skipping durable graph storage.")
        return Neo4jGraphStore(driver=None)

    driver = db_env.neo4j_driver()
    if driver is None:
        _log(f"[!] Neo4j unavailable (not configured, or unreachable at "
             f"{db_env.NEO4J_URI}) -- durable graph storage disabled for this run.")
        return Neo4jGraphStore(driver=None)

    _log(f"[OK] Neo4j connected at {db_env.NEO4J_URI} "
         f"(instance: {db_env.NEO4J_INSTANCENAME or '?'}, "
         f"database: {db_env.NEO4J_DATABASE}) -- user graph "
         f"will persist durably across runs.")
    return Neo4jGraphStore(driver, database=db_env.NEO4J_DATABASE)


# ---------------------------------------------------------------------------
# Step 3: title resolution
# ---------------------------------------------------------------------------

def resolve_problem_ids(problem_ids: list, qdrant, collection: str = "problems_full") -> dict:
    """
    Resolve problem_id strings -> Qdrant payload using scroll() with a
    payload filter. retrieve() by point ID doesn't work here because the
    internal Qdrant point IDs are raw integers assigned at ingest time,
    not the xxhash values the old code was computing.
    """
    if not problem_ids:
        return {}
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        points, _ = qdrant.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="problem_id", match=MatchAny(any=problem_ids))
            ]),
            limit=len(problem_ids),
            with_payload=True,
            with_vectors=False,
        )
        return {
            p.payload["problem_id"]: p.payload
            for p in points
            if p.payload and "problem_id" in p.payload
        }
    except Exception as exc:
        _log(f"[!] Could not resolve problem titles: {exc}")
        return {}


def resolve_recommendations(result_dict: dict, qdrant, collection: str = "problems_full") -> dict:
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

    from pipeline.recommender.services.recommend import get_recommendations

    qdrant = db_env.qdrant_client(timeout=10)

    # BKT/HLR stores and db session are owned by the backend, not this
    # script. This demo runs with empty stores and db=None (cold-start
    # path). In production, the backend controller passes its own
    # bkt_store, hlr_store, and db session into get_recommendations()
    # directly -- the ML service never connects to Postgres itself.
    bkt_store: dict = {}
    hlr_store: dict = {}
    db = None
    conn = None

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
# Entry point
# ---------------------------------------------------------------------------

def _load_request(args) -> dict:
    if args.input_json:
        with open(args.input_json, encoding="utf-8") as f:
            return json.load(f)

    if args.user_id:
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
    parser.add_argument("user_id", nargs="?", default=None)
    parser.add_argument("--input-json", default=None)
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
        print(json.dumps({"error": str(exc), "error_type": exc.__class__.__name__}))
        sys.exit(1)

    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
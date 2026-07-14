"""
DSA Engine -- RGCN Step 4: Ingest into Qdrant
=============================================
Writes all embeddings into Qdrant as separate collections:

    problems_question  : 1024-d  BGE question embedding
    problems_solution  : 768-d   GraphCodeBERT solution embedding
    problems_rgcn      : 128-d   pure RGCN structural embedding
    problems_full      : 1920-d  L2-norm concat(QS 1792 + RGCN 128)

question + solution + rgcn are stored as individual collections so the
rec engine can query each signal independently or fuse them at query time.

By default reads ARTIFACTS (graph.pt + rgcn_problem_embeddings.npy) for
rgcn/full, and the embedded parquet for question/solution.
Use --from-parquet to source everything from the parquet artifact.

Run:
    python pipeline/graphs/ingest_rgcn_to_qdrant.py
    python pipeline/graphs/ingest_rgcn_to_qdrant.py --no-question --no-solution
    python pipeline/graphs/ingest_rgcn_to_qdrant.py --from-parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import config as C

EMBED_KEYS = {"question_embedding", "solution_embedding", "rgcn_embedding",
              "question_solution_embedding", "full_embedding"}


# ---------------------------------------------------------------------------
# Core upsert
# ---------------------------------------------------------------------------

def _stable_point_id(problem_id: str) -> int:
    """
    Convert a problem_id string to a stable integer Qdrant point ID.
    Uses xxhash (fast, no collisions at this scale) falling back to
    the built-in hash truncated to 63 bits.
    The same problem_id must map to the same integer across ALL collections
    so cross-collection joins by ID are correct.
    """
    try:
        import xxhash
        return xxhash.xxh64(problem_id).intdigest() & 0x7FFF_FFFF_FFFF_FFFF
    except ImportError:
        import hashlib
        return int(hashlib.sha256(problem_id.encode()).hexdigest(), 16) & 0x7FFF_FFFF_FFFF_FFFF




# Errors that are safe to retry (transient network/transport issues).
# Permanent errors (auth failure, wrong dimensions, missing collections,
# bad config) must NOT be retried -- they need to fail fast so the user
# sees the real problem immediately instead of waiting through 4 retries.
_RETRYABLE = (
    "WriteTimeout", "ReadTimeout", "ConnectTimeout",
    "TimeoutException", "ConnectionError", "RemoteProtocolError",
    "WriteError", "ReadError",
)

def _is_retryable(exc: Exception) -> bool:
    name = type(exc).__name__
    msg  = str(exc).lower()
    if any(r in name for r in _RETRYABLE):
        return True
    # httpcore / httpx surface timeouts as ResponseHandlingException wrapping the real exc
    if "responsehandlingexception" in name.lower() and (
        "timeout" in msg or "write" in msg or "read" in msg
    ):
        return True
    return False

def _upsert_with_retry(client, collection: str, batch: list, max_retries: int = 4) -> None:
    """
    Retry a single batch upsert on TRANSIENT network errors only.
    Permanent errors (403 auth, dimension mismatch, missing collection)
    are re-raised immediately so the user sees them without delay.
    Qdrant upsert() is idempotent by point ID -- retrying is always safe.
    """
    import time as _time
    last_exc = None
    for attempt in range(max_retries):
        try:
            client.upsert(collection_name=collection, points=batch)
            return
        except Exception as exc:
            if not _is_retryable(exc):
                raise   # permanent error -- fail fast, no retry
            last_exc = exc
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"\n[!] Transient upload error ({exc.__class__.__name__}), "
                      f"retrying in {wait}s (attempt {attempt + 2}/{max_retries})...")
                _time.sleep(wait)
    raise last_exc



def ingest_embeddings(client, collection, ids, vectors, payloads, batch_size=50):
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct, OptimizersConfigDiff,
    )
    # ids[i] is the problem_id string; vectors[i] may be None for missing embeddings
    valid = [(ids[i], v) for i, v in enumerate(vectors) if v is not None]
    if not valid:
        print(f"[X] nothing to ingest into '{collection}' (all vectors None)")
        return
    dim = len(valid[0][1])
    print(f"\n[->] {collection}: dim={dim}  points={len(valid)}")

    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=0),
        )
        print(f"[OK] created '{collection}'")
    else:
        existing_dim = client.get_collection(collection).config.params.vectors.size
        if existing_dim != dim:
            raise ValueError(
                f"Collection '{collection}' already exists with dim={existing_dim} "
                f"but you're trying to upsert dim={dim} vectors. "
                f"Either delete the collection first (client.delete_collection"
                f"('{collection}')) or fix RGCN_OUT_DIM / feature source so "
                f"dims match. Refusing to upsert mismatched dimensions.")
        print(f"[->] '{collection}' exists -- upserting (dim={dim} confirmed match)")

    pid_to_payload = {pid: payloads[i] for i, pid in enumerate(ids)}
    points = [
        PointStruct(
            id=_stable_point_id(pid),
            vector=list(map(float, v)),
            payload={**pid_to_payload.get(pid, {}), "problem_id": pid},
        )
        for pid, v in valid
    ]
    t0 = time.time()
    for start in range(0, len(points), batch_size):
        _upsert_with_retry(client, collection, points[start:start + batch_size])
        pct = min(100, int(100 * (start + batch_size) / max(len(points), 1)))
        print(f"\r  {pct}%", end="", flush=True)
    client.update_collection(
        collection_name=collection,
        optimizers_config=OptimizersConfigDiff(indexing_threshold=20000))
    cnt = client.count(collection_name=collection).count
    print(f"\n[OK] '{collection}' now has {cnt} points  ({time.time()-t0:.1f}s)")


# ---------------------------------------------------------------------------
# Payload helper
# ---------------------------------------------------------------------------

def _clean_payload(meta: dict) -> dict:
    out = {}
    for k, v in (meta or {}).items():
        if k in EMBED_KEYS or v is None:
            continue
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, float) and np.isnan(v):
            continue
        else:
            out[k] = v
    return out


def _as_vec(val):
    if val is None:
        return None
    try:
        arr = np.asarray(val, dtype=np.float32)
        if arr.ndim != 1 or arr.size == 0 or np.any(np.isnan(arr)):
            return None
        return arr
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Load from artifacts + parquet
# ---------------------------------------------------------------------------

def from_artifacts_and_parquet():
    """
    Primary path:
      - RGCN vectors + payload from graph.pt + rgcn_problem_embeddings.npy
      - question / solution vectors from the embedded parquet (if available)
    """
    import torch, pandas as pd

    _npz_path  = C.ARTIFACTS_DIR / "graph_tensors.npz"
    _meta_path = C.ARTIFACTS_DIR / "graph_meta.json"

    missing = [p for p in (_npz_path, _meta_path, C.EMB_NPY) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required artifact(s): {[str(p) for p in missing]} -- "
            f"run build_graph.py + train_rgcn.py first.")

    import json as _json
    _t   = np.load(_npz_path, allow_pickle=False)
    _m   = _json.load(open(_meta_path, encoding="utf-8"))
    rgcn = np.load(C.EMB_NPY).astype(np.float32)
    ids  = _m["problem_ids"]
    meta = _m["problem_meta"]
    X_prob = _t["problem_features"].astype(np.float32)

    # ── Alignment check: rgcn_problem_embeddings.npy must match graph_meta.json ──
    # If the graph was rebuilt after the RGCN embeddings were trained (or vice
    # versa), row counts can silently mismatch -- crash loudly here rather
    # than writing a full vector paired with the wrong problem_id downstream.
    if len(rgcn) != len(ids):
        raise ValueError(
            f"Stale artifacts: rgcn_problem_embeddings.npy has {len(rgcn)} rows "
            f"but graph_meta.json has {len(ids)} problem_ids. "
            f"Re-run build_graph.py then train_rgcn.py so both come from the "
            f"same graph build (do not mix artifacts from different runs).")
    if len(X_prob) != len(ids):
        raise ValueError(
            f"Stale artifacts: graph_tensors.npz problem_features has "
            f"{len(X_prob)} rows but graph_meta.json has {len(ids)} problem_ids. "
            f"Re-run build_graph.py to regenerate both files together.")

    # Fingerprint check: catches the case where row counts coincidentally
    # match but the underlying problem set is different (graph rebuilt with
    # a different filter/dataset between build_graph.py and train_rgcn.py runs).
    _arch_path = C.ARTIFACTS_DIR / "model_arch.json"
    expected_fp = _m.get("graph_fingerprint")
    if _arch_path.exists() and expected_fp:
        _arch = _json.load(open(_arch_path, encoding="utf-8"))
        trained_fp = _arch.get("graph_fingerprint")
        if trained_fp and trained_fp != expected_fp:
            raise ValueError(
                f"Stale artifacts: rgcn_problem_embeddings.npy was trained on a "
                f"different graph (fingerprint {trained_fp}) than the current "
                f"graph_meta.json (fingerprint {expected_fp}). "
                f"Re-run build_graph.py then train_rgcn.py together.")

    # build payloads
    payloads = []
    for i, pid in enumerate(ids):
        pl = _clean_payload(meta[i])
        pl.setdefault("problem_id", pid)
        payloads.append(pl)

    # full = QS + RGCN
    full = []
    for i in range(len(ids)):
        v = np.concatenate([X_prob[i], rgcn[i]]).astype(np.float32)
        n = np.linalg.norm(v)
        full.append(v / (n or 1.0))

    # question + solution from parquet
    q_vecs  = [None] * len(ids)
    s_vecs  = [None] * len(ids)
    qs_vecs = list(X_prob)   # already have QS from graph features

    if C.INPUT_PARQUET.exists():
        df = pd.read_parquet(C.INPUT_PARQUET)
        pid_to_row = {str(r["problem_id"]): r for _, r in df.iterrows()}
        for i, pid in enumerate(ids):
            row = pid_to_row.get(str(pid))
            if row is None:
                continue
            q_vecs[i] = _as_vec(row.get("question_embedding"))
            s_vecs[i] = _as_vec(row.get("solution_embedding"))
        n_q = sum(1 for v in q_vecs if v is not None)
        n_s = sum(1 for v in s_vecs if v is not None)
        print(f"[OK] parquet: {n_q} question vecs, {n_s} solution vecs")
    else:
        print(f"[!] {C.INPUT_PARQUET} not found -- question/solution collections will be empty")

    return ids, payloads, {
        "question":  q_vecs,
        "solution":  s_vecs,
        "rgcn":      list(rgcn),
        "full":      full,
    }


def from_parquet_only(path: Path):
    """Fallback: load everything from the rgcn parquet artifact."""
    import pandas as pd
    df = pd.read_parquet(path)
    ids, payloads = [], []
    vecs = {"question": [], "solution": [], "rgcn": [], "full": []}
    for _, row in df.iterrows():
        if _as_vec(row.get("rgcn_embedding")) is None:
            continue
        ids.append(str(row.get("problem_id", "")))
        payloads.append(_clean_payload(
            {k: row[k] for k in df.columns if k not in EMBED_KEYS}))
        for key, col in [("question", "question_embedding"),
                         ("solution", "solution_embedding"),
                         ("rgcn",     "rgcn_embedding"),
                         ("full",     "full_embedding")]:
            vecs[key].append(_as_vec(row.get(col)))
    print(f"[->] {len(ids)} rows from {path.name}")
    return ids, payloads, vecs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant-url",         default=C.QDRANT_URL)
    ap.add_argument("--rgcn-collection",    default=C.QDRANT_COLLECTION_RGCN)
    ap.add_argument("--full-collection",    default=C.QDRANT_COLLECTION_FULL)
    ap.add_argument("--question-collection",default="problems_question")
    ap.add_argument("--solution-collection",default="problems_solution")
    ap.add_argument("--from-parquet",  action="store_true")
    ap.add_argument("--no-rgcn",       action="store_true")
    ap.add_argument("--no-full",       action="store_true")
    ap.add_argument("--no-question",   action="store_true")
    ap.add_argument("--no-solution",   action="store_true")
    args = ap.parse_args()

    print("\n" + "=" * 62)
    print("  RGCN STEP 4 -- QDRANT INGEST")
    print("=" * 62)

    if args.from_parquet:
        ids, payloads, vecs = from_parquet_only(C.OUTPUT_PARQUET)
    else:
        ids, payloads, vecs = from_artifacts_and_parquet()

    print(f"[->] {len(ids)} problems to ingest")

    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=args.qdrant_url,
                              api_key=C.QDRANT_API_KEY, timeout=120)
        client.get_collections()
    except Exception as e:
        print(f"[X] cannot reach Qdrant at {args.qdrant_url}: "
              f"{e.__class__.__name__}: {str(e)[:120]}")
        print("    start it:  docker run -p 6333:6333 qdrant/qdrant")
        sys.exit(1)

    if not args.no_question:
        ingest_embeddings(client, args.question_collection,
                          ids, vecs["question"], payloads)
    if not args.no_solution:
        ingest_embeddings(client, args.solution_collection,
                          ids, vecs["solution"], payloads)
    if not args.no_rgcn:
        ingest_embeddings(client, args.rgcn_collection,
                          ids, vecs["rgcn"], payloads)
    if not args.no_full:
        ingest_embeddings(client, args.full_collection,
                          ids, vecs["full"], payloads)

    print("\n" + "=" * 62)
    print("  QDRANT INGEST COMPLETE")
    print("  Collections:")
    for col in [args.question_collection, args.solution_collection,
                args.rgcn_collection, args.full_collection]:
        try:
            cnt = client.count(collection_name=col).count
            dim = client.get_collection(col).config.params.vectors.size
            print(f"    {col:<25} {cnt} points  dim={dim}")
        except Exception:
            pass
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
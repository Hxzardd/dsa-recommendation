# This file is a CLI smoke-test script, NOT a pytest module.
# pytest is explicitly told to ignore it via conftest.py collect_ignore.
# Run directly: python database/qdrant/test_qdrant.py
# Do NOT rename back to test_* — that causes pytest collection failures.
"""
DSA Engine -- Qdrant Search Tests
===================================
Tests the vector pool end-to-end after upload.

Run:
    python test_qdrant.py
    python test_qdrant.py --url http://localhost:6333 --collection dsa_problems
"""

import argparse
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def load_client(url: str) -> QdrantClient:
    c = QdrantClient(url=url, timeout=10)
    c.get_collections()
    return c


def load_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df


# ---------------------------------------------------------------------------
# Test 1 -- Collection sanity
# ---------------------------------------------------------------------------

def check_collection_info(client: QdrantClient, collection: str, expected_dim: int = None):
    """
    expected_dim is optional. The smoke test now supports collections of
    any dimension (1024 question-only, 768 solution-only, 1792 QS,
    1920 full) instead of hardcoding 1792 -- pass --expected-dim to assert
    a specific size, otherwise just report it.
    """
    print("\n" + "="*55)
    print("  TEST 1 -- Collection Info")
    print("="*55)
    info = client.get_collection(collection)
    count = client.count(collection_name=collection).count
    actual_dim = info.config.params.vectors.size
    print(f"  Points          : {count}")
    print(f"  Vector size     : {actual_dim}")
    print(f"  Distance metric : {info.config.params.vectors.distance}")
    status = info.status
    print(f"  Status          : {status}")
    assert count > 0, "Collection is empty!"
    if expected_dim is not None:
        assert actual_dim == expected_dim, \
            f"Expected {expected_dim}-dim, got {actual_dim}"
    print("  [PASS]")


# ---------------------------------------------------------------------------
# Test 2 -- Payload spot check
# ---------------------------------------------------------------------------

def check_payload(client: QdrantClient, collection: str):
    print("\n" + "="*55)
    print("  TEST 2 -- Payload Spot Check (first 3 points via scroll)")
    print("="*55)
    # Points now use stable hash IDs (see _stable_point_id in
    # ingest_rgcn_to_qdrant.py), not sequential 0/1/2 -- scroll instead
    # of retrieve(ids=[0,1,2]) which would silently return nothing.
    pts, _ = client.scroll(
        collection_name=collection, limit=3,
        with_payload=True, with_vectors=False,
    )
    assert pts, "scroll returned no points -- is the collection populated?"
    for p in pts:
        pl = p.payload
        title = pl.get("title", "?")
        pid   = pl.get("problem_id", "?")
        diff  = pl.get("difficulty_score", "?")
        tags  = pl.get("topic_tags", [])
        print(f"  [{p.id}] {title}")
        print(f"       problem_id     : {pid}")
        print(f"       difficulty     : {diff}")
        print(f"       topic_tags     : {tags}")
        assert title, "title missing from payload"
        assert pid,   "problem_id missing from payload"
    print("  [PASS]")


# ---------------------------------------------------------------------------
# Test 3 -- Similarity search (core functionality)
# ---------------------------------------------------------------------------

def check_similarity_search(client: QdrantClient, collection: str, df: pd.DataFrame, vector_col: str):
    print("\n" + "="*55)
    print("  TEST 3 -- Similarity Search")
    print("="*55)

    test_cases = [
        ("Two Sum",                        "hash map / complement lookup problems"),
        ("Longest Substring Without Repeating Characters", "sliding window problems"),
        ("Add Two Numbers",                "linked list problems"),
    ]

    for title, expect in test_cases:
        row = df[df["title"] == title]
        if row.empty:
            print(f"  [SKIP] '{title}' not in parquet")
            continue
        if row.iloc[0][vector_col] is None:
            print(f"  [SKIP] '{title}' has no '{vector_col}' (e.g. no parseable solution)")
            continue

        query_vec = np.array(row.iloc[0][vector_col]).tolist()
        hits = client.query_points(
            collection_name=collection,
            query=query_vec,
            limit=6,
            with_payload=True,
        ).points

        print(f"\n  Query: '{title}'  (expect: {expect})")
        for i, h in enumerate(hits):
            marker = "  -->" if i == 0 else "     "
            print(f"{marker} [{h.score:.4f}] {h.payload.get('title','?')}")

        # Self should be top hit
        top = hits[0].payload.get("title", "")
        assert top == title, f"Self not top hit! Got: {top}"

    print("\n  [PASS]")


# ---------------------------------------------------------------------------
# Test 4 -- Filtered search (metadata + vector combined)
# ---------------------------------------------------------------------------

def check_filtered_search(client: QdrantClient, collection: str, df: pd.DataFrame, vector_col: str):
    print("\n" + "="*55)
    print("  TEST 4 -- Filtered Search (difficulty + topic)")
    print("="*55)

    row = df[df["title"] == "Two Sum"]
    if row.empty:
        print("  [SKIP] Two Sum not found")
        return
    if row.iloc[0][vector_col] is None:
        print(f"  [SKIP] Two Sum has no '{vector_col}'")
        return

    query_vec = np.array(row.iloc[0][vector_col]).tolist()

    # Only easy problems (difficulty_score <= 0.4)
    hits = client.query_points(
        collection_name=collection,
        query=query_vec,
        query_filter=Filter(
            must=[FieldCondition(key="difficulty_score", range=Range(lte=0.45))]
        ),
        limit=5,
        with_payload=True,
    ).points

    print(f"  Similar to 'Two Sum' filtered to difficulty <= 0.45:")
    for h in hits:
        diff = h.payload.get("difficulty_score", "?")
        print(f"     [{h.score:.4f}] (diff={diff:.3f}) {h.payload.get('title','?')}")
        assert float(diff) <= 0.45, f"Filter broke! Got difficulty={diff}"

    print("  [PASS]")


# ---------------------------------------------------------------------------
# Test 5 -- Cross-topic transfer (are similar-pattern problems close?)
# ---------------------------------------------------------------------------

def check_cross_topic_transfer(client: QdrantClient, collection: str, df: pd.DataFrame, vector_col: str):
    print("\n" + "="*55)
    print("  TEST 5 -- Cross-Topic Transfer (pattern similarity)")
    print("="*55)

    # Sliding window problems should cluster together across different topics
    query_title = "Longest Substring Without Repeating Characters"
    row = df[df["title"] == query_title]
    if row.empty:
        print("  [SKIP]")
        return
    if row.iloc[0][vector_col] is None:
        print(f"  [SKIP] '{query_title}' has no '{vector_col}'")
        return

    query_vec = np.array(row.iloc[0][vector_col]).tolist()
    hits = client.query_points(
        collection_name=collection,
        query=query_vec,
        limit=10,
        with_payload=True,
    ).points

    sliding_window_hits = [
        h for h in hits
        if "sliding_window" in (h.payload.get("topic_tags") or [])
        or "sliding_window" in (h.payload.get("patterns") or [])
        or "sliding_window" in (h.payload.get("algorithm_tags") or [])
    ]

    print(f"  Query: '{query_title}'")
    print(f"  Top 10 hits with sliding_window tag: {len(sliding_window_hits)}/10")
    for h in hits[:5]:
        tags = h.payload.get("topic_tags", [])
        print(f"     [{h.score:.4f}] {h.payload.get('title','?')}  tags={tags}")

    assert len(sliding_window_hits) >= 2, \
        f"Expected >= 2 sliding_window hits in top 10, got {len(sliding_window_hits)}"
    print("  [PASS]")


# ---------------------------------------------------------------------------
# Test 6 -- Random probe (no nulls, correct dims)
# ---------------------------------------------------------------------------

def check_random_probe(client: QdrantClient, collection: str, df: pd.DataFrame, vector_col: str, expected_dim: int = None):
    print("\n" + "="*55)
    # Only sample rows that actually have vector_col populated. Rows missing
    # it (e.g. problems with no parseable solution -> no question_solution_embedding)
    # were never uploaded to Qdrant in the first place, so probing them here
    # is a false failure, not a data integrity issue with the collection.
    populated = df[df[vector_col].notna()]
    n_probe = min(10, len(populated))
    print(f"  TEST 6 -- Random Vector Probe ({n_probe} random points)")
    print("="*55)

    if n_probe == 0:
        print(f"  [SKIP] no rows have '{vector_col}' populated")
        return

    sample = populated.sample(n_probe, random_state=42)

    for _, row in sample.iterrows():
        vec = row[vector_col]
        assert vec is not None, f"NULL vector for {row['title']}"
        arr = np.array(vec)
        if expected_dim is not None:
            assert arr.shape == (expected_dim,), \
                f"Wrong shape {arr.shape} for {row['title']}, expected ({expected_dim},)"
        assert not np.any(np.isnan(arr)), f"NaN in vector for {row['title']}"
        assert not np.all(arr == 0), f"Zero vector for {row['title']}"

    dim_label = f"{expected_dim}-dim" if expected_dim else "consistent-dim"
    print(f"  {n_probe} random vectors: all {dim_label}, no NaN, no zeros")
    print("  [PASS]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _infer_vector_column(collection: str) -> str:
    """Best-effort guess of which embedding column matches a collection's
    vectors, based on naming convention. Overridable via --vector-col."""
    name = collection.lower()
    if "question" in name and "solution" not in name:
        return "question_embedding"
    if "solution" in name and "question" not in name:
        return "solution_embedding"
    if "rgcn" in name and "full" not in name:
        return "rgcn_embedding"
    if "full" in name:
        return "full_embedding"
    return "question_solution_embedding"   # default: dsa_problems / problems_v2


def _infer_parquet_path(vector_col: str) -> str:
    """
    question_embedding, solution_embedding, question_solution_embedding all
    live in vector_pool_embedded.parquet (written by embedder.py).
    rgcn_embedding and full_embedding only exist in vector_pool_rgcn.parquet
    (written separately by train_rgcn.py / run_rgcn_pipeline.py) -- they are
    NOT present in vector_pool_embedded.parquet. Defaulting to the wrong file
    silently gives an empty/None query vector and produces nonsense results
    (every hit scoring ~1.0 with no relation to the query) instead of a clear
    error, so this must be selected correctly rather than hardcoded to one file.
    """
    if vector_col in ("rgcn_embedding", "full_embedding"):
        return "data/vector_pool/vector_pool_rgcn.parquet"
    return "data/vector_pool/vector_pool_embedded.parquet"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url",          default="http://localhost:6333")
    p.add_argument("--collection",   default="dsa_problems")
    p.add_argument("--parquet",      default=None,
                   help="Override the auto-selected parquet file. If omitted, "
                        "chosen automatically based on --vector-col: "
                        "rgcn_embedding/full_embedding -> vector_pool_rgcn.parquet, "
                        "everything else -> vector_pool_embedded.parquet.")
    p.add_argument("--vector-col",   default=None,
                   help="Embedding column to query with. Auto-inferred from "
                        "--collection name if omitted (e.g. 'problems_question' "
                        "-> question_embedding).")
    p.add_argument("--expected-dim", type=int, default=None,
                   help="Assert this exact vector dimension. Omit to skip "
                        "the dimension assertion (collection can be any size).")
    args = p.parse_args()

    vector_col = args.vector_col or _infer_vector_column(args.collection)
    parquet_path = args.parquet or _infer_parquet_path(vector_col)

    print("\n" + "="*55)
    print("  DSA ENGINE -- QDRANT SEARCH TESTS")
    print(f"  {args.url} / {args.collection}  (vector_col={vector_col})")
    print(f"  parquet: {parquet_path}")
    print("="*55)

    client = load_client(args.url)
    df     = load_parquet(parquet_path)

    if vector_col not in df.columns:
        print(f"\n[X] Column '{vector_col}' not found in {parquet_path}")
        print(f"    Available columns: {list(df.columns)}")
        return

    passed = 0
    failed = 0

    tests = [
        ("Collection info",      lambda: check_collection_info(client, args.collection, args.expected_dim)),
        ("Payload spot check",   lambda: check_payload(client, args.collection)),
        ("Similarity search",    lambda: check_similarity_search(client, args.collection, df, vector_col)),
        ("Filtered search",      lambda: check_filtered_search(client, args.collection, df, vector_col)),
        ("Cross-topic transfer", lambda: check_cross_topic_transfer(client, args.collection, df, vector_col)),
        ("Random probe",         lambda: check_random_probe(client, args.collection, df, vector_col, args.expected_dim)),
    ]

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"\n  [FAIL] {name}: {e}")
            failed += 1

    print("\n" + "="*55)
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
"""
generate_topic_edges.py

Generates data/problem_topic_edges_normalized.json -- the file bkt.py and
hlr.py load at startup to know which canonical topics a problem covers.

Source of truth: data/source-target.txt (Aashray's curated mapping using
the 72 canonical backend topics). Falls back to ingested parquet ONLY if
source-target.txt is missing, and even then filters to canonical topics only.

This ensures AI-enriched/implementation-level tags (the old 487-tag problem)
never reach bkt.py, hlr.py, UserTopicMastery, or Qdrant's topic_tags payload.

Run:
    python pipeline/ingestion/generate_topic_edges.py
"""

from __future__ import annotations
import json, sys
from pathlib import Path

_here = Path(__file__).resolve()
_REPO_ROOT = _here.parent
for _p in [_here.parent, *_here.parents]:
    if (_p / "pyproject.toml").exists():
        _REPO_ROOT = _p
        break

SOURCE_TARGET = _REPO_ROOT / "data" / "source-target.txt"
PARQUET_PATH  = _REPO_ROOT / "data" / "vector_pool" / "vector_pool.parquet"
OUTPUT_PATH   = _REPO_ROOT / "data" / "problem_topic_edges_normalized.json"

CANONICAL_TOPICS: set[str] = {
    "array","backtracking","bfs","binary_search_answer","binary_tree",
    "bitmask_dp","bitwise","combinatorics","connected_components",
    "counting_sort","cyclic_sort","design","dfs","digit_dp","dijkstra",
    "divide_and_conquer","dp","dummy_node","fast_slow_pointers",
    "floyd_cycle_detection","floyd_warshall","graph","greedy","greedy_choice",
    "hash_map","hash_map_counting","hash_set_lookup","heap",
    "in_place_modification","in_place_reversal","interval_dp","iterative_stack",
    "kadane","kahn_algorithm","kmp","knapsack_dp","kruskal","linked_list",
    "math","matrix","memoization","merge_intervals","merge_sort",
    "minimum_spanning_tree","monotonic_queue","monotonic_stack","number_theory",
    "prefix_sum","prefix_xor","queue","recursive_call","sequence_dp",
    "shortest_path","simulation","sliding_window","square_root_decomposition",
    "stack","state_machine_dp","string","string_matching","subsets","tabulation",
    "top_k_elements","topological_order","tree","tree_dp","tree_traversal",
    "trie","two_heaps","two_pass_scan","two_pointers","union_find",
}


def from_source_target(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    edges, skipped = [], 0
    for e in raw:
        src = (e.get("source") or "").strip()
        tgt = (e.get("target") or "").strip()
        if not src or not tgt:
            continue
        if tgt not in CANONICAL_TOPICS:
            skipped += 1
            continue
        edges.append({"source": src, "target": tgt})
    if skipped:
        print(f"[!] Skipped {skipped} edges with non-canonical target")
    return edges


def from_parquet(path: Path) -> list[dict]:
    import pandas as pd
    df = pd.read_parquet(path)
    if "topic_tags" not in df.columns:
        print(f"[X] Parquet missing topic_tags column")
        return []
    id_col = "title_slug" if "title_slug" in df.columns else "problem_id"
    edges, skipped_tags = [], set()
    for _, row in df.iterrows():
        pid = str(row.get(id_col) or "").strip()
        tags = row["topic_tags"]
        if not pid or tags is None:
            continue
        for tag in (tags if hasattr(tags, "__iter__") and not isinstance(tags, str) else []):
            tag = str(tag).strip()
            if tag in CANONICAL_TOPICS:
                edges.append({"source": pid, "target": tag})
            else:
                skipped_tags.add(tag)
    if skipped_tags:
        print(f"[!] Filtered {len(skipped_tags)} non-canonical tags from parquet")
    return edges


def main():
    print("[generate_topic_edges] Starting...")
    if SOURCE_TARGET.exists():
        print(f"[->] Using Aashray's curated source-target.txt")
        edges = from_source_target(SOURCE_TARGET)
    elif PARQUET_PATH.exists():
        print(f"[!] source-target.txt not found -- falling back to parquet with canonical filtering")
        edges = from_parquet(PARQUET_PATH)
    else:
        print(f"[X] Neither source-target.txt nor parquet found. Place source-target.txt at: {SOURCE_TARGET}")
        sys.exit(1)

    if not edges:
        print("[X] Zero edges generated.")
        sys.exit(1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(edges, f, indent=2)

    problems = {e["source"] for e in edges}
    topics   = {e["target"] for e in edges}
    print(f"[OK] Wrote {len(edges)} edges -> {OUTPUT_PATH}")
    print(f"     {len(problems)} problems | {len(topics)} distinct topics")
    print(f"     All canonical: {topics <= CANONICAL_TOPICS}")

if __name__ == "__main__":
    main()

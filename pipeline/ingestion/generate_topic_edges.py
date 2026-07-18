"""
generate_topic_edges.py

Generates data/problem_topic_edges_normalized.json -- the file bkt.py and
hlr.py load at startup to know which canonical topics a problem covers.

Source of truth: data/source-target.txt (). Falls back to ingested parquet ONLY if
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

# FIX: this was the OLD, pre-reconciliation 72-tag ML taxonomy -- kept in
# sync by hand, which is exactly how it went stale. Now loaded straight
# from data/topic_tags_taxonomy_v2.json's canonical_tags (the same file
# database/postgres/topic_taxonomy.py treats as the single source of
# truth), so the two can never drift apart again. See topic_taxonomy.py's
# module docstring for the full reconciliation story.
import sys as _sys
_sys.path.insert(0, str(_REPO_ROOT))
from database.postgres.topic_taxonomy import CANONICAL_TAGS as CANONICAL_TOPICS


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
        print(f"[->] Using  curated source-target.txt")
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

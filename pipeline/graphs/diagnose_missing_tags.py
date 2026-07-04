"""
Diagnostic: list problems with missing/empty topic_tags in Qdrant payload.

Confirms whether the sanity check's "140 missing" is genuine data sparsity
(topic_tags: [] in the source) vs a real payload bug. Also flags the
downstream impact: these problems are invisible to any pool that filters
by topic_tags (pools A, D, E, F, G) and only reachable via vector similarity
(pools B_C, vector).

Run:
    python pipeline/graphs/diagnose_missing_tags.py
"""

from __future__ import annotations

from qdrant_client import QdrantClient

COLLECTION = "problems_full"
QDRANT_URL = "http://localhost:6333"


def main():
    client = QdrantClient(url=QDRANT_URL, timeout=10)

    missing = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            limit=200,
            offset=offset,
            with_payload=["problem_id", "title", "topic_tags"],
            with_vectors=False,
        )
        for p in points:
            pl = p.payload or {}
            tags = pl.get("topic_tags")
            if not tags:   # None or empty list
                missing.append({
                    "problem_id": pl.get("problem_id", str(p.id)),
                    "title": pl.get("title", "?"),
                    "topic_tags": tags,
                })
        if offset is None:
            break

    print(f"Problems with missing/empty topic_tags: {len(missing)}")
    print()
    for m in missing[:30]:
        print(f"  {m['problem_id']}  {m['title']!r}  topic_tags={m['topic_tags']}")
    if len(missing) > 30:
        print(f"  ... and {len(missing) - 30} more")

    print()
    print("Downstream impact: these problems are filtered out of any pool")
    print("that queries by topic_tags (CoursePathPool, WeaknessPool,")
    print("SpacedReviewPool, StretchPool, NoveltyPool). They remain reachable")
    print("only via TransferPool/VectorPool (pure ANN, no topic_tags filter).")
    print()
    print("Fix: re-run tag enrichment on these problem_ids, or manually")
    print("assign topic_tags before the next embedding/ingest cycle.")


if __name__ == "__main__":
    main()
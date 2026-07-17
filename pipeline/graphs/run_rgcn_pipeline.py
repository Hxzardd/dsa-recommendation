"""
DSA Engine -- RGCN Pipeline Runner
==================================
build graph -> train RGCN -> ingest into Qdrant.

By default the pipeline is DB-fed end to end:
  * problem vectors + tags are read FROM Qdrant (the embedder's collection)
  * curated concept edges can be read FROM Neo4j (--graph-source neo4j)
  * the learned embeddings are written back INTO Qdrant

Run from your PROJECT ROOT:

    python rgcn/run_rgcn_pipeline.py
    python rgcn/run_rgcn_pipeline.py --feature-source qdrant --graph-source neo4j
    python rgcn/run_rgcn_pipeline.py --graph-source normalized   # JSON files
    python rgcn/run_rgcn_pipeline.py --feature-source parquet     # offline/file mode
    python rgcn/run_rgcn_pipeline.py --skip-qdrant
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path



def main():
    ap = argparse.ArgumentParser(description="DSA Engine -- RGCN pipeline")
    ap.add_argument("--feature-source", choices=["qdrant", "parquet"], default=None,
                    help="where problem vectors+tags come from (default qdrant)")
    ap.add_argument("--source-collection", default=None,
                    help="Qdrant collection the embedder uploaded to (default dsa_problems)")
    ap.add_argument("--graph-source", choices=["tags", "normalized", "neo4j"], default=None,
                    help="curated concept edges: none / JSON files / Neo4j")
    ap.add_argument("--graph-dir", default=None,
                    help=" graph JSON folder (default ./question-graph/data)")
    ap.add_argument("--problem-feature", default=None,
                    choices=["question_solution_embedding", "question_embedding",
                             "solution_embedding"])
    ap.add_argument("--concept-features", choices=["centroid", "text"], default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--qdrant-url", default=None)
    ap.add_argument("--no-parquet", action="store_true", help="don't write the parquet artifact")
    ap.add_argument("--skip-qdrant", action="store_true", help="don't write to Qdrant")
    args = ap.parse_args()

    # CLI -> env BEFORE importing config
    if args.feature_source:    os.environ["RGCN_FEATURE_SOURCE"] = args.feature_source
    if args.source_collection: os.environ["QDRANT_SOURCE_COLLECTION"] = args.source_collection
    if args.graph_source:      os.environ["RGCN_GRAPH_SOURCE"] = args.graph_source
    if args.graph_dir:         os.environ["RGCN_GRAPH_DIR"] = args.graph_dir
    if args.problem_feature:   os.environ["RGCN_PROBLEM_FEATURE_COL"] = args.problem_feature
    if args.concept_features:  os.environ["RGCN_CONCEPT_FEATURE_MODE"] = args.concept_features
    if args.epochs:            os.environ["RGCN_EPOCHS"] = str(args.epochs)
    if args.qdrant_url:        os.environ["QDRANT_URL"] = args.qdrant_url
    if args.no_parquet:        os.environ["RGCN_WRITE_PARQUET"] = "0"

    import config as C
    from build_graph import build_graph
    from train_rgcn import train

    print("\n" + "#" * 62)
    print("#  DSA ENGINE -- RGCN PIPELINE")
    print("#" * 62)
    print(C.summary())

    graph_data = build_graph()

    # Mirror the offline concept graph (topic nodes, problem nodes,
    # problem-topic edges, topic-topic edges) into Neo4j, kept separate
    # from the online per-user graph -- see pipeline/graphs/
    # neo4j_offline_writer.py's module docstring. Reuses
    # push_offline_graph_to_neo4j.py's main() so this reads straight from
    # the same question-graph/data/*.json files regenerate_graph_artifacts.py
    # just (re)wrote (not graph_data's tensors), and there is exactly ONE
    # place that knows how to do this push, not two independently-drifting
    # copies. Additive and best-effort: NEO4J_PASSWORD unset (or any
    # connection failure) just skips this, same as every other Neo4j
    # touchpoint in this repo -- Qdrant + the local .npz/.json artifacts
    # remain the actual source of truth this repo runs on.
    try:
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
        import pipeline.graphs.push_offline_graph_to_neo4j as _push
        _push.main()
    except SystemExit:
        pass   # _push.main() exits non-zero on a missing file / unreachable Neo4j -- non-fatal here
    except Exception as e:
        print(f"\n[!] Neo4j offline graph mirror skipped: {e.__class__.__name__}: {str(e)[:140]}")

    final_cluster = train()   # train() returns final topic-clustering score, not AUC

    if not args.skip_qdrant:
        try:
            from qdrant_client import QdrantClient
            import ingest_rgcn_to_qdrant as ing
            client = QdrantClient(url=C.QDRANT_URL, api_key=C.QDRANT_API_KEY, timeout=30)
            client.get_collections()
            ids, payloads, vecs = ing.from_artifacts_and_parquet()
            ing.ingest_embeddings(client, C.QDRANT_COLLECTION_RGCN, ids, vecs["rgcn"], payloads)
            ing.ingest_embeddings(client, C.QDRANT_COLLECTION_FULL, ids, vecs["full"], payloads)
        except Exception as e:
            print(f"\n[!] Qdrant write skipped: {e.__class__.__name__}: {str(e)[:140]}")
            print("    Re-run later:  python pipeline/graphs/ingest_rgcn_to_qdrant.py")
    else:
        print("\n[->] --skip-qdrant set; ingest later with pipeline/graphs/ingest_rgcn_to_qdrant.py")

    print("\n" + "#" * 62)
    print(f"#  RGCN PIPELINE COMPLETE   final_cluster={final_cluster:.4f}")
    print("#" * 62 + "\n")


if __name__ == "__main__":
    main()

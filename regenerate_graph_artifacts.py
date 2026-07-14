"""
regenerate_graph_artifacts.py

Regenerates the Problem-Concept Graph (PCG) and Concept-Concept Graph (CCG)
artifacts from Aashray's curated source-target.txt, which already uses the
72 canonical backend topics. Replaces the old AI-enriched 487-tag graph.

What this replaces / why:
    The old problem_topic_edges_normalized.json was built from AI-enriched
    tags that produced ~487 distinct topics -- far more granular than the
    backend's taxonomy and causing BKT/HLR to update topics the backend
    table doesn't even have columns for. source-target.txt is Aashray's
    hand-curated mapping using exactly the 72 canonical tags the backend
    uses. This script uses it as the single source of truth.

Output files (same names as before -- nothing downstream changes):

    data/
        problem_topic_edges_normalized.json   <- direct from source-target.txt
        topic_topic_edges_normalized.json     <- co-occurrence derived

    question-graph/data/
        problem_nodes.json
        topic_nodes.json
        problem_topic_edges.json
        topic_topic_edges.json

Run:
    python regenerate_graph_artifacts.py
    python regenerate_graph_artifacts.py --source data/source-target.txt
    python regenerate_graph_artifacts.py --manifest data/1000_manifest_final.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

# ---------------------------------------------------------------------------
# Find repo root
# ---------------------------------------------------------------------------

_here = Path(__file__).resolve()
REPO_ROOT = _here.parent
for _p in [_here.parent, *_here.parents]:
    if (_p / "pyproject.toml").exists():
        REPO_ROOT = _p
        break


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_SOURCE_TARGET = REPO_ROOT / "data" / "source-target.txt"
DEFAULT_MANIFEST      = REPO_ROOT / "data" / "1000_manifest_final.json"

# data/ outputs (used by bkt.py, hlr.py, generate_dataset.py, sources.py)
out_pt_edges_norm   = REPO_ROOT / "data" / "problem_topic_edges_normalized.json"
out_tt_edges_norm   = REPO_ROOT / "data" / "topic_topic_edges_normalized.json"

# question-graph/data/ outputs (used by graph_builder.py and RGCN pipeline)
QG_DATA             = REPO_ROOT / "question-graph" / "data"
out_problem_nodes   = QG_DATA / "problem_nodes.json"
out_topic_nodes     = QG_DATA / "topic_nodes.json"
out_qg_pt_edges     = QG_DATA / "problem_topic_edges.json"
out_qg_tt_edges     = QG_DATA / "topic_topic_edges.json"


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def load_source_target(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Filter out entries with empty source (a few sentinel rows in Aashray's file)
    return [e for e in data if e.get("source") and e.get("target")]


def load_manifest(path: Path) -> dict:
    """Returns {title_slug: {problem_id, title, title_slug}}"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for p in data:
        slug = p.get("title_slug")
        if slug:
            result[slug] = {
                "problem_id": p.get("problem_id", slug),
                "title":      p.get("title", slug),
                "title_slug": slug,
            }
    return result


def build_problem_topic_map(edges: list[dict]) -> dict[str, list[str]]:
    """Returns {title_slug: [topic, ...]}"""
    pt: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        pt[e["source"]].append(e["target"])
    return dict(pt)


def build_topic_cooccurrence(
    pt_map: dict[str, list[str]]
) -> list[dict]:
    """
    Derives topic-topic co-occurrence edges from the problem-topic mapping.
    For each problem, every pair of topics that co-occur gets an edge.
    Returns list of {source, target, edgeType, shared_problem_count, jaccard}.

    Jaccard = shared_problems / union_problems, where:
        shared_problems = problems tagged with BOTH topic A and topic B
        union_problems  = problems tagged with A OR B (or both)
    """
    # Count how many problems each topic appears in
    topic_problem_counts: dict[str, set[str]] = defaultdict(set)
    pair_counts: dict[tuple[str,str], int] = defaultdict(int)

    for slug, topics in pt_map.items():
        for t in topics:
            topic_problem_counts[t].add(slug)
        # All ordered pairs (alphabetically to avoid duplicates)
        for a, b in combinations(sorted(set(topics)), 2):
            pair_counts[(a, b)] += 1

    edges = []
    for (a, b), shared in pair_counts.items():
        union = len(topic_problem_counts[a] | topic_problem_counts[b])
        jaccard = round(shared / union, 6) if union > 0 else 0.0
        # Both directions so the graph is undirected from a query perspective
        for src, tgt in [(a, b), (b, a)]:
            edges.append({
                "source":              src,
                "target":              tgt,
                "edgeType":            "CO_OCCURS_WITH",
                "shared_problem_count": shared,
                "jaccard":             jaccard,
            })

    return edges


def write_json(path: Path, data, label: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  [OK] {label}: {len(data)} entries -> {path.relative_to(REPO_ROOT)}")


def _find_repo_root(start: Path) -> Path:
    """Walk up from start until we find pyproject.toml."""
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return start


def main(source_target: Path, manifest_path: Path):
    print(f"\n{'='*64}")
    print("  REGENERATING GRAPH ARTIFACTS")
    print(f"  Source: {source_target}")
    print(f"{'='*64}\n")

    # Derive repo root from the source-target file location at runtime
    repo_root = _find_repo_root(source_target.resolve().parent)

    out_pt_edges_norm = repo_root / "data" / "problem_topic_edges_normalized.json"
    out_tt_edges_norm = repo_root / "data" / "topic_topic_edges_normalized.json"
    qg_data           = repo_root / "question-graph" / "data"
    out_problem_nodes = qg_data / "problem_nodes.json"
    out_topic_nodes   = qg_data / "topic_nodes.json"
    out_qg_pt_edges   = qg_data / "problem_topic_edges.json"
    out_qg_tt_edges   = qg_data / "topic_topic_edges.json"

    # -----------------------------------------------------------------------
    # Load inputs
    # -----------------------------------------------------------------------
    print("[1/4] Loading inputs...")
    edges = load_source_target(source_target)
    manifest = load_manifest(manifest_path)

    topics_set = sorted({e["target"] for e in edges})
    slugs_set  = sorted({e["source"] for e in edges})
    print(f"      {len(edges)} edges | {len(slugs_set)} problems | {len(topics_set)} topics")

    # -----------------------------------------------------------------------
    # Build derived structures
    # -----------------------------------------------------------------------
    pt_map = build_problem_topic_map(edges)
    tt_edges = build_topic_cooccurrence(pt_map)

    # -----------------------------------------------------------------------
    # data/ outputs
    # NOTE: bkt.py/hlr.py look up topics by title_slug, so source-target.txt
    # format ({"source": title_slug, "target": topic}) is exactly right here.
    # -----------------------------------------------------------------------
    print("\n[2/4] Writing data/ files (used by bkt.py, hlr.py, generate_dataset.py)...")

    # problem_topic_edges_normalized.json -- direct from source-target.txt
    write_json(out_pt_edges_norm, edges, "problem_topic_edges_normalized.json")

    # topic_topic_edges_normalized.json -- derived co-occurrence
    write_json(out_tt_edges_norm, tt_edges, "topic_topic_edges_normalized.json")

    # -----------------------------------------------------------------------
    # question-graph/data/ outputs
    # NOTE: the RGCN pipeline uses problem_id (the manifest hash) as the
    # node key, so problem_nodes.json and problem_topic_edges.json use
    # problem_id (hash), while data/problem_topic_edges_normalized.json
    # uses title_slug (for the online bkt.py lookup). These are separate
    # files serving separate purposes.
    # -----------------------------------------------------------------------
    print("\n[3/4] Writing question-graph/data/ files (used by RGCN pipeline)...")

    problem_nodes = []
    for slug in slugs_set:
        info = manifest.get(slug, {})
        problem_nodes.append({
            "problem_id": info.get("problem_id", slug),   # manifest hash
            "title":      info.get("title", slug),
            "title_slug": slug,
        })
    write_json(out_problem_nodes, problem_nodes, "problem_nodes.json")

    topic_nodes = [
        {
            "topic_id":   t,
            "topic_name": t.replace("_", " ").title(),
        }
        for t in topics_set
    ]
    write_json(out_topic_nodes, topic_nodes, "topic_nodes.json")

    # problem_topic_edges.json for graph_builder -- uses manifest problem_id
    slug_to_pid = {slug: manifest.get(slug, {}).get("problem_id", slug) for slug in slugs_set}
    qg_pt_edges = [
        {
            "problem_id": slug_to_pid[e["source"]],
            "topic_id":   e["target"],
        }
        for e in edges
    ]
    write_json(out_qg_pt_edges, qg_pt_edges, "problem_topic_edges.json")

    # topic_topic_edges.json for graph_builder
    # Deduplicate to one direction only (graph_builder expects directed edges,
    # not both directions -- the bidirectional data lives in topic_topic_edges_normalized)
    seen_pairs: set[tuple[str,str]] = set()
    qg_tt_edges = []
    for e in tt_edges:
        pair = (e["source"], e["target"])
        rev  = (e["target"], e["source"])
        if pair not in seen_pairs and rev not in seen_pairs:
            seen_pairs.add(pair)
            qg_tt_edges.append({
                "source_topic_id":      e["source"],
                "target_topic_id":      e["target"],
                "shared_problem_count": e["shared_problem_count"],
                "jaccard":              e["jaccard"],
            })
    write_json(out_qg_tt_edges, qg_tt_edges, "topic_topic_edges.json")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n[4/4] Verification summary:")
    print(f"      Problems with at least 1 topic:  {len(pt_map)}")
    print(f"      Canonical topics:                {len(topics_set)}")
    print(f"      Problem-topic edges:             {len(edges)}")
    print(f"      Topic-topic pairs (undirected):  {len(qg_tt_edges)}")
    print(f"      Topic-topic edges (bidirectional): {len(tt_edges)}")
    print()
    print("  All artifacts generated. Next step:")
    print("  Run the RGCN pipeline to rebuild embeddings:")
    print()
    print("    python pipeline/graphs/run_rgcn_pipeline.py --graph-source normalized")
    print()
    print("  Or if you want to fully re-run the offline pipeline:")
    print()
    print("    python run_full_pipeline.py <user_id> --force-offline")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",   default=str(DEFAULT_SOURCE_TARGET),
                        help="Path to source-target.txt (Aashray's curated mapping)")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                        help="Path to 1000_manifest_final.json")
    args = parser.parse_args()
    main(Path(args.source), Path(args.manifest))
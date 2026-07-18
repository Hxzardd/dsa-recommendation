"""
pipeline/graphs/push_offline_graph_to_neo4j.py

Pushes ALL the graph-related JSON files -- topic nodes, problem nodes,
problem-topic edges, topic-topic edges -- into Neo4j as the offline
concept graph. Reads exactly what regenerate_graph_artifacts.py writes:

    question-graph/data/
        topic_nodes.json           -> :OfflineTopic nodes (with .name)
        problem_nodes.json         -> :OfflineProblem nodes (with .title, .title_slug)
        problem_topic_edges.json   -> :OfflineProblem -[:HAS_TOPIC_OFFLINE]-> :OfflineTopic
        topic_topic_edges.json     -> :OfflineTopic -[:CO_OCCURS_OFFLINE]-> :OfflineTopic

Run this any time after regenerate_graph_artifacts.py -- no GPU/RGCN
training needed, this is a pure data push.

Requires NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD reachable (see db_env.py's
docstring) -- exits with an error message, not a silent no-op, if Neo4j
is unreachable, so this is safe to use as a real verification step rather
than something that might have quietly done nothing.

Run:
    python pipeline/graphs/push_offline_graph_to_neo4j.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Guarantees db_env.neo4j_driver()'s diagnostic log.error(...) is actually
# visible when this runs as a standalone script -- without a configured
# handler, a bare `logging.getLogger(...).error(...)` call can go nowhere
# depending on the interpreter's default logging state, which is exactly
# the kind of silent failure this script's docstring promises not to have.
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import db_env
from pipeline.graphs.neo4j_offline_writer import write_offline_graph, clear_offline_graph

QG_DATA = _REPO_ROOT / "question-graph" / "data"
TOPIC_NODES_PATH = QG_DATA / "topic_nodes.json"
PROBLEM_NODES_PATH = QG_DATA / "problem_nodes.json"
PROBLEM_TOPIC_EDGES_PATH = QG_DATA / "problem_topic_edges.json"
TOPIC_TOPIC_EDGES_PATH = QG_DATA / "topic_topic_edges.json"

REQUIRED_PATHS = [
    TOPIC_NODES_PATH, PROBLEM_NODES_PATH,
    PROBLEM_TOPIC_EDGES_PATH, TOPIC_TOPIC_EDGES_PATH,
]


def _load(path: Path) -> list:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    driver = db_env.neo4j_driver()
    if driver is None:
        print("[X] Neo4j unreachable -- check NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD "
              "in .env and that this machine can reach the host.")
        sys.exit(1)
    driver.close()

    missing = [p for p in REQUIRED_PATHS if not p.exists()]
    if missing:
        print("[X] Missing input file(s). Run `python regenerate_graph_artifacts.py` first.")
        for p in missing:
            print(f"    MISSING: {p}")
        sys.exit(1)

    topic_nodes = _load(TOPIC_NODES_PATH)
    problem_nodes = _load(PROBLEM_NODES_PATH)

    problem_topic_edges = [
        (e["problem_id"], e["topic_id"])
        for e in _load(PROBLEM_TOPIC_EDGES_PATH)
        if e.get("problem_id") and e.get("topic_id")
    ]
    cooccurs_edges = [
        (e["source"], e["target"], float(e.get("jaccard", 0.0)))
        for e in _load(TOPIC_TOPIC_EDGES_PATH)
        if e.get("source") and e.get("target")
    ]

    print(f"[->] Pushing to Neo4j database {db_env.NEO4J_OFFLINE_DATABASE!r}:")
    print(f"       {len(topic_nodes)} topic nodes")
    print(f"       {len(problem_nodes)} problem nodes")
    print(f"       {len(problem_topic_edges)} HAS_TOPIC_OFFLINE edges")
    print(f"       {len(cooccurs_edges)} CO_OCCURS_OFFLINE edges")

    clear_offline_graph()
    ok = write_offline_graph(
        cooccurs_edges=cooccurs_edges,
        problem_topic_edges=problem_topic_edges,
        topic_nodes=topic_nodes,
        problem_nodes=problem_nodes,
    )
    if ok:
        print("[OK] Offline graph pushed to Neo4j.")
    else:
        print("[X] Push failed -- see warnings above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
pipeline/graphs/neo4j_offline_writer.py

Reads AND writes the OFFLINE concept graph (topic-topic co-occurrence
structure, plus problem-topic membership) to/from Neo4j, kept separate
from the ONLINE per-user graph Neo4jGraphStore
(pipeline/recommender/services/neo4j_graph_store.py) writes -- that one's
:User/:Concept/:CC_EDGE nodes represent a specific user's live state;
this module's :OfflineTopic/:OfflineProblem nodes represent the static,
shared concept structure every user's recommendations are built against.
Keeping them apart means a full offline pipeline re-run (which MERGEs/
overwrites this module's nodes wholesale) can never touch or corrupt a
live user's online graph, and vice versa.

Separation is two-layered:
  1. Database: reads/writes db_env.NEO4J_OFFLINE_DATABASE (falls back to
     NEO4J_DATABASE if unset).
  2. Labels/relationship types: :OfflineTopic / :OfflineProblem (not
     :Concept), CO_OCCURS_OFFLINE / PREREQ_OFFLINE / HAS_TOPIC_OFFLINE
     (not CC_EDGE) -- so the separation holds even when both end up in
     the same physical database (Neo4j Community Edition only supports
     one database per instance).

Callers (pipeline/recommender/services/user_graph_service.py's
load_offline_concept_graph, pipeline/recommender/bkt.py, pipeline/
recommender/hlr.py) try Neo4j FIRST and fall back to the local JSON
files (question-graph/data/topic_topic_edges.json,
data/problem_topic_edges_normalized.json) if Neo4j is unavailable --
same graceful-degrade convention as every other Neo4j touchpoint in this
repo. Neo4j being unreachable (unset credentials, network partition, DNS
failure) is never fatal; it just means this run uses the static files
instead of the shared, centrally-updated graph.

Usage (write, e.g. from the offline pipeline after regenerate_graph_artifacts.py):
    from pipeline.graphs.neo4j_offline_writer import write_offline_graph

    write_offline_graph(
        cooccurs_edges=[("array", "two_pointers", 0.42), ...],
        prereq_edges=[("array", "two_pointers"), ...],
        problem_topic_edges=[("two-sum", "array"), ...],   # (title_slug, topic)
    )

Usage (read, e.g. from user_graph_service.py / bkt.py at startup):
    from pipeline.graphs.neo4j_offline_writer import (
        load_cooccurs_edges, load_prereq_edges, load_problem_topics,
    )
"""

from __future__ import annotations

import logging
from typing import Iterable

import db_env

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def write_offline_graph(
    cooccurs_edges: Iterable[tuple] = (),
    prereq_edges: Iterable[tuple] = (),
    problem_topic_edges: Iterable[tuple] = (),
    topic_nodes: Iterable[dict] = (),
    problem_nodes: Iterable[dict] = (),
) -> bool:
    """
    cooccurs_edges:      iterable of (source_slug, target_slug, weight).
    prereq_edges:        iterable of (prereq_slug, unlocks_slug).
    problem_topic_edges: iterable of (problem_id, topic_slug) -- problem_id
                          is this repo's own ingestion hash (matches
                          question-graph/data/problem_topic_edges.json and
                          problem_nodes.json's problem_id, NOT the backend's
                          problem.problem_id CUID -- see database/postgres/
                          db.py's resolve_problem_ids_by_title_slugs
                          docstring for that distinction).
    topic_nodes:          iterable of {topic_slug/topic_id, topic_name}
                          dicts (question-graph/data/topic_nodes.json's own
                          shape) -- sets :OfflineTopic.name.
    problem_nodes:        iterable of {problem_id, title, title_slug} dicts
                          (question-graph/data/problem_nodes.json's own
                          shape) -- sets :OfflineProblem.title/.title_slug
                          alongside .problem_id, so a caller can look a
                          problem up by either key.

    Returns True if the write succeeded, False if Neo4j is unavailable
    (matches Neo4jGraphStore's graceful no-op-on-unavailable convention --
    the offline pipeline's other outputs, Qdrant + local JSON/parquet
    files, are still the source of truth this repo runs on day to day;
    Neo4j mirroring here is additive, not required).
    """
    driver = db_env.neo4j_driver()
    if driver is None:
        log.warning("Neo4j unavailable -- skipping offline graph write "
                    "(Qdrant/local files remain the source of truth).")
        return False

    cooccurs_edges = list(cooccurs_edges)
    prereq_edges = list(prereq_edges)
    problem_topic_edges = list(problem_topic_edges)
    topic_nodes = list(topic_nodes)
    problem_nodes = list(problem_nodes)

    try:
        with driver.session(database=db_env.NEO4J_OFFLINE_DATABASE) as session:
            session.execute_write(
                _write_tx, cooccurs_edges, prereq_edges, problem_topic_edges,
                topic_nodes, problem_nodes,
            )
        log.info(
            "Wrote %d topic nodes, %d problem nodes, %d CO_OCCURS_OFFLINE, "
            "%d PREREQ_OFFLINE, %d HAS_TOPIC_OFFLINE edges to Neo4j "
            "database %r.",
            len(topic_nodes), len(problem_nodes), len(cooccurs_edges),
            len(prereq_edges), len(problem_topic_edges),
            db_env.NEO4J_OFFLINE_DATABASE,
        )
        return True
    finally:
        driver.close()


def _write_tx(tx, cooccurs_edges: list, prereq_edges: list, problem_topic_edges: list,
              topic_nodes: list, problem_nodes: list) -> None:
    for t in topic_nodes:
        slug = t.get("topic_slug") or t.get("topic_id") or t.get("slug")
        if not slug:
            continue
        tx.run(
            """
            MERGE (t:OfflineTopic {slug: $slug})
            SET t.name = $name
            """,
            slug=slug, name=t.get("topic_name"),
        )

    for p in problem_nodes:
        problem_id = p.get("problem_id")
        if not problem_id:
            continue
        tx.run(
            """
            MERGE (p:OfflineProblem {problem_id: $problem_id})
            SET p.title = $title, p.title_slug = $title_slug
            """,
            problem_id=problem_id, title=p.get("title"), title_slug=p.get("title_slug"),
        )

    for src, tgt, weight in cooccurs_edges:
        tx.run(
            """
            MERGE (a:OfflineTopic {slug: $src})
            MERGE (b:OfflineTopic {slug: $tgt})
            MERGE (a)-[e:CO_OCCURS_OFFLINE]->(b)
            SET e.weight = $weight
            """,
            src=src, tgt=tgt, weight=weight,
        )
    for prereq, unlocks in prereq_edges:
        tx.run(
            """
            MERGE (a:OfflineTopic {slug: $prereq})
            MERGE (b:OfflineTopic {slug: $unlocks})
            MERGE (a)-[:PREREQ_OFFLINE]->(b)
            """,
            prereq=prereq, unlocks=unlocks,
        )
    for problem_id, topic in problem_topic_edges:
        tx.run(
            """
            MERGE (p:OfflineProblem {problem_id: $problem_id})
            MERGE (t:OfflineTopic {slug: $topic})
            MERGE (p)-[:HAS_TOPIC_OFFLINE]->(t)
            """,
            problem_id=problem_id, topic=topic,
        )


def clear_offline_graph() -> bool:
    """Deletes every :OfflineTopic and :OfflineProblem node (and their
    relationships) from the offline database -- for a clean re-run before
    write_offline_graph, so a re-run with fewer edges than before doesn't
    leave stale ones behind. Never touches :User/:Concept (the online
    graph) even if they happen to share a physical database, since it
    only matches the :OfflineTopic/:OfflineProblem labels."""
    driver = db_env.neo4j_driver()
    if driver is None:
        log.warning("Neo4j unavailable -- skipping offline graph clear.")
        return False
    try:
        with driver.session(database=db_env.NEO4J_OFFLINE_DATABASE) as session:
            session.run("MATCH (t:OfflineTopic) DETACH DELETE t")
            session.run("MATCH (p:OfflineProblem) DETACH DELETE p")
        return True
    finally:
        driver.close()


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def load_cooccurs_edges() -> list:
    """Returns [(source_slug, target_slug, weight), ...], or [] if Neo4j
    is unavailable or empty -- callers must fall back to the local JSON
    file in that case, not treat [] as "genuinely no co-occurrence data"."""
    driver = db_env.neo4j_driver()
    if driver is None:
        return []
    try:
        with driver.session(database=db_env.NEO4J_OFFLINE_DATABASE) as session:
            result = session.run(
                "MATCH (a:OfflineTopic)-[e:CO_OCCURS_OFFLINE]->(b:OfflineTopic) "
                "RETURN a.slug, b.slug, e.weight"
            )
            return [(r[0], r[1], float(r[2] or 0.0)) for r in result]
    except Exception as exc:
        log.warning("Failed to load CO_OCCURS_OFFLINE edges from Neo4j: %s", exc)
        return []
    finally:
        driver.close()


def load_prereq_edges() -> list:
    """Returns [(prereq_slug, unlocks_slug), ...], or [] if Neo4j is
    unavailable or empty."""
    driver = db_env.neo4j_driver()
    if driver is None:
        return []
    try:
        with driver.session(database=db_env.NEO4J_OFFLINE_DATABASE) as session:
            result = session.run(
                "MATCH (a:OfflineTopic)-[:PREREQ_OFFLINE]->(b:OfflineTopic) "
                "RETURN a.slug, b.slug"
            )
            return [(r[0], r[1]) for r in result]
    except Exception as exc:
        log.warning("Failed to load PREREQ_OFFLINE edges from Neo4j: %s", exc)
        return []
    finally:
        driver.close()


def load_problem_topics() -> dict:
    """Returns {title_slug: [topic_slug, ...]}, or {} if Neo4j is
    unavailable or empty."""
    driver = db_env.neo4j_driver()
    if driver is None:
        return {}
    try:
        with driver.session(database=db_env.NEO4J_OFFLINE_DATABASE) as session:
            result = session.run(
                "MATCH (p:OfflineProblem)-[:HAS_TOPIC_OFFLINE]->(t:OfflineTopic) "
                "RETURN p.title_slug, t.slug"
            )
            out: dict = {}
            for title_slug, topic in result:
                out.setdefault(title_slug, []).append(topic)
            return out
    except Exception as exc:
        log.warning("Failed to load HAS_TOPIC_OFFLINE edges from Neo4j: %s", exc)
        return {}
    finally:
        driver.close()

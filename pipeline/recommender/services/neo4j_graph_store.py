"""
Neo4j-backed durable UserGraph store.

Redis (see UserGraphService._from_cache/_to_cache) is a FAST cache with a
300s TTL -- it exists to avoid rebuilding a graph from Postgres on every
request within a session, but it is explicitly NOT durable: it expires,
and it's gone if Redis restarts.

This module adds a second, DURABLE tier: the user's graph is mirrored into
Neo4j as an actual graph (User/Concept/Problem nodes, typed relationships
carrying all the BKT/HLR/SM2 signal data), with no TTL. It survives past
the Redis TTL, past a server restart, past the end of any single user
session -- exactly what "even after the session ends" requires.

Three-tier read path, wired into UserGraphService.get():
    1. Redis (fast, 300s TTL)          -- hit: return immediately
    2. Neo4j (durable, no TTL)          -- hit: refresh Redis, return
    3. Rebuild from Postgres telemetry  -- always correct, slowest
       -> write through to BOTH Redis and Neo4j afterward

Neo4j is a MIRROR of Postgres telemetry, not a replacement for it --
Postgres (Submission, UserTopicMastery, etc.) remains the source of truth.
Neo4j exists purely so a graph doesn't need to be rebuilt from five
Postgres queries every time the Redis cache has expired.

Schema:
    (:User {user_id, username, elo_rating, cf_rating, lc_solved,
            total_xp, current_level, onboarding_complete,
            solved_ids, deprioritised_ids, exposed_ids_json, built_at})
    (:Concept {slug})
    (:Problem {problem_id})

    (:User)-[:PROBLEM_EDGE {edge_type, normalised_score, timestamp,
                            recency_weight}]->(:Problem)
    (:User)-[:CONCEPT_EDGE {edge_type, mastery_score, confidence, urgency,
                            half_life, severity, sm2_ef, sm2_interval,
                            next_review_date, last_attempted}]->(:Concept)
    (:Concept)-[:CC_EDGE {edge_type, weight}]->(:Concept)
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from pipeline.recommender.models.user_graph import (
    UserGraph, UserNode, ProblemEdge, ConceptEdge, ConceptConceptEdge, EdgeType,
)

log = logging.getLogger(__name__)


class Neo4jGraphStore:
    """
    Usage:
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
        store = Neo4jGraphStore(driver)
        store.save(graph)
        graph = store.load(user_id)   # None if never saved
        store.delete(user_id)          # e.g. account deletion / GDPR request

    Pass `driver=None` to disable Neo4j entirely (same pattern as redis=None
    disabling the Redis cache) -- every method becomes a no-op returning
    None, so callers never need to branch on whether Neo4j is configured.
    """

    def __init__(self, driver=None, database: str = "neo4j"):
        self._driver = driver
        self._database = database

    @property
    def enabled(self) -> bool:
        return self._driver is not None

    # ------------------------------------------------------------- save

    def save(self, graph: UserGraph) -> None:
        """
        Write-through: MERGE every node and relationship so repeated saves
        for the same user UPDATE existing data rather than duplicating it.
        Safe to call after every rebuild -- idempotent.
        """
        if not self.enabled:
            return
        try:
            with self._driver.session(database=self._database) as session:
                session.execute_write(self._save_tx, graph)
        except Exception as exc:
            log.warning("Neo4j save failed for user %s: %s", graph.user.user_id, exc)

    @staticmethod
    def _save_tx(tx, graph: UserGraph) -> None:
        u = graph.user
        tx.run(
            """
            MERGE (u:User {user_id: $user_id})
            SET u.username           = $username,
                u.elo_rating         = $elo_rating,
                u.cf_rating          = $cf_rating,
                u.lc_solved          = $lc_solved,
                u.total_xp           = $total_xp,
                u.current_level      = $current_level,
                u.onboarding_complete = $onboarding_complete,
                u.solved_ids         = $solved_ids,
                u.deprioritised_ids  = $deprioritised_ids,
                u.exposed_ids_json   = $exposed_ids_json,
                u.built_at           = $built_at
            """,
            user_id=u.user_id, username=u.username,
            elo_rating=u.elo_rating, cf_rating=u.cf_rating,
            lc_solved=u.lc_solved, total_xp=u.total_xp,
            current_level=u.current_level,
            onboarding_complete=u.onboarding_complete,
            solved_ids=list(graph.solved_ids),
            deprioritised_ids=list(graph.deprioritised_ids),
            exposed_ids_json=json.dumps(graph.exposed_ids),
            built_at=graph.built_at,
        )

        for pid, edge in graph.problem_edges.items():
            tx.run(
                """
                MATCH (u:User {user_id: $user_id})
                MERGE (p:Problem {problem_id: $problem_id})
                MERGE (u)-[e:PROBLEM_EDGE]->(p)
                SET e.edge_type        = $edge_type,
                    e.normalised_score = $normalised_score,
                    e.timestamp        = $timestamp,
                    e.recency_weight   = $recency_weight
                """,
                user_id=u.user_id, problem_id=pid,
                edge_type=edge.edge_type.value,
                normalised_score=edge.normalised_score,
                timestamp=edge.timestamp,
                recency_weight=edge.recency_weight,
            )

        for slug, edge in graph.concept_edges.items():
            tx.run(
                """
                MATCH (u:User {user_id: $user_id})
                MERGE (c:Concept {slug: $slug})
                MERGE (u)-[e:CONCEPT_EDGE]->(c)
                SET e.edge_type         = $edge_type,
                    e.mastery_score     = $mastery_score,
                    e.confidence        = $confidence,
                    e.urgency           = $urgency,
                    e.half_life         = $half_life,
                    e.severity          = $severity,
                    e.sm2_ef            = $sm2_ef,
                    e.sm2_interval      = $sm2_interval,
                    e.next_review_date  = $next_review_date,
                    e.last_attempted    = $last_attempted
                """,
                user_id=u.user_id, slug=slug,
                edge_type=edge.edge_type.value,
                mastery_score=edge.mastery_score,
                confidence=edge.confidence,
                urgency=edge.urgency,
                half_life=edge.half_life,
                severity=edge.severity,
                sm2_ef=edge.sm2_ef,
                sm2_interval=edge.sm2_interval,
                next_review_date=edge.next_review_date,
                last_attempted=edge.last_attempted,
            )

        for src, edges in graph.cc_edges.items():
            for cc in edges:
                tx.run(
                    """
                    MERGE (a:Concept {slug: $src})
                    MERGE (b:Concept {slug: $tgt})
                    MERGE (a)-[e:CC_EDGE {edge_type: $edge_type}]->(b)
                    SET e.weight = $weight
                    """,
                    src=cc.source_slug, tgt=cc.target_slug,
                    edge_type=cc.edge_type.value, weight=cc.weight,
                )

    # ------------------------------------------------------------- load

    def load(self, user_id: str) -> Optional[UserGraph]:
        """Returns None if this user has never been saved to Neo4j."""
        if not self.enabled:
            return None
        try:
            with self._driver.session(database=self._database) as session:
                return session.execute_read(self._load_tx, user_id)
        except Exception as exc:
            log.warning("Neo4j load failed for user %s: %s", user_id, exc)
            return None

    @staticmethod
    def _load_tx(tx, user_id: str) -> Optional[UserGraph]:
        user_record = tx.run(
            "MATCH (u:User {user_id: $user_id}) RETURN u LIMIT 1",
            user_id=user_id,
        ).single()
        if user_record is None:
            return None
        u = user_record["u"]

        graph = UserGraph(user=UserNode(
            user_id=u["user_id"], username=u.get("username", ""),
            elo_rating=u.get("elo_rating", 1200), cf_rating=u.get("cf_rating", 0),
            lc_solved=u.get("lc_solved", 0), total_xp=u.get("total_xp", 0),
            current_level=u.get("current_level", 1),
            onboarding_complete=u.get("onboarding_complete", False),
        ))
        graph.solved_ids = set(u.get("solved_ids") or [])
        graph.deprioritised_ids = set(u.get("deprioritised_ids") or [])
        graph.exposed_ids = json.loads(u.get("exposed_ids_json") or "{}")
        graph.built_at = u.get("built_at", graph.built_at)

        for rec in tx.run(
            """
            MATCH (u:User {user_id: $user_id})-[e:PROBLEM_EDGE]->(p:Problem)
            RETURN p.problem_id AS pid, e
            """,
            user_id=user_id,
        ):
            e = rec["e"]
            graph.problem_edges[rec["pid"]] = ProblemEdge(
                problem_id=rec["pid"],
                edge_type=EdgeType(e["edge_type"]),
                normalised_score=e.get("normalised_score", 0.0),
                timestamp=e.get("timestamp"),
                recency_weight=e.get("recency_weight", 1.0),
            )

        for rec in tx.run(
            """
            MATCH (u:User {user_id: $user_id})-[e:CONCEPT_EDGE]->(c:Concept)
            RETURN c.slug AS slug, e
            """,
            user_id=user_id,
        ):
            e = rec["e"]
            graph.concept_edges[rec["slug"]] = ConceptEdge(
                concept_slug=rec["slug"],
                edge_type=EdgeType(e["edge_type"]),
                mastery_score=e.get("mastery_score", 0.0),
                confidence=e.get("confidence", 0.0),
                urgency=e.get("urgency", 0.0),
                half_life=e.get("half_life", 1.0),
                severity=e.get("severity", 0.0),
                sm2_ef=e.get("sm2_ef", 2.5),
                sm2_interval=e.get("sm2_interval", 1),
                next_review_date=e.get("next_review_date"),
                last_attempted=e.get("last_attempted"),
            )

        # cc_edges are concept<->concept, not user-scoped -- load the
        # subgraph reachable from this user's touched concepts plus one
        # hop out (covers prereqs of untouched target concepts too, same
        # fix as UserGraphService._load_cc_edges).
        for rec in tx.run(
            """
            MATCH (a:Concept)-[e:CC_EDGE]->(b:Concept)
            RETURN a.slug AS src, b.slug AS tgt, e
            """
        ):
            e = rec["e"]
            graph.add_cc_edge(ConceptConceptEdge(
                source_slug=rec["src"], target_slug=rec["tgt"],
                edge_type=EdgeType(e["edge_type"]), weight=e.get("weight", 1.0),
            ))

        return graph

    # ----------------------------------------------------------- delete

    def delete(self, user_id: str) -> None:
        """Removes this user's node and every relationship touching it (not shared Concept/Problem nodes)."""
        if not self.enabled:
            return
        try:
            with self._driver.session(database=self._database) as session:
                session.execute_write(self._delete_tx, user_id)
        except Exception as exc:
            log.warning("Neo4j delete failed for user %s: %s", user_id, exc)

    @staticmethod
    def _delete_tx(tx, user_id: str) -> None:
        tx.run(
            "MATCH (u:User {user_id: $user_id}) DETACH DELETE u",
            user_id=user_id,
        )

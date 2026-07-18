"""
pipeline/recommender/services/user_graph_service.py
=====================================================
Builds a UserGraph from live backend schema telemetry (Postgres) and
the offline concept structure (loaded once at startup).

This is the "M" layer's persistence bridge.  It translates raw DB rows
into typed UserGraph edges.  The controller calls this; the model knows
nothing about the DB.

Offline graph (loaded at startup -- see main.py's startup hook, which
calls load_offline_concept_graph() below; before that hook existed this
was defined but never actually invoked in production, so _PREREQ_CACHE
stayed permanently empty and every user's cc_edges was always [] --
prerequisite gating (UserGraph.is_locked) and co-occurrence-based
exploration (NoveltyPool) were silent no-ops regardless of what data
existed):
    - concept_graph: dict[slug -> list[ConceptConceptEdge]]
      Neo4j FIRST (pipeline/graphs/neo4j_offline_writer.py), falling back
      to question-graph/data/topic_topic_edges.json + Postgres
      TopicPrerequisite table if Neo4j is unavailable.

Online telemetry (loaded per-user on request):
    - Submission         → ProblemEdge (SOLVED / ATTEMPTED)
    - RecommendationLog  → ProblemEdge (EXPOSED / SKIPPED)
    - UserTopicMastery   → ConceptEdge (MASTERED / LEARNING)
    - ConceptGapProfile  → ConceptEdge (WEAK, merges into above)
    - BKT state          → ConceptEdge.mastery_score
    - HLR state          → ConceptEdge.urgency

Redis cache:
    Key   : "user_graph:{user_id}"
    TTL   : 300 seconds (5 min)
    Value : JSON of UserGraph.to_dict()
    Invalidated immediately on POST /user-state/update
"""

from __future__ import annotations

import json
import math
import time
import logging
from pathlib import Path
from typing import Optional

from pipeline.recommender.models.user_graph import (
    UserGraph, UserNode, ProblemEdge, ConceptEdge,
    ConceptConceptEdge, EdgeType,
)
from database.postgres.db import (
    get_connection, release_connection, _topic_id_to_ml_slug,
)

log = logging.getLogger(__name__)

# Path to offline graph — relative to project root
_TOPIC_TOPIC_JSON = Path("question-graph") / "data" / "topic_topic_edges.json"
_PREREQ_CACHE: dict[str, list[ConceptConceptEdge]] = {}   # loaded once


# ---------------------------------------------------------------------------
# Offline graph loader (called at FastAPI startup)
# ---------------------------------------------------------------------------

def load_offline_concept_graph(db=None) -> dict[str, list[ConceptConceptEdge]]:
    """
    Load concept<->concept edges. Neo4j FIRST (pipeline/graphs/
    neo4j_offline_writer.py's :OfflineTopic graph, the shared,
    centrally-updated copy any pipeline run can push to), falling back to
    the local JSON file + Postgres if Neo4j is unavailable or empty --
    same graceful-degrade convention as every other Neo4j touchpoint in
    this repo (see neo4j_offline_writer.py's module docstring for the
    full separation-from-the-online-graph rationale):
      1. Neo4j :OfflineTopic CO_OCCURS_OFFLINE / PREREQ_OFFLINE edges.
      2. Fallback: question-graph/data/topic_topic_edges.json (CO_OCCURS)
         + Postgres TopicPrerequisite table (PREREQ).
    Returns {source_slug: [ConceptConceptEdge, ...]}
    """
    global _PREREQ_CACHE
    cc: dict[str, list[ConceptConceptEdge]] = {}

    from pipeline.graphs.neo4j_offline_writer import load_cooccurs_edges, load_prereq_edges

    neo4j_cooccurs = load_cooccurs_edges()
    neo4j_prereq = load_prereq_edges()

    if neo4j_cooccurs:
        for src, tgt, weight in neo4j_cooccurs:
            cc.setdefault(src, []).append(ConceptConceptEdge(
                source_slug=src, target_slug=tgt,
                edge_type=EdgeType.COOCCURS, weight=weight,
            ))
        log.info("Loaded %d CO_OCCURS concept edges from Neo4j", len(neo4j_cooccurs))
    elif _TOPIC_TOPIC_JSON.exists():
        try:
            raw = json.loads(_TOPIC_TOPIC_JSON.read_text(encoding="utf-8-sig"))
            for e in raw:
                src = str(e.get("source") or e.get("source_topic_id") or "")
                tgt = str(e.get("target") or e.get("target_topic_id") or "")
                if not src or not tgt:
                    continue
                shared = int(e.get("shared_problem_count", 0))
                if shared < 2:          # drop single-problem noise
                    continue
                edge = ConceptConceptEdge(
                    source_slug=src,
                    target_slug=tgt,
                    edge_type=EdgeType.COOCCURS,
                    weight=float(e.get("jaccard", 0.0)),
                )
                cc.setdefault(src, []).append(edge)
            log.info("Neo4j unavailable/empty -- loaded %d CO_OCCURS concept "
                     "edges from local JSON instead",
                     sum(len(v) for v in cc.values()))
        except Exception as exc:
            log.warning("Failed to load topic_topic_edges.json: %s", exc)

    if neo4j_prereq:
        for prereq, unlocks in neo4j_prereq:
            cc.setdefault(prereq, []).append(ConceptConceptEdge(
                source_slug=prereq, target_slug=unlocks,
                edge_type=EdgeType.PREREQ, weight=1.0,
            ))
        log.info("Loaded %d PREREQ edges from Neo4j", len(neo4j_prereq))
    elif db is not None:
        try:
            rows = db.execute(
                "SELECT topic_id, prerequisite_id FROM TopicPrerequisite"
            ).fetchall()
            for row in rows:
                src = str(row[1])   # prerequisite is the source (must come first)
                tgt = str(row[0])   # topic is unlocked after
                edge = ConceptConceptEdge(
                    source_slug=src, target_slug=tgt,
                    edge_type=EdgeType.PREREQ, weight=1.0,
                )
                cc.setdefault(src, []).append(edge)
            log.info("Neo4j unavailable/empty -- loaded %d PREREQ edges from "
                     "Postgres instead", len(rows))
        except Exception as exc:
            log.warning("Failed to load TopicPrerequisite from Postgres: %s", exc)

    _PREREQ_CACHE = cc
    return cc


# ---------------------------------------------------------------------------
# Per-user graph builder
# ---------------------------------------------------------------------------

class UserGraphService:
    """
    Builds and caches UserGraph instances.

    Three-tier read path in get():
        1. Redis (fast, 300s TTL)
        2. Neo4j (durable, no TTL -- survives past Redis expiry and past
           the end of any session, per the "store even after session ends"
           requirement)
        3. Rebuild from Postgres telemetry (always correct, slowest) --
           writes through to BOTH Redis and Neo4j afterward so the next
           read anywhere hits a fast/durable tier instead of rebuilding.

    Constructor args:
        db      : SQLAlchemy session (or any connection with .execute())
        redis   : redis.Redis client (or None to skip the fast cache)
        neo4j   : Neo4jGraphStore instance (or None to skip durable storage)
        bkt     : Shraddha's BKT store dict {user_id: {topic: P(L)}}
        hlr     : Shraddha's HLR store dict {user_id: {topic: urgency}}
    """

    CACHE_TTL = 300   # seconds -- Redis tier only; Neo4j has no TTL

    def __init__(self, db, redis=None, neo4j=None, bkt: dict = None, hlr: dict = None):
        self._db    = db
        self._redis = redis
        self._neo4j = neo4j
        self._bkt   = bkt or {}
        self._hlr   = hlr or {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get(self, user_id: str) -> UserGraph:
        """Return a UserGraph for user_id. Redis -> Neo4j -> Postgres rebuild, in that order."""
        cached = self._from_cache(user_id)
        if cached is not None:
            return cached

        if self._neo4j is not None:
            durable = self._neo4j.load(user_id)
            if durable is not None:
                self._to_cache(user_id, durable)   # refresh the fast tier
                return durable

        graph = self._build(user_id)
        self._to_cache(user_id, graph)
        if self._neo4j is not None:
            self._neo4j.save(graph)
        return graph

    def new_user_graph(self, user_id: str, username: str = "") -> UserGraph:
        """
        Explicit new-user path -- matches the "no pre-req -> candidate
        generation (new user)" branch of the architecture diagram: a single
        user node with base ELO/SM2/BKT defaults and no concept/problem
        edges, no DB round trip needed since a brand new signup has nothing
        in Submission/UserTopicMastery/RecommendationLog/ConceptGapProfile
        yet anyway.

        Call this right after signup (before the first submission exists)
        instead of get(), which would otherwise issue five queries that are
        guaranteed to return zero rows for a user with no history. Once the
        user has any telemetry, use get()/invalidate() as normal -- this
        method is only for the very first graph a user ever gets.
        """
        graph = UserGraph(user=UserNode(user_id=user_id, username=username))
        self._to_cache(user_id, graph)
        if self._neo4j is not None:
            self._neo4j.save(graph)
        return graph

    def invalidate(self, user_id: str) -> None:
        """
        Call this immediately after a new submission is processed, BEFORE
        persist() below -- clears the Redis (fast) tier first so a partial
        persist() failure never leaves a stale Redis entry behind.

        Does NOT touch Neo4j -- persist() (or get()'s own rebuild path)
        is what writes the fresh copy there, using MERGE so Neo4j never
        goes empty in between.
        """
        if self._redis:
            try:
                self._redis.delete(f"user_graph:{user_id}")
            except Exception as exc:
                log.warning("Redis invalidate failed: %s", exc)

    def persist(self, user_id: str, graph: UserGraph) -> None:
        """
        Write-through to BOTH tiers for an already-mutated graph.

        get()/new_user_graph() write-through internally after a rebuild --
        but a caller like StateUpdateService.process_submission() mutates
        its OWN already-fetched graph object directly (adds a ProblemEdge,
        updates ConceptEdges from fresh BKT/HLR output) without going
        through get()'s rebuild path at all. Without this method, such a
        caller could only reach the "private" _to_cache() (Redis only),
        silently never writing the update to Neo4j -- meaning the fresh
        mastery/problem data would vanish the moment the 5-minute Redis
        entry expired, and the next read would fall through to Neo4j and
        find the STALE pre-submission graph. persist() is the one call
        that keeps both tiers correctly in sync after a direct mutation.
        """
        self._to_cache(user_id, graph)
        if self._neo4j is not None:
            self._neo4j.save(graph)

    def delete_user(self, user_id: str) -> None:
        """Removes the user's graph from every tier -- Redis, Neo4j, and (implicitly) any future rebuild is unaffected since Postgres rows are untouched here."""
        if self._redis:
            try:
                self._redis.delete(f"user_graph:{user_id}")
            except Exception as exc:
                log.warning("Redis delete failed: %s", exc)
        if self._neo4j is not None:
            self._neo4j.delete(user_id)

    # ------------------------------------------------------------------
    # Build from Postgres telemetry
    # ------------------------------------------------------------------

    def _build(self, user_id: str) -> UserGraph:
        user_row = self._fetch_user(user_id)
        if user_row is None:
            raise ValueError(f"User {user_id!r} not found")

        graph = UserGraph(user=user_row)

        self._load_submissions(graph, user_id)
        self._load_recommendation_log(graph, user_id)
        self._load_topic_mastery(graph, user_id)
        self._load_concept_gaps(graph, user_id)
        self._load_cc_edges(graph)

        return graph

    def _fetch_user(self, user_id: str) -> Optional[UserNode]:
        # Table/column names match the REAL deployed schema, not the
        # PascalCase Prisma-style names this query previously assumed
        # (which don't exist in the actual database -- confirmed directly
        # against a live instance of this backend project: "user" is
        # lowercase and needs quoting since it's a reserved word; its
        # primary key column is `id`, not `user_id`; `name`, not
        # `username`; `onboarding_completed`, not `onboarding_complete`).
        try:
            row = self._db.execute(
                """
                SELECT u.id, u.name, u.onboarding_completed,
                       x.total_xp, x.current_level
                FROM   "user"  u
                LEFT JOIN user_xp x ON x.user_id = u.id
                WHERE  u.id = :uid
                """,
                {"uid": user_id},
            ).fetchone()
            if row is None:
                return None
            return UserNode(
                user_id=str(row[0]),
                username=str(row[1] or ""),
                onboarding_complete=bool(row[2]),
                total_xp=int(row[3] or 0),
                current_level=int(row[4] or 1),
            )
        except Exception as exc:
            log.error("Failed to fetch user %s: %s", user_id, exc)
            return None

    def _load_submissions(self, graph: UserGraph, user_id: str) -> None:
        """Load Submission rows → ProblemEdge (SOLVED / ATTEMPTED)."""
        try:
            rows = self._db.execute(
                """
                SELECT problem_id, verdict, normalised_score,
                       hints_used, submission_count, submitted_at
                FROM   submission
                WHERE  user_id = :uid AND status = 'COMPLETED'
                ORDER  BY submitted_at DESC
                LIMIT  500
                """,
                {"uid": user_id},
            ).fetchall()
        except Exception as exc:
            log.warning("Submission fetch failed: %s", exc)
            return

        for row in rows:
            pid, verdict, score, hints, attempts, ts = row
            edge_type = EdgeType.SOLVED if verdict == "Accepted" else EdgeType.ATTEMPTED
            ts_float  = ts.timestamp() if hasattr(ts, "timestamp") else float(ts or 0)
            days      = (time.time() - ts_float) / 86400
            decay     = math.exp(-0.05 * days)
            edge = ProblemEdge(
                problem_id=str(pid),
                edge_type=edge_type,
                normalised_score=float(score or 0) / 100.0,  # DB stores 0-100
                hints_used=int(hints or 0),
                attempt_count=int(attempts or 1),
                timestamp=ts_float,
                recency_weight=decay,
            )
            graph.add_problem_edge(edge)

    def _load_recommendation_log(self, graph: UserGraph, user_id: str) -> None:
        """Load RecommendationLog → ProblemEdge (EXPOSED / SKIPPED)."""
        try:
            rows = self._db.execute(
                """
                SELECT problem_id, recommended_at, was_skipped, skip_count
                FROM   recommendation_log
                WHERE  user_id = :uid
                ORDER  BY recommended_at DESC
                LIMIT  300
                """,
                {"uid": user_id},
            ).fetchall()
        except Exception as exc:
            log.warning("RecommendationLog fetch failed: %s", exc)
            return

        for row in rows:
            pid, rec_at, skipped, skip_count = row
            ts = rec_at.timestamp() if hasattr(rec_at, "timestamp") else float(rec_at or 0)
            # only add EXPOSED/SKIPPED if not already a SOLVED edge
            if pid in graph.solved_ids:
                continue
            edge_type = EdgeType.SKIPPED if (skipped and int(skip_count or 0) >= 3) \
                        else EdgeType.EXPOSED
            edge = ProblemEdge(
                problem_id=str(pid),
                edge_type=edge_type,
                timestamp=ts,
            )
            graph.add_problem_edge(edge)

    def _load_topic_mastery(self, graph: UserGraph, user_id: str) -> None:
        """
        Load UserTopicMastery → ConceptEdge. Merges BKT + HLR state.

        user_topic_mastery.topic_id is a FK to the backend's topic.id (an
        opaque generated id) -- it is NOT an ML-pipeline slug. Every
        downstream consumer of ConceptEdge.concept_slug (CoursePathPool's
        target_concepts, mastered_concepts(), Qdrant topic_tags matching,
        etc.) expects the ML slug ("array"), so each row's topic_id is
        translated via database.postgres.topic_taxonomy before use here --
        same translation get_user_mastery()/get_user_hlr() apply. Without
        this, concept_slug was literally the opaque topic.id string, which
        never matches any real Qdrant topic_tags value: every pool that
        targets "concepts the user knows/is learning" silently found zero
        candidates for any user with real seeded mastery, even though
        graph.concept_edges looked non-empty. Rows for a topic_id with no
        ML-slug counterpart (topic_taxonomy.UNMAPPED_TOPIC_SLUGS) are
        skipped, not guessed.
        """
        try:
            rows = self._db.execute(
                """
                SELECT topic_id, mastery_score, confidence,
                       attempt_count, problems_solved, last_attempted,
                       sm2_ef, sm2_interval, next_review_date
                FROM   user_topic_mastery
                WHERE  user_id = :uid
                """,
                {"uid": user_id},
            ).fetchall()
        except Exception as exc:
            log.warning("UserTopicMastery fetch failed: %s", exc)
            return

        bkt_user = self._bkt.get(user_id, {})
        hlr_user = self._hlr.get(user_id, {})

        topic_conn = get_connection()
        try:
            for row in rows:
                (raw_topic_id, mastery_score, confidence, attempt_count,
                 problems_solved, last_attempted, sm2_ef, sm2_interval,
                 next_review_date) = row

                topic_id = _topic_id_to_ml_slug(topic_conn, str(raw_topic_id))
                if topic_id is None:
                    continue
                self._add_topic_mastery_edge(
                    graph, topic_id, mastery_score, confidence, last_attempted,
                    sm2_ef, sm2_interval, next_review_date, bkt_user, hlr_user,
                )
        finally:
            release_connection(topic_conn)

    def _add_topic_mastery_edge(self, graph, topic_id, mastery_score, confidence,
                                last_attempted, sm2_ef, sm2_interval,
                                next_review_date, bkt_user, hlr_user) -> None:
        # BKT P(L) from Shraddha's online store takes precedence.
        # user_topic_mastery.mastery_score is ALREADY 0-1 in the real
        # schema (confirmed against a live instance of this backend
        # project) -- the /100.0 here previously assumed a 0-100 scale
        # that doesn't exist, silently shrinking every loaded mastery
        # value to ~1/100th of its real value.
        bkt_mastery = bkt_user.get(topic_id)
        final_mastery = float(bkt_mastery) if bkt_mastery is not None \
                        else float(mastery_score or 0)

        hlr_topic = hlr_user.get(topic_id, {})
        hlr_urgency = 0.0
        hlr_half_life = 1.0

        if hlr_topic:
            from pipeline.recommender.hlr import calculate_urgency
            import time as _t
            hlr_urgency   = calculate_urgency(hlr_topic, _t.time())
            hlr_half_life = float(hlr_topic.get("half_life", 1.0))

        # confidence = 1 - CV(last 5 performance scores).
        # HLR stores performance on last attempt; schema stores confidence
        # as low/medium/high string — map to float for the edge.
        conf_raw = str(confidence or "medium")
        conf_map = {"low": 0.33, "medium": 0.66, "high": 1.0}
        final_confidence = conf_map.get(conf_raw.lower(), 0.66)

        ts_last = None
        if last_attempted:
            ts_last = last_attempted.timestamp() \
                if hasattr(last_attempted, "timestamp") \
                else float(last_attempted)

        nrd_str = str(next_review_date) if next_review_date else None

        if final_mastery >= 0.7:
            etype = EdgeType.MASTERED
        elif final_mastery > 0.0:
            etype = EdgeType.LEARNING
        else:
            return

        edge = ConceptEdge(
            concept_slug=str(topic_id),
            edge_type=etype,
            mastery_score=final_mastery,
            confidence=final_confidence,
            half_life=hlr_half_life,
            urgency=float(hlr_urgency),
            sm2_ef=float(sm2_ef or 2.5),
            sm2_interval=int(sm2_interval or 1),
            next_review_date=nrd_str,
            last_attempted=ts_last,
        )
        graph.add_concept_edge(edge)

    def _load_concept_gaps(self, graph: UserGraph, user_id: str) -> None:
        """Load ConceptGapProfile → augments ConceptEdge with severity."""
        try:
            rows = self._db.execute(
                """
                SELECT gap_name, severity
                FROM   concept_gap_profile
                WHERE  user_id = :uid AND severity > 0.3
                """,
                {"uid": user_id},
            ).fetchall()
        except Exception as exc:
            log.warning("ConceptGapProfile fetch failed: %s", exc)
            return

        for row in rows:
            gap_name, severity = row
            slug = str(gap_name)
            # gap_name may be a skill tag, not a topic slug —
            # we store it as a WEAK concept edge either way
            existing = graph.concept_edges.get(slug)
            if existing:
                existing.severity = max(existing.severity, float(severity))
                if float(severity) >= 0.6:
                    existing.edge_type = EdgeType.WEAK
            else:
                graph.add_concept_edge(ConceptEdge(
                    concept_slug=slug,
                    edge_type=EdgeType.WEAK,
                    severity=float(severity),
                ))

    def _load_cc_edges(self, graph: UserGraph) -> None:
        """
        Attach the FULL offline concept-concept (prerequisite/cooccurrence)
        graph -- not just edges sourced from concepts the user has already
        touched.

        The previous version only loaded edges where source_slug was
        already a key in graph.concept_edges. But is_locked() needs to
        look up prerequisites for CANDIDATE target concepts, which the
        user has, by definition, often never touched yet -- that is
        exactly what "locked" means. Filtering by user-touched concepts
        silently dropped precisely the edges prerequisite gating depends
        on, letting locked candidates pass filtering as if unlocked.

        _PREREQ_CACHE is the full offline prerequisite graph (~465
        concepts), already loaded into memory once at process startup --
        copying all of it into every user's graph costs nothing extra
        and is the only way to guarantee is_locked() has complete data
        for any candidate, not just ones the user happens to have a
        ConceptEdge for already.
        """
        for edges in _PREREQ_CACHE.values():
            for cc_edge in edges:
                graph.add_cc_edge(cc_edge)

    # ------------------------------------------------------------------
    # Redis cache
    # ------------------------------------------------------------------

    def _from_cache(self, user_id: str) -> Optional[UserGraph]:
        if self._redis is None:
            return None
        try:
            raw = self._redis.get(f"user_graph:{user_id}")
            if raw:
                return UserGraph.from_dict(json.loads(raw))
        except Exception as exc:
            log.warning("Redis get failed: %s", exc)
        return None

    def _to_cache(self, user_id: str, graph: UserGraph) -> None:
        if self._redis is None:
            return
        try:
            self._redis.setex(
                f"user_graph:{user_id}",
                self.CACHE_TTL,
                json.dumps(graph.to_dict()),
            )
        except Exception as exc:
            log.warning("Redis set failed: %s", exc)
"""
pipeline/recommender/services/user_graph_service.py
=====================================================
Builds a UserGraph from live backend schema telemetry (Postgres) and
the offline concept structure (loaded once at startup).

This is the "M" layer's persistence bridge.  It translates raw DB rows
into typed UserGraph edges.  The controller calls this; the model knows
nothing about the DB.

Offline graph (loaded at startup):
    - concept_graph: dict[slug -> list[ConceptConceptEdge]]
      Loaded from question-graph/data/topic_topic_edges.json +
      Postgres TopicPrerequisite table.

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

log = logging.getLogger(__name__)

# Path to offline graph — relative to project root
_TOPIC_TOPIC_JSON = Path("question-graph") / "data" / "topic_topic_edges.json"
_PREREQ_CACHE: dict[str, list[ConceptConceptEdge]] = {}   # loaded once


# ---------------------------------------------------------------------------
# Offline graph loader (called at FastAPI startup)
# ---------------------------------------------------------------------------

def load_offline_concept_graph(db=None) -> dict[str, list[ConceptConceptEdge]]:
    """
    Load concept<->concept edges from:
      1. question-graph/data/topic_topic_edges.json  (jaccard CO_OCCURS edges)
      2. Postgres TopicPrerequisite table             (PREREQ edges)
    Returns {source_slug: [ConceptConceptEdge, ...]}
    """
    global _PREREQ_CACHE
    cc: dict[str, list[ConceptConceptEdge]] = {}

    # -- CO_OCCURS from normalized JSON --
    if _TOPIC_TOPIC_JSON.exists():
        try:
            raw = json.loads(_TOPIC_TOPIC_JSON.read_text(encoding="utf-8-sig"))
            for e in raw:
                src = str(e.get("source") or "")
                tgt = str(e.get("target") or "")
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
            log.info("Loaded %d CO_OCCURS concept edges from JSON",
                     sum(len(v) for v in cc.values()))
        except Exception as exc:
            log.warning("Failed to load topic_topic_edges.json: %s", exc)

    # -- PREREQ from Postgres TopicPrerequisite --
    if db is not None:
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
            log.info("Loaded %d PREREQ edges from Postgres", len(rows))
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

    Constructor args:
        db      : SQLAlchemy session (or any connection with .execute())
        redis   : redis.Redis client (or None to skip caching)
        bkt     : Shraddha's BKT store dict {user_id: {topic: P(L)}}
        hlr     : Shraddha's HLR store dict {user_id: {topic: urgency}}
    """

    CACHE_TTL = 300   # seconds

    def __init__(self, db, redis=None, bkt: dict = None, hlr: dict = None):
        self._db    = db
        self._redis = redis
        self._bkt   = bkt or {}
        self._hlr   = hlr or {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get(self, user_id: str) -> UserGraph:
        """Return a UserGraph for user_id.  Hits Redis cache first."""
        cached = self._from_cache(user_id)
        if cached is not None:
            return cached
        graph = self._build(user_id)
        self._to_cache(user_id, graph)
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
        return graph

    def invalidate(self, user_id: str) -> None:
        """Call this immediately after a new submission is processed."""
        if self._redis:
            try:
                self._redis.delete(f"user_graph:{user_id}")
            except Exception as exc:
                log.warning("Redis invalidate failed: %s", exc)

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
        try:
            row = self._db.execute(
                """
                SELECT u.user_id, u.username, u.onboarding_complete,
                       x.total_xp, x.current_level
                FROM   "User"  u
                LEFT JOIN "UserXP" x ON x.user_id = u.user_id
                WHERE  u.user_id = :uid
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
                FROM   "Submission"
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
                FROM   "RecommendationLog"
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
        """Load UserTopicMastery → ConceptEdge.  Merges BKT + HLR state."""
        try:
            rows = self._db.execute(
                """
                SELECT topic_id, mastery_score, confidence,
                       attempt_count, problems_solved, last_attempted,
                       sm2_ef, sm2_interval, next_review_date
                FROM   "UserTopicMastery"
                WHERE  user_id = :uid
                """,
                {"uid": user_id},
            ).fetchall()
        except Exception as exc:
            log.warning("UserTopicMastery fetch failed: %s", exc)
            return

        bkt_user = self._bkt.get(user_id, {})
        hlr_user = self._hlr.get(user_id, {})

        for row in rows:
            (topic_id, mastery_score, confidence, attempt_count,
             problems_solved, last_attempted, sm2_ef, sm2_interval,
             next_review_date) = row

            # BKT P(L) from Shraddha's online store takes precedence
            bkt_mastery = bkt_user.get(topic_id)
            final_mastery = float(bkt_mastery) if bkt_mastery is not None \
                            else float(mastery_score or 0) / 100.0

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
                continue

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
                FROM   "ConceptGapProfile"
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
        Attach offline concept-concept edges for concepts the user
        has interacted with.  Keeps the graph small (only user-relevant
        subgraph, not the full 465-concept graph).
        """
        for slug in list(graph.concept_edges.keys()):
            for cc_edge in _PREREQ_CACHE.get(slug, []):
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
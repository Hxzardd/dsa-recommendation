"""
pipeline/recommender/models/user_graph.py
==========================================
The user graph: a single user node with typed edges to concept nodes
and problem nodes.  Backed by the backend schema telemetry tables.

Design decisions:
  - User is the ONLY node of its type (single user context per graph instance).
  - Concept nodes come from the topic/pattern/skill taxonomy.
  - Problem nodes are references to Qdrant point IDs (problem_id strings).
  - All edges are directional from the user outward, plus concept<->concept
    co-occurrence / prereq edges imported from the offline graph.
  - The graph is ephemeral in memory; Redis caches the serialised form
    keyed by user_id with a TTL (rebuilt on every new submission).

Telemetry sources (what feeds each edge type):
  SOLVED     <- Submission (verdict=Accepted, normalised_score, submitted_at)
  ATTEMPTED  <- Submission (any verdict, attempt count, hints_used)
  MASTERED   <- UserTopicMastery (mastery_score, sm2_ef, next_review_date)
  WEAK       <- ConceptGapProfile (gap_name maps to concept slug, severity)
  EXPOSED    <- RecommendationLog (recommended_at, was_skipped, skip_count)
  PREREQ     <- TopicPrerequisite (offline, loaded once at startup)
  COOCCURS   <- question-graph/data/topic_topic_edges.json (offline jaccard)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Edge types
# ---------------------------------------------------------------------------

class EdgeType(str, Enum):
    # user → problem
    SOLVED    = "solved"      # accepted submission, carries normalised_score
    ATTEMPTED = "attempted"   # any submission outcome (including failures)
    EXPOSED   = "exposed"     # rec engine showed this problem to the user
    SKIPPED   = "skipped"     # user dismissed without attempting

    # user → concept
    MASTERED  = "mastered"    # UserTopicMastery mastery_score ≥ 70
    LEARNING  = "learning"    # mastery_score 30–69 (in progress)
    WEAK      = "weak"        # ConceptGapProfile severity > 0.6

    # concept → concept  (offline, from question-graph)
    PREREQ    = "prereq"      # TopicPrerequisite
    COOCCURS  = "cooccurs"    # topic_topic_edges jaccard-weighted


# ---------------------------------------------------------------------------
# Edge payloads
# ---------------------------------------------------------------------------

@dataclass
class ProblemEdge:
    """User → Problem edge."""
    problem_id:       str
    edge_type:        EdgeType
    normalised_score: float = 0.0     # 0–1 from backend Submission.normalised_score
    hints_used:       int   = 0
    attempt_count:    int   = 1
    timestamp:        float = field(default_factory=time.time)
    # recency weight: e^(-lambda * days_since), computed at build time
    recency_weight:   float = 1.0


@dataclass
class ConceptEdge:
    """User → Concept edge."""
    concept_slug:     str
    edge_type:        EdgeType
    mastery_score:    float = 0.0     # BKT P(L), 0–1
    confidence:       float = 1.0     # 1 - CV over last 5 sessions. high=consistent
    half_life:        float = 1.0     # HLR half-life in days (from hlr_store)
    urgency:          float = 0.0     # HLR urgency, 0–1 (derived: 1 - recall_prob)
    severity:         float = 0.0     # ConceptGapProfile.severity, 0–1
    sm2_ef:           float = 2.5     # SM-2 easiness factor
    sm2_interval:     int   = 1       # days
    next_review_date: Optional[str] = None
    last_attempted:   Optional[float] = None


@dataclass
class ConceptConceptEdge:
    """Concept → Concept edge (offline graph structure)."""
    source_slug:  str
    target_slug:  str
    edge_type:    EdgeType
    weight:       float = 1.0   # jaccard for COOCCURS, 1.0 for PREREQ


# ---------------------------------------------------------------------------
# User node (single instance per graph)
# ---------------------------------------------------------------------------

@dataclass
class UserNode:
    user_id:       str
    username:      str      = ""
    # Stored here AND indexed in Qdrant problems_full — kept in sync by UserStateBuilder
    embedding:     Optional[list] = None   # 1920-d user state vector (flat list for JSON)
    elo_rating:    int      = 1200         # ELO rating — update logic TBD, field reserved
    cf_rating:     int      = 0
    lc_solved:     int      = 0
    total_xp:      int      = 0
    current_level: int      = 1
    onboarding_complete: bool = False
    created_at:    Optional[float] = None


# ---------------------------------------------------------------------------
# User graph
# ---------------------------------------------------------------------------

@dataclass
class UserGraph:
    """
    Full in-memory user graph.  One instance per user per request.

    Lifecycle:
        1. UserStateBuilder.build(user_id, db) constructs this from
           Postgres telemetry + the offline concept graph.
        2. The graph is serialised to Redis (TTL = 5 min) keyed by user_id.
        3. UserStateBuilder.to_vector(graph) projects it to a 1920-d vector.
        4. On the next submission the cache is invalidated and rebuilt.

    The user node is ALWAYS present (even for cold-start users with no
    submissions).  Edge lists start empty and fill from telemetry.
    """
    user:            UserNode

    # user ↔ problem edges  (keyed by problem_id for O(1) dedup lookup)
    problem_edges:   dict[str, ProblemEdge]        = field(default_factory=dict)

    # user ↔ concept edges  (keyed by concept_slug)
    concept_edges:   dict[str, ConceptEdge]        = field(default_factory=dict)

    # concept ↔ concept edges (offline structure, keyed source→[targets])
    cc_edges:        dict[str, list[ConceptConceptEdge]] = field(default_factory=dict)

    # solved problem IDs (fast dedup for candidate pool filtering)
    solved_ids:      set[str]                      = field(default_factory=set)

    # recommended problem IDs with timestamps (7-day dedup window)
    exposed_ids:     dict[str, float]              = field(default_factory=dict)

    # skipped 3+ times → deprioritise from pool
    deprioritised_ids: set[str]                    = field(default_factory=set)

    built_at:        float = field(default_factory=time.time)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def mastered_concepts(self, threshold: float = 0.7) -> list[str]:
        return [s for s, e in self.concept_edges.items()
                if e.mastery_score >= threshold]

    def weak_concepts(self, threshold: float = 0.6) -> list[str]:
        return [s for s, e in self.concept_edges.items()
                if e.severity >= threshold]

    def urgent_concepts(self, threshold: float = 0.6) -> list[str]:
        return [s for s, e in self.concept_edges.items()
                if e.urgency >= threshold]

    def concept_mastery(self, slug: str) -> float:
        e = self.concept_edges.get(slug)
        return e.mastery_score if e else 0.0

    def recently_exposed(self, problem_id: str, window_days: float = 7.0) -> bool:
        ts = self.exposed_ids.get(problem_id)
        if ts is None:
            return False
        return (time.time() - ts) < window_days * 86400

    def add_problem_edge(self, edge: ProblemEdge) -> None:
        existing = self.problem_edges.get(edge.problem_id)
        if existing is None or edge.timestamp > existing.timestamp:
            self.problem_edges[edge.problem_id] = edge
        if edge.edge_type == EdgeType.SOLVED:
            self.solved_ids.add(edge.problem_id)
        if edge.edge_type == EdgeType.EXPOSED:
            self.exposed_ids[edge.problem_id] = edge.timestamp
        if edge.edge_type == EdgeType.SKIPPED:
            # check skip_count via caller; caller sets SKIPPED when skip_count >= 3
            self.deprioritised_ids.add(edge.problem_id)

    def add_concept_edge(self, edge: ConceptEdge) -> None:
        existing = self.concept_edges.get(edge.concept_slug)
        if existing is None:
            self.concept_edges[edge.concept_slug] = edge
        else:
            # merge: take the highest mastery + worst severity
            existing.mastery_score = max(existing.mastery_score, edge.mastery_score)
            existing.severity      = max(existing.severity,      edge.severity)
            existing.urgency       = max(existing.urgency,       edge.urgency)

    def update_concept_state(self, concept_slug: str, **fields) -> None:
        """
        Authoritative overwrite for a single concept's state, used by live
        post-submission updates (StateUpdateService) where the new BKT/HLR
        output IS the new truth for this topic -- not one of several
        conflicting sources to conservatively merge.

        This is deliberately NOT add_concept_edge(), which max-merges
        mastery_score across multiple construction-time sources (DB rows,
        BKT store, etc.) on the assumption that a lower value might just be
        stale data from a source that hasn't caught up yet. That assumption
        is wrong for a live update: if BKT computes a genuinely LOWER
        mastery after a poor submission, add_concept_edge's max-merge would
        silently keep the old higher value, making the graph never reflect
        a real mastery decrease. update_concept_state always overwrites.

        Usage: graph.update_concept_state("arrays", mastery_score=0.62,
                                          urgency=0.3, half_life=5.0)
        """
        existing = self.concept_edges.get(concept_slug)
        if existing is None:
            self.concept_edges[concept_slug] = ConceptEdge(
                concept_slug=concept_slug,
                edge_type=fields.pop("edge_type", EdgeType.LEARNING),
                **fields,
            )
            return
        for key, value in fields.items():
            setattr(existing, key, value)

    def add_cc_edge(self, edge: ConceptConceptEdge) -> None:
        self.cc_edges.setdefault(edge.source_slug, []).append(edge)
        self._prereq_index_cache = None   # invalidate on any graph mutation

    def is_locked(self, topic_tags: list, mastery_threshold: float = 0.7) -> bool:
        """
        True if ANY of the given topic tags requires a prerequisite concept
        the user hasn't mastered yet. Single source of truth for prereq
        gating -- both the candidate filtering layer and every pool's own
        self-filter call this instead of each building their own reverse
        index from cc_edges, so "locked" always means the same thing
        everywhere in the pipeline.

        cc_edges is keyed by source concept with PREREQ edges pointing to
        the concept it unlocks (source is prerequisite OF target). The
        reverse index built here maps target -> [required prereq slugs].
        """
        if not topic_tags:
            return False
        index = self._prereq_index()
        mastered = set(self.mastered_concepts(mastery_threshold))
        for tag in topic_tags:
            for prereq_slug in index.get(tag, []):
                if prereq_slug not in mastered:
                    return True
        return False

    def _prereq_index(self) -> dict:
        cache = getattr(self, "_prereq_index_cache", None)
        if cache is not None:
            return cache
        reverse: dict = {}
        for src, edges in self.cc_edges.items():
            for e in edges:
                if e.edge_type == EdgeType.PREREQ:
                    reverse.setdefault(e.target_slug, []).append(src)
        self._prereq_index_cache = reverse
        return reverse

    # ------------------------------------------------------------------
    # Serialisation (compact dict for Redis)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "user":           self.user.__dict__,
            "problem_edges":  {k: v.__dict__ for k, v in self.problem_edges.items()},
            "concept_edges":  {k: v.__dict__ for k, v in self.concept_edges.items()},
            "cc_edges":       {
                k: [e.__dict__ for e in edges]
                for k, edges in self.cc_edges.items()
            },
            "solved_ids":     list(self.solved_ids),
            "exposed_ids":    self.exposed_ids,
            "deprioritised":  list(self.deprioritised_ids),
            "built_at":       self.built_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UserGraph":
        g = cls(user=UserNode(**d["user"]))
        g.problem_edges = {
            k: ProblemEdge(**v) for k, v in d["problem_edges"].items()
        }
        g.concept_edges = {
            k: ConceptEdge(**v) for k, v in d["concept_edges"].items()
        }
        # .get(..., {}) handles graphs cached BEFORE this fix -- they'll
        # restore with empty cc_edges instead of raising KeyError.
        g.cc_edges = {
            k: [ConceptConceptEdge(**e) for e in edges]
            for k, edges in d.get("cc_edges", {}).items()
        }
        g.solved_ids        = set(d["solved_ids"])
        g.exposed_ids       = d["exposed_ids"]
        g.deprioritised_ids = set(d["deprioritised"])
        g.built_at          = d["built_at"]
        return g
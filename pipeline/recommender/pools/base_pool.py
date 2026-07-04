"""
Candidate pools.

Each pool produces a list of candidate problem_ids for one strategy. They all
share the same interface so the pool generation layer can call them uniformly
and hand their output to candidate filtering.

A pool NEVER makes the final recommendation. It only proposes candidates.
Filtering (remove solved / locked / out-of-range / missing prereqs) and
ranking happen downstream.

Pools:
  A       course path         - next problems in curriculum order
  B_C     near / far transfer  - same or analogous pattern to what user solved
  D       weakness recovery    - target weak / low-mastery concepts
  E       spaced review        - overdue SM-2 / high-urgency HLR concepts
  F       stretch              - slightly above current ability
  G       novelty              - unseen concepts reachable from mastered ones
  vector  semantic similarity  - ANN over the user state vector

Every pool takes:
  - graph  : UserGraph (BKT / HLR / SM-2 / severity per concept)
  - state  : UserStateVector (1920-d user embedding + concept lists)
  - qdrant : a client with .query_points / .scroll (may be None for graph-only pools)
  - n      : how many candidates to return (the controller's weight decides this)

and returns:
  list[Candidate]  where Candidate = (problem_id, pool_name)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pipeline.recommender.models.user_graph import UserGraph
from pipeline.recommender.models.user_state import UserStateVector


@dataclass
class Candidate:
    problem_id:      str
    pool:            str
    score:           float = 0.0    # pool-local relevance, filled where meaningful
    topic_tags:      list  = None   # populated from Qdrant payload when available
    difficulty_score: float = None  # populated from Qdrant payload when available

    def __post_init__(self):
        if self.topic_tags is None:
            self.topic_tags = []


class BasePool:
    """Shared helpers for all pools."""

    name = "base"

    def __init__(self, qdrant=None, collection: str = "problems_full"):
        self.qdrant = qdrant
        self.collection = collection

    def generate(self, graph: UserGraph, state: UserStateVector,
                 n: int = 20) -> list[Candidate]:
        raise NotImplementedError

    # -- shared helpers ---------------------------------------------------

    def _exclude_ids(self, graph: UserGraph) -> set:
        """Problems we never want to propose: already solved or deprioritised."""
        return set(graph.solved_ids) | set(graph.deprioritised_ids)

    def _problems_by_concept(self, concept_slugs, n, exclude,
                             difficulty=None) -> list[Candidate]:
        """
        Pull problems tagged with any of the given concepts from Qdrant,
        optionally filtered to a difficulty band. Uses scroll (metadata only,
        no vector needed). difficulty is (lo, hi) on difficulty_score or None.
        """
        if not self.qdrant or not concept_slugs:
            return []
        from qdrant_client.models import (
            Filter, FieldCondition, MatchAny, Range,
        )
        must = [FieldCondition(key="topic_tags", match=MatchAny(any=list(concept_slugs)))]
        if difficulty is not None:
            lo, hi = difficulty
            must.append(FieldCondition(
                key="difficulty_score", range=Range(gte=lo, lte=hi)))
        try:
            points, _ = self.qdrant.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(must=must),
                limit=n * 3, with_payload=True, with_vectors=False,
            )
        except Exception:
            return []
        out = []
        for p in points:
            pl = p.payload or {}
            pid = str(pl.get("problem_id", p.id))
            if pid in exclude:
                continue
            out.append(Candidate(
                pid, self.name,
                topic_tags=pl.get("topic_tags") or [],
                difficulty_score=pl.get("difficulty_score"),
            ))
            if len(out) >= n:
                break
        return out

    def _ann(self, query_vec, n, exclude) -> list[Candidate]:
        """ANN search over the user/query vector on the full collection."""
        if not self.qdrant or query_vec is None:
            return []
        try:
            hits = self.qdrant.query_points(
                collection_name=self.collection,
                query=query_vec,
                limit=n * 3, with_payload=True, with_vectors=False,
            ).points
        except Exception:
            return []
        out = []
        for h in hits:
            pl = h.payload or {}
            pid = str(pl.get("problem_id", h.id))
            if pid in exclude:
                continue
            out.append(Candidate(
                pid, self.name, score=float(getattr(h, "score", 0.0)),
                topic_tags=pl.get("topic_tags") or [],
                difficulty_score=pl.get("difficulty_score"),
            ))
            if len(out) >= n:
                break
        return out

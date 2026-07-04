"""
The seven candidate pools. Each subclasses BasePool and implements generate().
"""

from __future__ import annotations

from pipeline.recommender.models.user_graph import UserGraph, EdgeType
from pipeline.recommender.models.user_state import UserStateVector
from pipeline.recommender.pools.base_pool import BasePool, Candidate


# Difficulty bands on the 0-1 difficulty_score. No ELO -- these are fixed
# bands; the adaptive controller decides how much of each band to draw.
EASY_BAND = (0.0, 0.34)
MED_BAND  = (0.34, 0.66)
HARD_BAND = (0.66, 1.0)


class CoursePathPool(BasePool):
    """
    Pool A - course path.
    Next problems in curriculum order: concepts the user is currently learning
    (not yet mastered) plus concepts unlocked by their mastered prerequisites.
    """
    name = "A"

    def generate(self, graph, state, n=20):
        exclude = self._exclude_ids(graph)
        mastered = set(graph.mastered_concepts())

        # concepts in progress (has an edge but not mastered)
        in_progress = [s for s, e in graph.concept_edges.items()
                       if s not in mastered]

        # concepts unlocked: targets whose prereqs are all mastered
        unlocked = []
        for src, edges in graph.cc_edges.items():
            for e in edges:
                if e.edge_type == EdgeType.PREREQ and src in mastered:
                    unlocked.append(e.target_slug)

        target_concepts = list(dict.fromkeys(in_progress + unlocked))
        if not target_concepts:
            # cold start: fall back to any low-difficulty problems
            return self._problems_by_concept(
                [e for e in graph.concept_edges] or [], n, exclude,
                difficulty=EASY_BAND)
        return self._problems_by_concept(target_concepts, n, exclude)


class TransferPool(BasePool):
    """
    Pool B/C - near and far transfer.
    Near: same pattern as recently solved, different surface.
    Far:  analogous pattern reachable via concept co-occurrence edges.
    Implemented as ANN over the user vector, since the user state vector
    already encodes the patterns they have been solving.
    """
    name = "B_C"

    def generate(self, graph, state, n=20):
        exclude = self._exclude_ids(graph)
        qv = state.to_query_vector()
        if qv is not None:
            return self._ann(qv, n, exclude)
        # graph-only fallback: co-occurring concepts of what they know
        cooccur = []
        for edges in graph.cc_edges.values():
            for e in edges:
                if e.edge_type == EdgeType.COOCCURS:
                    cooccur.append(e.target_slug)
        return self._problems_by_concept(cooccur, n, exclude)


class WeaknessPool(BasePool):
    """
    Pool D - weakness recovery.
    Targets concepts with high gap severity or low BKT mastery. Draws easier
    problems so the user can rebuild the skill.
    """
    name = "D"

    def generate(self, graph, state, n=20):
        exclude = self._exclude_ids(graph)
        weak = set(graph.weak_concepts())
        low_mastery = [s for s, e in graph.concept_edges.items()
                       if e.mastery_score < 0.4]
        target = list(weak | set(low_mastery))
        if not target:
            return []
        # weakness problems lean easy-to-medium
        return self._problems_by_concept(
            target, n, exclude, difficulty=(EASY_BAND[0], MED_BAND[1]))


class SpacedReviewPool(BasePool):
    """
    Pool E - spaced review.
    Concepts that are overdue (SM-2 next_review_date in the past) or high HLR
    urgency (user is forgetting). Draws at the difficulty the user learned them.
    """
    name = "E"

    def generate(self, graph, state, n=20):
        exclude = self._exclude_ids(graph)
        urgent = set(graph.urgent_concepts())

        # overdue by SM-2 date
        import time
        from datetime import datetime, timezone
        now = time.time()
        overdue = set()
        for s, e in graph.concept_edges.items():
            if not e.next_review_date:
                continue
            try:
                due = datetime.fromisoformat(e.next_review_date)
                if due.tzinfo is None:
                    due = due.replace(tzinfo=timezone.utc)
                if due.timestamp() <= now:
                    overdue.add(s)
            except (ValueError, TypeError):
                continue

        target = list(urgent | overdue)
        if not target:
            return []
        return self._problems_by_concept(target, n, exclude)


class StretchPool(BasePool):
    """
    Pool F - stretch.
    Problems slightly above current ability on concepts the user has some
    grip on (partial mastery). Draws medium-to-hard difficulty.
    """
    name = "F"

    def generate(self, graph, state, n=20):
        exclude = self._exclude_ids(graph)
        # concepts with moderate mastery: enough grip to stretch, not mastered
        stretch_concepts = [s for s, e in graph.concept_edges.items()
                            if 0.4 <= e.mastery_score < 0.75]
        if not stretch_concepts:
            # if nothing partial, stretch on mastered concepts at hard difficulty
            stretch_concepts = list(graph.mastered_concepts())
        if not stretch_concepts:
            return []
        return self._problems_by_concept(
            stretch_concepts, n, exclude, difficulty=(MED_BAND[0], HARD_BAND[1]))


class NoveltyPool(BasePool):
    """
    Pool G - novelty.
    Concepts the user has never touched, reachable from their mastered
    concepts via prerequisite or co-occurrence edges. Introduces new topics
    gently at easier difficulty.
    """
    name = "G"

    def generate(self, graph, state, n=20):
        exclude = self._exclude_ids(graph)
        seen = set(graph.concept_edges.keys())
        mastered = set(graph.mastered_concepts())

        # new concepts reachable from mastered ones
        novel = []
        for src, edges in graph.cc_edges.items():
            if src not in mastered:
                continue
            for e in edges:
                if e.target_slug not in seen:
                    novel.append(e.target_slug)

        novel = list(dict.fromkeys(novel))
        if not novel:
            return []
        return self._problems_by_concept(
            novel, n, exclude, difficulty=EASY_BAND)


class VectorPool(BasePool):
    """
    Pool vector - semantic similarity.
    Pure ANN over the user state vector on the 1920-d full collection.
    Surfaces structurally similar problems regardless of topic label.
    """
    name = "vector"

    def generate(self, graph, state, n=20):
        exclude = self._exclude_ids(graph)
        qv = state.to_query_vector()
        if qv is None:
            return []
        return self._ann(qv, n, exclude)


# registry so the pool generation layer can build them by name
POOL_CLASSES = {
    "A":      CoursePathPool,
    "B_C":    TransferPool,
    "D":      WeaknessPool,
    "E":      SpacedReviewPool,
    "F":      StretchPool,
    "G":      NoveltyPool,
    "vector": VectorPool,
}


def build_pools(qdrant=None, collection="problems_full") -> dict:
    """Instantiate every pool, keyed by name."""
    return {name: cls(qdrant=qdrant, collection=collection)
            for name, cls in POOL_CLASSES.items()}

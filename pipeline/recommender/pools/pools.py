"""
The seven candidate pools. Each subclasses BasePool and implements generate().

generate(graph, state, n, mix) now takes TWO quotas from the pipeline:
  - n    : how many candidates this pool should return (from the adaptive
           difficulty controller's per-pool WEIGHT, converted to a count by
           the pool generation orchestrator)
  - mix  : this pool's easy/medium/hard percentage split (from the adaptive
           difficulty controller's per-pool MIX), so the pool draws the
           right proportion of each difficulty band instead of one fixed band

Each pool restricts the controller's mix to its own natural difficulty lean
via ALLOWED_BANDS (e.g. pool D never draws "hard" even if the controller's
global mix includes some hard percentage -- that gets renormalised away).
_draw_with_mix on BasePool does the actual proportional splitting.
"""

from __future__ import annotations

from pipeline.recommender.models.user_graph import UserGraph, EdgeType
from pipeline.recommender.models.user_state import UserStateVector
from pipeline.recommender.pools.base_pool import (
    BasePool, Candidate, EASY_BAND, MED_BAND, HARD_BAND,
)

# Foundational topics for a genuinely cold-start user: someone with ZERO
# concept_edges and ZERO cc_edges (cc_edges are only loaded for concepts the
# user has already touched, so a brand new user has none either). Without
# this, CoursePathPool's and NoveltyPool's cold-start branches had nothing
# to fall back to except graph.concept_edges/graph.cc_edges -- which are
# exactly what's empty in the case they exist to handle, so both pools
# silently returned zero candidates for every new user.
#
# Matches KNode's difficulty_tier=1 seeded topics ("1=Arrays/Strings" per
# the schema) -- the two topics every learner starts with regardless of
# background. Adjust this list if the seeded topic taxonomy changes.
STARTER_CONCEPTS = ["arrays", "strings", "hash_map", "sorting"]


class CoursePathPool(BasePool):
    """
    Pool A - course path.
    Next problems in curriculum order: concepts the user is currently learning
    (not yet mastered) plus concepts unlocked by their mastered prerequisites.
    """
    name = "A"
    ALLOWED_BANDS = ("easy", "medium", "hard")   # follows the controller's base mix

    def generate(self, graph, state, n=20, mix=None):
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
            # Genuinely cold start (no concepts, no unlocks): the old
            # fallback here read graph.concept_edges, which is empty by
            # definition for exactly this case -- it always returned
            # nothing. Fixed to use STARTER_CONCEPTS, but the FIRST fix
            # still only ever searched EASY_BAND with a single query --
            # if the dataset simply has few/no easy-difficulty problems
            # tagged with these starter concepts (common for general
            # problem manifests), this returned zero even when
            # medium-difficulty matches existed. Using _draw_with_mix
            # respects the controller's actual easy/medium/hard split for
            # this pool, matching every other pool's approach, so it finds
            # candidates across whatever bands actually have data.
            return self._draw_with_mix(STARTER_CONCEPTS, n, exclude, mix, graph=graph)
        return self._draw_with_mix(target_concepts, n, exclude, mix, graph=graph)


class TransferPool(BasePool):
    """
    Pool B/C - near and far transfer.
    Near: same pattern as recently solved, different surface.
    Far:  analogous pattern reachable via concept co-occurrence edges.
    Implemented as ANN over the user vector, since the user state vector
    already encodes the patterns they have been solving. ANN doesn't filter
    by difficulty band directly, so `mix` only applies to the graph-fallback
    path (no vector available).
    """
    name = "B_C"
    ALLOWED_BANDS = ("easy", "medium", "hard")

    def generate(self, graph, state, n=20, mix=None):
        exclude = self._exclude_ids(graph)
        qv = state.to_query_vector() if state is not None else None
        if qv is not None:
            return self._ann(
    qv,
    n,
    exclude,
    graph=graph,
    mix=mix,
)
        # graph-only fallback: co-occurring concepts of what they know
        cooccur = []
        for edges in graph.cc_edges.values():
            for e in edges:
                if e.edge_type == EdgeType.COOCCURS:
                    cooccur.append(e.target_slug)
        return self._draw_with_mix(cooccur, n, exclude, mix, graph=graph)


class WeaknessPool(BasePool):
    """
    Pool D - weakness recovery.
    Targets concepts with high gap severity or low BKT mastery. Never draws
    hard problems -- the controller's mix is renormalised across easy/medium
    only, so a global mix with e.g. 20% hard still keeps this pool gentle.
    """
    name = "D"
    ALLOWED_BANDS = ("easy", "medium")

    def generate(self, graph, state, n=20, mix=None):
        exclude = self._exclude_ids(graph)
        weak = set(graph.weak_concepts())
        low_mastery = [s for s, e in graph.concept_edges.items()
                       if e.mastery_score < 0.4]
        target = list(weak | set(low_mastery))
        if not target:
            return []
        return self._draw_with_mix(target, n, exclude, mix, graph=graph)


class SpacedReviewPool(BasePool):
    """
    Pool E - spaced review.
    Concepts that are overdue (SM-2 next_review_date in the past) or high HLR
    urgency (user is forgetting). Draws at the difficulty the user learned
    them, following the controller's full mix.
    """
    name = "E"
    ALLOWED_BANDS = ("easy", "medium", "hard")

    def generate(self, graph, state, n=20, mix=None):
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
        return self._draw_with_mix(target, n, exclude, mix, graph=graph)


class StretchPool(BasePool):
    """
    Pool F - stretch.
    Problems slightly above current ability on concepts the user has some
    grip on (partial mastery). Never draws easy -- renormalised across
    medium/hard only, so growth stays growth even if the controller's
    global mix has an easy percentage.
    """
    name = "F"
    ALLOWED_BANDS = ("medium", "hard")

    def generate(self, graph, state, n=20, mix=None):
        exclude = self._exclude_ids(graph)
        # concepts with moderate mastery: enough grip to stretch, not mastered
        stretch_concepts = [s for s, e in graph.concept_edges.items()
                            if 0.4 <= e.mastery_score < 0.75]
        if not stretch_concepts:
            # if nothing partial, stretch on mastered concepts instead
            stretch_concepts = list(graph.mastered_concepts())
        if not stretch_concepts:
            return []
        return self._draw_with_mix(stretch_concepts, n, exclude, mix, graph=graph)


class NoveltyPool(BasePool):
    """
    Pool G - novelty.
    Concepts the user has never touched, reachable from their mastered
    concepts via prerequisite or co-occurrence edges. Introduces new topics
    gently -- restricted to easy/medium, no hard.
    """
    name = "G"
    ALLOWED_BANDS = ("easy", "medium")

    def generate(self, graph, state, n=20, mix=None):
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
            # Genuinely cold start (nothing mastered yet, so nothing to
            # branch novelty from) -- fall back to starter topics rather
            # than returning nothing. Excludes anything already in the
            # user's graph so this doesn't just duplicate CoursePathPool's
            # fallback for a user who has SOME data but no mastered concepts.
            fallback = [c for c in STARTER_CONCEPTS if c not in seen]
            if not fallback:
                return []
            return self._draw_with_mix(fallback, n, exclude, mix, graph=graph)
        return self._draw_with_mix(novel, n, exclude, mix, graph=graph)


class VectorPool(BasePool):
    """
    Pool vector - semantic similarity.
    Pure ANN over the user state vector on the 1920-d full collection.
    Surfaces structurally similar problems regardless of topic label.
    ANN doesn't filter by difficulty band -- mix is accepted for interface
    consistency but has no effect here (pure similarity search).
    """
    name = "vector"
    ALLOWED_BANDS = ("easy", "medium", "hard")

    def generate(self, graph, state, n=20, mix=None):
        exclude = self._exclude_ids(graph)
        qv = state.to_query_vector() if state is not None else None
        if qv is None:
            return []
        return self._ann(qv, n, exclude, graph=graph, mix=mix)


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
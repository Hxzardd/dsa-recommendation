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
#
# FIX: "arrays"/"strings"/"sorting" are not real tags in
# data/problem_topic_edges_normalized.json (verified directly: the real
# manifest uses singular "array"/"string", and there is no generic
# "sorting" tag at all -- only specific variants like merge_sort/
# cyclic_sort/counting_sort, each with a handful of problems). With the
# wrong plural tags, CoursePathPool's cold-start fallback (see below) was
# effectively only ever matching "hash_map" (118 problems) -- 3 of its 4
# starter topics silently matched zero problems, badly narrowing the easy-
# difficulty candidate pool a brand new user's very first recommendation
# draws from. "array" alone has 790 tagged problems in this catalog.
STARTER_CONCEPTS = ["array", "string", "hash_map"]

# How many of the easiest STARTER_CONCEPTS problems NoveltyPool's cold-start
# fallback skips past before taking its own slice (see
# BasePool._lowest_difficulty_by_concept's `skip` docstring). Generous
# enough to clear CoursePathPool's own cold-start draw comfortably: at
# cold-start weights (adaptive_difficulty.py's is_cold branch) A's requested
# count is the largest of any pool and, even at the MAX_PER_POOL_ABSOLUTE
# ceiling, stays well under this value -- so the two pools' cold-start
# slices don't overlap in practice, without the two pools needing to know
# anything about each other's actual draw count.
NOVELTY_COLD_START_SKIP = 20


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
            # Genuinely cold start (no concepts, no unlocks). Previously used
            # _draw_with_mix, which filters into a fixed absolute EASY_BAND
            # (0-0.34) -- but this catalog has ZERO problems below 0.34
            # difficulty at all (confirmed by direct query), so the "easy"
            # quota always came back empty and cold-start users landed on
            # medium/hard instead. "Lowest possible difficulty" for a
            # catalog like this has to mean relative-lowest, not
            # absolute-lowest -- so cold start ranks by actual
            # difficulty_score ascending and takes the n lowest available,
            # rather than filtering into a band that may not exist.
            return self._lowest_difficulty_by_concept(STARTER_CONCEPTS, n, exclude, graph=graph)
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
            # Intentionally empty at cold start, not a gap to fill: "weak"
            # is a RELATIVE judgement (a concept the user is doing worse on
            # than their own average, or has attempted and struggled with)
            # -- there is no such comparison to make for a user with zero
            # concept_edges. Any candidate this pool proposed here would
            # have to invent a weakness that doesn't exist yet, which is
            # exactly what we must not do. Once the user has any real
            # attempt history, weak_concepts()/low_mastery populate
            # normally and this pool activates on genuine signal.
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
            # Intentionally empty at cold start: spaced review is defined
            # entirely in terms of PAST study events -- HLR urgency needs a
            # last_review timestamp, SM-2 overdue-ness needs a prior
            # next_review_date. A user with zero concept_edges has never
            # studied anything, so there is nothing due for review yet, by
            # definition -- not a missing signal to compensate for. Once
            # the user has solved even one problem, urgent_concepts()/
            # overdue populate from real history and this pool activates.
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
            # Genuinely cold start: no concept has any mastery to stretch
            # from. Unlike WeaknessPool/SpacedReviewPool, this pool's
            # ALLOWED_BANDS=(medium, hard) restriction is itself a real,
            # non-fabricated signal of what "stretch" means here -- draw
            # STARTER_CONCEPTS through the pool's own existing band-mix
            # mechanism (_draw_with_mix already renormalises to just
            # medium/hard for this pool) instead of inventing a mastery
            # number. This is a genuinely different slice of the catalog
            # from CoursePathPool/NoveltyPool's cold-start fallback, which
            # both rank by absolute difficulty_score with no band
            # restriction at all -- no new mechanism, just this pool using
            # the one it already has for every other case instead of
            # bailing out to nothing.
            return self._draw_with_mix(STARTER_CONCEPTS, n, exclude, mix, graph=graph)
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
            # Same catalog-difficulty-floor issue as CoursePathPool's
            # cold-start branch (see _lowest_difficulty_by_concept's
            # docstring) -- ranks by actual difficulty_score instead of a
            # fixed EASY_BAND this catalog has no data in.
            #
            # skip=NOVELTY_COLD_START_SKIP: for a genuinely cold user this
            # queries the SAME STARTER_CONCEPTS with the SAME exclude set as
            # CoursePathPool's own cold-start fallback -- without an offset
            # both pools would return the identical candidates, and
            # CandidateFilteringLayer's dedup would just relabel them under
            # both pool names, artificially inflating pool_agreement in the
            # ranker for what is really one signal, not two agreeing ones.
            # Skipping past CoursePathPool's slice keeps this pool's actual
            # purpose (introduce something the user hasn't already been
            # shown) genuinely true at cold start too, not just once the
            # user has real mastered/novel-reachable concepts.
            fallback = [c for c in STARTER_CONCEPTS if c not in seen]
            if not fallback:
                return []
            return self._lowest_difficulty_by_concept(
                fallback, n, exclude, graph=graph, skip=NOVELTY_COLD_START_SKIP)
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
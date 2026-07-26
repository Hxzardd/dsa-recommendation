"""
Candidate filtering layer.

Sits between the 7 candidate pools and the ranker (Shraddha's LightGBM /
scoring layer). Per the architecture diagram:

    pool A, B_C, D, E, F, G, vector
        --(each pool's raw candidates, already difficulty-banded)-->
    per-pool filter: removes too hard/easy, solved, missing prereqs
        -->
    CANDIDATE FILTERING:
        - each pool sends {problem_id: pool_name} into a local object
        - concatenated after removing garbage recommendations into a
          common object {[pid1, pid2, ... pidn]: pool_name}
        - GLOBAL FILTERING: removes duplicated questions, further
          processes the question pool
        -->
    filter to 55-80% predicted success (optimal ~68%, the ZPD band)
        -->
    Shraddha's ranker / scoring engine

This module owns everything up to and including the ZPD band filter. It does
NOT rank or score candidates for final ordering -- that is the ranker's job.
It DOES need a predicted-success estimate to apply the ZPD filter; a simple
BKT-mastery-based estimator is provided as a default, since Shraddha's BKT
mastery scores are already present on every ConceptEdge in the UserGraph
(same store this whole recommender reads from). If a real trained
success-probability model exists later, pass it in via
`success_estimator=` and this layer will use that instead.

This is the connection point to Shraddha's recommendation engine: every
mastery/urgency number used here comes directly from the same UserGraph her
BKT/HLR stores populate (via UserGraphService) -- no separate data source.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from pipeline.recommender.models.user_graph import UserGraph
from pipeline.recommender.pools.base_pool import Candidate


# Locked / eligibility ------------------------------------------------------

# ZPD band from the architecture diagram: candidates with predicted success
# probability outside [ZPD_LO, ZPD_HI] are filtered out. ZPD_OPTIMAL is the
# sweet spot the diagram calls out (~68%) -- not a hard cutoff, just the
# center of the band, exposed for the ranker to use as a tie-breaker if it
# wants to prefer candidates closer to it.
ZPD_LO      = 0.55
ZPD_HI      = 0.80
ZPD_OPTIMAL = 0.68


_UNSET = object()   # distinguishes "not computed yet" from a genuine None result


@dataclass
class MergedCandidate:
    """One problem, deduplicated across every pool that proposed it."""
    problem_id:        str
    pool_sources:       list          # e.g. ["A", "vector"] -- every pool that proposed it
    best_score:         float         # max pool-local score across sources
    topic_tags:         list
    difficulty_score:   Optional[float]
    predicted_success:  Optional[float] = None   # filled by apply_zpd_filter
    # Cache for CandidateFilteringLayer._topic_mastery_and_urgency -- see its
    # docstring. Not part of this object's public shape (excluded from
    # equality/repr so it doesn't leak into any existing comparison/debug
    # output), just a memoization slot.
    _topic_stats_cache: object = field(default=_UNSET, repr=False, compare=False)

    @property
    def pool_count(self) -> int:
        return len(self.pool_sources)


@dataclass
class FilterReport:
    """What the filtering layer removed and why -- for debugging / logging."""
    input_count:          int = 0
    removed_solved:       int = 0
    removed_deprioritised: int = 0
    removed_locked:       int = 0
    removed_duplicates:   int = 0     # candidates merged into an existing entry
    removed_zpd:          int = 0
    output_count:         int = 0

    def to_dict(self) -> dict:
        return {
            "input_count":           self.input_count,
            "removed_solved":        self.removed_solved,
            "removed_deprioritised": self.removed_deprioritised,
            "removed_locked":        self.removed_locked,
            "removed_duplicates":    self.removed_duplicates,
            "removed_zpd":           self.removed_zpd,
            "output_count":          self.output_count,
        }


class CandidateFilteringLayer:
    """
    Usage:
        layer = CandidateFilteringLayer(graph)
        merged, report = layer.run({
            "A": pool_a_candidates,
            "D": pool_d_candidates,
            ...
        })
        ranker_input = layer.to_ranker_input(merged)
    """

    def __init__(self, graph: UserGraph,
                 success_estimator: Optional[Callable[["MergedCandidate", UserGraph], float]] = None):
        self.graph = graph
        self._success_estimator = success_estimator or self._default_success_estimator

    # ------------------------------------------------------------------ public

    def run(self, pool_candidates: dict, apply_zpd: bool = True) -> tuple:
        """
        Full pipeline: per-pool filter -> merge/dedup -> ZPD band filter.
        Returns (list[MergedCandidate], FilterReport).

        apply_zpd=False skips the final ZPD band step (still runs the
        per-pool solved/locked filter and dedup merge), returning the
        pre-ZPD merged list instead. Default True preserves this method's
        exact existing behaviour/output for every current caller -- see
        pipeline/recommender/services/recommend.py's get_recommendations
        for the one caller that uses apply_zpd=False, as a graceful-
        degradation retry when the strict ZPD band leaves zero candidates
        (a real, not hypothetical, case: a user with real but still-low
        mastery recommended against a catalog whose difficulty floor sits
        above what the ZPD band tolerates for that mastery level -- every
        candidate legitimately reads as "too hard", emptying the strict
        pass entirely). An empty /recommend response is a worse outcome
        than serving the best-available (pre-ZPD) candidates.
        """
        report = FilterReport()
        filtered_by_pool = {}
        for pool_name, candidates in pool_candidates.items():
            report.input_count += len(candidates)
            filtered_by_pool[pool_name] = self._filter_pool(candidates, report)

        merged, dup_count = self._merge(filtered_by_pool)
        report.removed_duplicates = dup_count

        if apply_zpd:
            merged = self._apply_zpd_filter(merged, report)
        report.output_count = len(merged)
        return merged, report

    # ------------------------------------------------------------- per-pool

    def _filter_pool(self, candidates: list, report: FilterReport) -> list:
        """Remove solved, deprioritised, and locked candidates from one pool's raw list."""
        out = []
        for c in candidates:
            if c.problem_id in self.graph.solved_ids:
                report.removed_solved += 1
                continue
            if c.problem_id in self.graph.deprioritised_ids:
                report.removed_deprioritised += 1
                continue
            if self._is_locked(c):
                report.removed_locked += 1
                continue
            out.append(c)
        return out

    def _is_locked(self, c: Candidate) -> bool:
        """
        A candidate is locked if ANY of its topic_tags requires a prerequisite
        concept the user hasn't mastered yet. Delegates to graph.is_locked()
        -- the single source of truth for prereq gating (see its docstring),
        also used by every pool's own self-filter -- instead of maintaining
        a second, independent reverse-index/mastered-set implementation that
        could silently diverge from it (e.g. if is_locked()'s default
        mastery_threshold is ever retuned, a local copy here would not
        follow along).
        """
        return self.graph.is_locked(c.topic_tags)

    # ------------------------------------------------------------- merge

    def _merge(self, filtered_by_pool: dict) -> tuple:
        """
        Concatenate every pool's filtered candidates into one deduplicated
        object. A problem recommended by multiple pools keeps all pool names
        in pool_sources and the best (max) score across them -- this is the
        "removing garbage recommendations into a common object" step from
        the diagram.
        """
        merged: dict = {}
        dup_count = 0
        for pool_name, candidates in filtered_by_pool.items():
            for c in candidates:
                if c.problem_id in merged:
                    dup_count += 1
                    existing = merged[c.problem_id]
                    if pool_name not in existing.pool_sources:
                        existing.pool_sources.append(pool_name)
                    existing.best_score = max(existing.best_score, c.score)
                    if not existing.topic_tags and c.topic_tags:
                        existing.topic_tags = c.topic_tags
                    if existing.difficulty_score is None:
                        existing.difficulty_score = c.difficulty_score
                else:
                    merged[c.problem_id] = MergedCandidate(
                        problem_id=c.problem_id,
                        pool_sources=[pool_name],
                        best_score=c.score,
                        topic_tags=list(c.topic_tags or []),
                        difficulty_score=c.difficulty_score,
                    )
        return list(merged.values()), dup_count

    # ------------------------------------------------------------- shared per-candidate stats

    def _topic_mastery_and_urgency(self, mc: MergedCandidate, graph: UserGraph) -> tuple:
        """
        Average BKT mastery and max HLR urgency across mc's topic tags --
        (None, None) if none of the tags have a ConceptEdge yet.

        Computed once per candidate and cached on the MergedCandidate itself
        (not on this layer instance) because _default_success_estimator
        (called from run()/_apply_zpd_filter) and to_ranker_input need the
        exact same aggregation, and the two are commonly called as two
        separate steps against the SAME MergedCandidate objects within one
        recommendation request (see pipeline/recommender/services/
        recommend.py and pool_generation.py, which each construct their own
        CandidateFilteringLayer instance around the same already-filtered
        merged list) -- caching on the object survives across those
        separate instances, unlike caching on self.
        """
        if mc._topic_stats_cache is not _UNSET:
            return mc._topic_stats_cache

        masteries = [
            graph.concept_edges[t].mastery_score
            for t in mc.topic_tags
            if t in graph.concept_edges
        ]
        urgencies = [
            graph.concept_edges[t].urgency
            for t in mc.topic_tags
            if t in graph.concept_edges
        ]
        avg_mastery = sum(masteries) / len(masteries) if masteries else None
        max_urgency = max(urgencies) if urgencies else None
        mc._topic_stats_cache = (avg_mastery, max_urgency)
        return mc._topic_stats_cache

    # ------------------------------------------------------------- ZPD filter

    def _default_success_estimator(self, mc: MergedCandidate, graph: UserGraph) -> float:
        """
        Fallback predicted-success estimate when no trained model is supplied.
        Combines average BKT mastery across the candidate's topic tags with
        a difficulty-delta penalty -- higher mastery and lower difficulty both
        raise predicted success. This directly reads Shraddha's BKT mastery
        scores off the same ConceptEdge objects the rest of the recommender
        uses (mastery_score is P(L) from her bkt.py, surfaced on the graph).

        Returns a value in (0, 1). Cold-start candidates (no mastery data on
        any of their tags) default to 0.68 -- the diagram's own optimal ZPD
        point -- so they aren't unfairly filtered out before enough signal
        exists to judge them.
        """
        if not mc.topic_tags:
            return ZPD_OPTIMAL

        avg_mastery, _ = self._topic_mastery_and_urgency(mc, graph)
        if avg_mastery is None:
            return ZPD_OPTIMAL

        difficulty = mc.difficulty_score if mc.difficulty_score is not None else 0.5

        # Sigmoid centered so that mastery == difficulty gives ~0.5, and
        # higher mastery relative to difficulty pushes success upward.
        delta = (avg_mastery - difficulty) * 6.0
        success = 1.0 / (1.0 + math.exp(-delta))
        return round(success, 4)

    def _apply_zpd_filter(self, merged: list, report: FilterReport) -> list:
        """
        Keep only candidates whose predicted success probability lands in
        [ZPD_LO, ZPD_HI] -- the productive-struggle zone from the diagram.
        Below it: too hard, likely frustration. Above it: too easy, no growth.
        """
        out = []
        for mc in merged:
            mc.predicted_success = self._success_estimator(mc, self.graph)
            if ZPD_LO <= mc.predicted_success <= ZPD_HI:
                out.append(mc)
            else:
                report.removed_zpd += 1
        return out

    # ------------------------------------------------------------- ranker bridge

    def to_ranker_input(self, merged: list) -> list:
        """
        Flatten MergedCandidate objects into plain dicts shaped for a
        downstream ranker (Shraddha's LightGBM feature vector). Every field
        here is traceable back to the same UserGraph her BKT/HLR stores feed --
        this is the actual integration point with her recommendation engine,
        not a separate data path.
        """
        rows = []
        for mc in merged:
            avg_mastery, max_urgency = self._topic_mastery_and_urgency(mc, self.graph)
            rows.append({
                "problem_id":        mc.problem_id,
                "pool_sources":      mc.pool_sources,
                "pool_count":        mc.pool_count,
                "topic_tags":        mc.topic_tags,
                "difficulty_score":  mc.difficulty_score,
                "avg_mastery":       round(avg_mastery, 4) if avg_mastery is not None else None,
                "max_urgency":       round(max_urgency, 4) if max_urgency is not None else None,
                "predicted_success": mc.predicted_success,
                "best_pool_score":   mc.best_score,
            })
        return rows

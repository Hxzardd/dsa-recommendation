"""
Pool generation orchestrator (PGO).

This is the piece that ties "7 independent, tested pools" into "an actual
recommendation for a user." It's the function the API layer calls with a
user_id's graph/state and gets back a filtered, deduplicated candidate list
ready for Shraddha's ranker.

Pipeline:
    UserGraph + UserStateVector
        --> AdaptiveDifficultyController.build_plan()
              per pool: WEIGHT (how much of the slate this pool should fill)
                        MIX    (this pool's own easy/medium/hard percentages)
        --> for each pool:
              n   = weight * total_n, capped by MAX_PER_POOL_ABSOLUTE
              mix = plan.mix_of(pool_name)
              pool.generate(graph, state, n=n, mix=mix)
        --> CandidateFilteringLayer.run()   (dedup, locked, ZPD band)
        --> merged candidates, ready for:
              - Shraddha's ranker (scores them)
              - DiversityMixer (enforces pool/topic/difficulty spread)

Two quotas are enforced here, per pool AND overall:
  - each pool gets its own n (from weight) AND its own easy/med/hard split
    (from mix) -- it no longer just gets a flat count with no difficulty
    awareness.
  - total_n itself is hard-capped by MAX_TOTAL_CANDIDATES so a
    misconfigured caller can't request an unbounded number of candidates
    (e.g. every pool's Qdrant scroll/query call scales with n).

This module does NOT rank or score for final ordering -- that's the ranker's
job. It stops at "here is the clean, deduplicated, eligible candidate set."
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pipeline.recommender.models.user_graph import UserGraph
from pipeline.recommender.models.user_state import UserStateVector
from pipeline.recommender.services.adaptive_difficulty import (
    AdaptiveDifficultyController, DifficultyPlan, POOLS,
)
from pipeline.recommender.services.candidate_filtering import (
    CandidateFilteringLayer, MergedCandidate, FilterReport,
)
from pipeline.recommender.pools.pools import build_pools
from pipeline.recommender.pools.base_pool import Candidate

# Minimum candidates requested per pool even if its weight rounds to 0.
# A pool with a very small weight (e.g. 0.02) still gets a chance to
# contribute rather than being silently starved out at low total_n.
MIN_PER_POOL = 2

# Hard ceiling on the TOTAL number of candidates requested across all pools
# combined, regardless of what total_n the caller passes in. Protects
# against an unbounded number of Qdrant round trips if total_n is
# misconfigured (e.g. accidentally passed as a per-user session count
# instead of a slate size).
MAX_TOTAL_CANDIDATES = 200

# Hard ceiling on candidates from any SINGLE pool, even if its weight alone
# would justify more (e.g. weight=0.9 at total_n=200 would otherwise ask
# one pool for 180 candidates). Keeps one dominant pool from monopolising
# Qdrant traffic and keeps the raw candidate set pool-diverse before it
# even reaches the diversity mixer.
MAX_PER_POOL_ABSOLUTE = 60


@dataclass
class PoolGenerationResult:
    """Everything the API layer needs after one generation pass."""
    merged_candidates: list                  # list[MergedCandidate]
    filter_report:      FilterReport
    difficulty_plan:     DifficultyPlan
    raw_counts:          dict = field(default_factory=dict)   # pool -> raw candidate count before filtering
    requested_counts:    dict = field(default_factory=dict)   # pool -> n actually requested (post-cap)

    def to_dict(self) -> dict:
        return {
            "difficulty_plan":  self.difficulty_plan.to_dict(),
            "filter_report":    self.filter_report.to_dict(),
            "raw_counts":       self.raw_counts,
            "requested_counts": self.requested_counts,
            "candidate_count":  len(self.merged_candidates),
        }


class PoolGenerationOrchestrator:
    """
    Usage:
        pgo = PoolGenerationOrchestrator(qdrant_client)
        result = pgo.generate(graph, state, total_n=30)
        ranker_input = CandidateFilteringLayer(graph).to_ranker_input(result.merged_candidates)
        # hand ranker_input to Shraddha's scoring model
    """

    def __init__(self, qdrant=None, collection: str = "problems_full",
                 controller: AdaptiveDifficultyController = None,
                 pools: dict = None):
        self.qdrant = qdrant
        self.collection = collection
        self.controller = controller or AdaptiveDifficultyController()
        # allow injecting pre-built pools (for tests / custom Qdrant clients);
        # otherwise build the standard 7 against this orchestrator's qdrant client
        self.pools = pools or build_pools(qdrant=qdrant, collection=collection)

    def generate(self, graph: UserGraph, state: UserStateVector,
                 total_n: int = 30) -> PoolGenerationResult:
        # Hard ceiling: no caller, misconfigured or not, can push this past
        # MAX_TOTAL_CANDIDATES total candidates requested across all pools.
        total_n = min(total_n, MAX_TOTAL_CANDIDATES)

        plan = self.controller.build_plan(graph)

        pool_candidates: dict = {}
        raw_counts: dict = {}
        requested_counts: dict = {}

        for pool_name in POOLS:
            pool = self.pools.get(pool_name)
            if pool is None:
                pool_candidates[pool_name] = []
                raw_counts[pool_name] = 0
                requested_counts[pool_name] = 0
                continue

            weight = plan.weight_of(pool_name)
            if weight <= 0:
                pool_candidates[pool_name] = []
                raw_counts[pool_name] = 0
                requested_counts[pool_name] = 0
                continue

            n = max(MIN_PER_POOL, round(weight * total_n))
            n = min(n, MAX_PER_POOL_ABSOLUTE)   # per-pool ceiling, independent of weight
            requested_counts[pool_name] = n

            mix = plan.mix_of(pool_name)
            candidates = pool.generate(graph, state, n=n, mix=mix)
            pool_candidates[pool_name] = candidates
            raw_counts[pool_name] = len(candidates)

        filtering = CandidateFilteringLayer(graph)
        merged, report = filtering.run(pool_candidates)

        # Graceful degradation: the strict ZPD band can legitimately empty
        # out every candidate for a real user (not just a hypothetical
        # edge case) -- e.g. real-but-still-low mastery recommended against
        # a catalog whose difficulty floor sits above what the ZPD band
        # tolerates for that mastery level. Retrying with apply_zpd=False
        # falls back to the pre-ZPD, still solved/locked/duplicate-filtered
        # candidates rather than surfacing zero recommendations. Only
        # engages when the raw pool candidates were non-empty but the ZPD
        # pass alone wiped them out -- a genuinely cold user with zero raw
        # candidates gets an (already correct) empty result either way.
        if not merged and any(pool_candidates.values()):
            merged, report = filtering.run(pool_candidates, apply_zpd=False)

        return PoolGenerationResult(
            merged_candidates=merged,
            filter_report=report,
            difficulty_plan=plan,
            raw_counts=raw_counts,
            requested_counts=requested_counts,
        )


def generate_recommendations(graph: UserGraph, state: UserStateVector,
                              qdrant=None, collection: str = "problems_full",
                              total_n: int = 30) -> PoolGenerationResult:
    """Convenience one-shot wrapper for callers that don't need to reuse the orchestrator."""
    return PoolGenerationOrchestrator(qdrant=qdrant, collection=collection).generate(
        graph, state, total_n=total_n)

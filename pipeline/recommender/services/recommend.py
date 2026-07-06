"""
The single callable entry point for the entire online recommendation engine.

This is what a controller calls. One function in, one ranked list out:

    get_recommendations(user_id, db, redis, qdrant, bkt_store, hlr_store)
        --> UserGraphService.get()/new_user_graph()   (build/fetch the graph)
        --> UserStateBuilder.build()                   (1920-d state vector)
        --> PoolGenerationOrchestrator.generate()       (7 pools -> filtered candidates)
        --> CandidateFilteringLayer.to_ranker_input()   (flatten for ranking)
        --> CandidateStore.save()                       (stage before ranking)
        --> HeuristicRanker.rank()                       (score every candidate)
        --> DiversityMixer.mix()                         (pool/topic spread on final slate)
        --> top 10 (or however many were asked for)

CRITICAL: this module NEVER calls anything from the OFFLINE pipeline
(pipeline/ingestion, pipeline/embeddings, pipeline/graphs). Ingestion,
embedding generation, RGCN training, and the RGCN->Qdrant ingest are batch
jobs that run once (or on a periodic schedule) to populate Qdrant -- they
are NOT part of the per-request online path. This function only ever READS
from Qdrant (via query_points/scroll, through the pools) and never writes
to it, never re-embeds, never re-trains. Calling get_recommendations() a
thousand times in a row costs zero additional ingestion/embedding work --
it only issues Qdrant reads and in-memory computation.

This separation is enforced structurally (no import of ingestion/embeddings/
graphs modules anywhere below), not just by convention -- see
tests/test_recommend.py::TestNoOfflinePipelineCalls for a static check on
this file's own import list, which fails loudly if that boundary is ever
crossed by an unwary future edit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from pipeline.recommender.models.user_graph import UserGraph
from pipeline.recommender.models.user_state import UserStateBuilder, UserStateVector
from pipeline.recommender.services.user_graph_service import UserGraphService
from pipeline.recommender.services.pool_generation import PoolGenerationOrchestrator
from pipeline.recommender.services.candidate_filtering import CandidateFilteringLayer
from pipeline.recommender.services.candidate_store import CandidateStore, InMemoryCandidateStore
from pipeline.recommender.services.heuristic_ranker import HeuristicRanker
from pipeline.recommender.services.diversity_mixer import DiversityMixer

log = logging.getLogger(__name__)

# Module-level default store. A controller can inject its own (e.g. a
# Postgres-backed CandidateStore once the backend team provisions
# CandidateStaging) via the `store=` parameter; this default exists so
# get_recommendations() works out of the box with zero setup.
_DEFAULT_STORE = InMemoryCandidateStore()


@dataclass
class RecommendationResult:
    """Everything a controller needs to respond to a 'give me questions' request."""
    user_id:           str
    recommendations:   list                    # top-k dicts, ranked + diversified
    difficulty_plan:   dict
    filter_report:     dict
    candidate_set_id:  Optional[str] = None
    is_cold_start:     bool = False

    def to_dict(self) -> dict:
        return {
            "user_id":          self.user_id,
            "recommendations":  self.recommendations,
            "difficulty_plan":  self.difficulty_plan,
            "filter_report":    self.filter_report,
            "candidate_set_id": self.candidate_set_id,
            "is_cold_start":    self.is_cold_start,
        }


def get_recommendations(
    user_id: str,
    db=None, redis=None, qdrant=None,
    bkt_store: dict = None, hlr_store: dict = None,
    collection: str = "problems_full",
    total_n: int = 30,
    k: int = 10,
    max_per_pool: int = 3,
    max_per_topic: int = 3,
    store: CandidateStore = None,
    relevance_scores: dict = None,
) -> RecommendationResult:
    """
    Single callable for the whole online recommendation pipeline.

    Args:
        user_id:    who to recommend for
        db:         Postgres session/connection (None is fine for a brand
                    new user -- falls back to new_user_graph())
        redis:      Redis client for the graph cache (None disables caching)
        qdrant:     Qdrant client (required for any real candidates; pools
                    return empty lists gracefully if None)
        bkt_store:  Shraddha's {user_id: {topic: mastery}} dict
        hlr_store:  Shraddha's {user_id: {topic: hlr_state}} dict
        total_n:    how many raw candidates to request across all 7 pools
                    before filtering (see MAX_TOTAL_CANDIDATES ceiling)
        k:          final slate size (default 10)
        max_per_pool / max_per_topic: diversity mixer caps
        store:      CandidateStore to stage the filtered set in before
                    ranking (defaults to a shared in-memory store)
        relevance_scores: optional {problem_id: score} from a real trained
                    ranker, if one exists -- overrides the heuristic ranker's
                    own scores when present, without changing anything else
                    in this function's structure

    Returns a RecommendationResult with up to k ranked, diversified problems.
    """
    store = store or _DEFAULT_STORE

    graph = _get_graph(user_id, db, redis, bkt_store, hlr_store)
    state = _build_state(graph, qdrant)

    orchestrator = PoolGenerationOrchestrator(qdrant=qdrant, collection=collection)
    gen_result = orchestrator.generate(graph, state, total_n=total_n)

    filtering = CandidateFilteringLayer(graph)
    ranker_rows = filtering.to_ranker_input(gen_result.merged_candidates)

    staged = store.save(user_id, ranker_rows, gen_result.difficulty_plan.to_dict())

    ranker = HeuristicRanker()
    ranked_rows = ranker.top_k(ranker_rows, k=max(k * 3, k))   # over-fetch for the mixer to diversify from

    # DiversityMixer works on MergedCandidate-shaped objects; ranked_rows are
    # plain dicts (post-ranker). Re-attach onto the original MergedCandidate
    # objects so pool_sources/topic_tags survive into the mixer, using the
    # ranker's score as the relevance signal it mixes against.
    by_id = {mc.problem_id: mc for mc in gen_result.merged_candidates}
    scores_by_id = relevance_scores or {row["problem_id"]: row["rank_score"] for row in ranked_rows}

    ordered_candidates = [by_id[row["problem_id"]] for row in ranked_rows if row["problem_id"] in by_id]

    mixer = DiversityMixer(max_per_pool=max_per_pool, max_per_topic=max_per_topic)
    final_candidates = mixer.mix(ordered_candidates, k=k, relevance_scores=scores_by_id)

    final_rows = []
    rows_by_id = {row["problem_id"]: row for row in ranked_rows}
    for mc in final_candidates:
        row = dict(rows_by_id.get(mc.problem_id, {"problem_id": mc.problem_id}))
        final_rows.append(row)

    return RecommendationResult(
        user_id=user_id,
        recommendations=final_rows,
        difficulty_plan=gen_result.difficulty_plan.to_dict(),
        filter_report=gen_result.filter_report.to_dict(),
        candidate_set_id=staged.set_id,
        is_cold_start=state.is_cold_start if state else True,
    )


# ------------------------------------------------------------------ helpers

def _get_graph(user_id: str, db, redis, bkt_store, hlr_store) -> UserGraph:
    """
    Fetch the user's graph. ALWAYS goes through UserGraphService.get() first
    -- which checks the Redis cache before ever touching the database -- so
    a graph mutated by StateUpdateService.process_submission() (cached via
    _to_cache) is actually seen here, even when db=None.

    Only falls back to a brand-new cold-start graph if the cache is ALSO
    empty (genuinely the user's first-ever call, or the ValueError/
    AttributeError path when db=None and there's nothing cached yet).

    The previous version short-circuited to a fresh graph whenever db was
    None, WITHOUT checking the cache at all -- meaning any state update
    written between calls was silently discarded every time db wasn't
    wired up, which is exactly the common case before a real Postgres
    session exists.
    """
    service = UserGraphService(db=db, redis=redis,
                               bkt=bkt_store or {}, hlr=hlr_store or {})
    try:
        return service.get(user_id)
    except (ValueError, AttributeError):
        return service.new_user_graph(user_id)


def _build_state(graph: UserGraph, qdrant) -> Optional[UserStateVector]:
    if qdrant is None:
        return None
    try:
        return UserStateBuilder(qdrant_client=qdrant).build(graph)
    except Exception as exc:
        log.warning("State vector build failed, continuing without it: %s", exc)
        return None
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
        --> resolve problem_id -> real title/tags from Qdrant
        --> map pool provenance -> RecommendationLog.source enum
        --> top 10, shaped to match the backend's actual schema

OUTPUT SCHEMA -- matches the backend's tables, nothing else:
    Each recommendation returned is exactly the fields that map onto
    RecommendationLog (problem_id, source, recommended_at) plus the Problem
    display fields a client needs to render it (title, title_slug,
    difficulty_score, topic_tags). Internal-only scoring detail --
    rank_components, avg_mastery, max_urgency, predicted_success,
    best_pool_score, pool_count, the full difficulty_plan, filter_report --
    is used to COMPUTE the ranking but is deliberately NOT included in the
    returned payload. It was never part of the backend's contract; it was
    debug output that happened to leak into the response shape.

    RecommendationLog.source is a 5-value enum (graph_walk / vector_similarity
    / revision / sm2_review / course_path) -- narrower than our 7 internal
    pool names (A, B_C, D, E, F, G, vector). _POOL_TO_SOURCE below maps each
    pool to the closest matching enum value; when a candidate came from more
    than one pool, the highest-priority pool's mapping wins (course_path and
    sm2_review take precedence as the most specific/actionable signals).
    This mapping is a best-effort interpretation of the schema doc's five
    values -- confirm it against the backend's actual source of truth if a
    stricter mapping is needed.

CRITICAL: this module NEVER calls anything from the OFFLINE pipeline
(pipeline/ingestion, pipeline/embeddings, pipeline/graphs). Ingestion,
embedding generation, RGCN training, and the RGCN->Qdrant ingest are batch
jobs that run once (or on a periodic schedule) to populate Qdrant -- they
are NOT part of the per-request online path. This function only ever READS
from Qdrant (via query_points/scroll/retrieve, through the pools and the
title-resolution step) and never writes to it, never re-embeds, never
re-trains. Calling get_recommendations() a thousand times in a row costs
zero additional ingestion/embedding work.

This separation is enforced structurally (no import of ingestion/embeddings/
graphs modules anywhere below), not just by convention -- see
tests/test_recommend.py::TestNoOfflinePipelineCalls for a static check on
this file's own import list, which fails loudly if that boundary is ever
crossed by an unwary future edit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from pipeline.recommender.models.user_graph import UserGraph
from pipeline.recommender.models.user_state import UserStateBuilder, UserStateVector
from pipeline.recommender.services.user_graph_service import UserGraphService
from pipeline.recommender.services.pool_generation import PoolGenerationOrchestrator
from pipeline.recommender.services.candidate_filtering import CandidateFilteringLayer
from pipeline.recommender.services.candidate_store import CandidateStore, InMemoryCandidateStore
from pipeline.recommender.services.lightgbm_ranker import rank_candidates
from pipeline.recommender.services.diversity_mixer import DiversityMixer

log = logging.getLogger(__name__)

# Module-level default store. A controller can inject its own (e.g. a
# Postgres-backed CandidateStore once the backend team provisions
# CandidateStaging) via the `store=` parameter; this default exists so
# get_recommendations() works out of the box with zero setup.
_DEFAULT_STORE = InMemoryCandidateStore()

# Maps our 7 internal pool names to the backend's RecommendationLog.source
# enum (graph_walk / vector_similarity / revision / sm2_review /
# course_path -- per the KNode schema doc). Order matters: when a candidate
# came from multiple pools, the FIRST matching pool in this priority list
# wins, most-specific/actionable signal first.
_SOURCE_PRIORITY = ["E", "A", "D", "vector", "B_C", "F", "G"]
_POOL_TO_SOURCE = {
    "A":      "course_path",       # course path pool -> direct match
    "E":      "sm2_review",        # spaced review (SM-2/HLR) -> direct match
    "vector": "vector_similarity", # pure ANN -> direct match
    "B_C":    "graph_walk",        # near/far transfer -- graph-distance based per the architecture doc
    "D":      "revision",          # weakness recovery -- closest in spirit to revising weak areas
    "F":      "course_path",       # stretch -- forward progression along the learning path
    "G":      "graph_walk",        # novelty -- walks cc_edges from mastered concepts to find new ones
}


def _pool_sources_to_backend_source(pool_sources: list) -> str:
    for pool in _SOURCE_PRIORITY:
        if pool in pool_sources:
            return _POOL_TO_SOURCE[pool]
    return "course_path"   # should be unreachable; safe fallback


def _resolve_titles(problem_ids: list, qdrant, collection: str) -> dict:
    """
    Resolve problem_id strings -> Qdrant payload (title, title_slug, etc).

    Uses scroll() with a problem_id payload filter instead of retrieve()
    by point ID. The internal Qdrant point IDs are raw integers assigned
    at ingest time -- not the xxhash values _stable_point_id() produces.
    retrieve() by hash ID always returns nothing. scroll() with a keyword
    filter on the problem_id payload field is the correct approach since
    problem_id is now indexed (after create_qdrant_indexes.py was run).
    """
    if not problem_ids or qdrant is None:
        return {}
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        points, _ = qdrant.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=[
                FieldCondition(
                    key="problem_id",
                    match=MatchAny(any=problem_ids),
                )
            ]),
            limit=len(problem_ids),
            with_payload=True,
            with_vectors=False,
        )
        return {
            p.payload["problem_id"]: p.payload
            for p in points
            if p.payload and "problem_id" in p.payload
        }
    except Exception as exc:
        log.warning("Title resolution failed: %s", exc)
        return {}


@dataclass
class RecommendationResult:
    """
    Everything a controller needs to respond to a 'give me questions'
    request -- shaped to match the backend's schema, not our internal
    scoring detail. recommendations is a list of dicts, each containing
    ONLY: problem_id, title, title_slug, difficulty_score, topic_tags,
    source, recommended_at.
    """
    user_id:           str
    recommendations:   list
    candidate_set_id:  Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "user_id":          self.user_id,
            "recommendations":  self.recommendations,
        }


def get_recommendations(
    user_id: str,
    db=None, redis=None, neo4j=None, qdrant=None,
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
        neo4j:      Neo4jGraphStore instance for durable graph persistence
                    beyond the Redis cache TTL (None disables it -- the
                    graph still works, just isn't mirrored durably)
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

    Returns a RecommendationResult whose .recommendations list matches the
    backend's schema exactly -- problem_id, title, title_slug,
    difficulty_score, topic_tags, source, recommended_at. Nothing else.
    """
    store = store or _DEFAULT_STORE

    graph = _get_graph(user_id, db, redis, neo4j, bkt_store, hlr_store)
    state = _build_state(graph, qdrant)

    orchestrator = PoolGenerationOrchestrator(qdrant=qdrant, collection=collection)
    gen_result = orchestrator.generate(graph, state, total_n=total_n)

    # Cold-start users get simpler, single-concept problems -- both to keep
    # the slate legible (a brand-new user seeing "dp, string, two_pointers,
    # tabulation" on their very first recommendation is confusing, not
    # helpful) and because it fixes a real interaction with DiversityMixer's
    # max_per_topic cap below: _within_caps rejects a candidate if ANY of
    # its tags is already at cap, so candidates carrying 3-5 tags each
    # (near-universal in this catalog) exhaust a shared tag's budget after
    # just 2-3 picks and silently starve the slate well short of k, even
    # when plenty of relevant candidates remain. Restricting to <=2 tags
    # for cold start keeps each pick "burning" at most 2 tag-budgets
    # instead of 3-5, so the requested k is actually reachable. Falls back
    # to the unfiltered list if this would empty it out entirely (a
    # genuinely tag-sparse catalog shouldn't return nothing).
    if gen_result.difficulty_plan.is_cold_start:
        # Progressive relaxation (<=2 -> <=3 -> unfiltered): this catalog's
        # entries mostly carry 3-5 tags each, so a strict <=2 cutoff alone
        # can leave too few candidates for a given topic/pool combination.
        # Prefer the simplest available slate rather than an empty or
        # single-candidate one.
        for max_tags in (2, 3):
            simple = [mc for mc in gen_result.merged_candidates if len(mc.topic_tags or []) <= max_tags]
            if len(simple) >= k:
                gen_result.merged_candidates = simple
                break

    filtering = CandidateFilteringLayer(graph)
    ranker_rows = filtering.to_ranker_input(gen_result.merged_candidates)

    staged = store.save(user_id, ranker_rows, gen_result.difficulty_plan.to_dict())

    # rank_candidates() picks heuristic / lightgbm / hybrid per the RANKER
    # env var (defaults to heuristic -- unchanged behaviour with no
    # configuration) and always returns the same shape HeuristicRanker.top_k()
    # does, falling back to heuristic automatically if the ML model is
    # missing/mismatched/fails -- see pipeline/recommender/services/lightgbm_ranker.py.
    ranked_rows = rank_candidates(
        ranker_rows=ranker_rows,
        merged_candidates=gen_result.merged_candidates,
        graph=graph, plan=gen_result.difficulty_plan,
        k=max(k * 3, k),   # over-fetch for the mixer to diversify from
    )

    # DiversityMixer works on MergedCandidate-shaped objects; ranked_rows are
    # plain dicts (post-ranker). Re-attach onto the original MergedCandidate
    # objects so pool_sources/topic_tags survive into the mixer, using the
    # ranker's score as the relevance signal it mixes against.
    by_id = {mc.problem_id: mc for mc in gen_result.merged_candidates}
    scores_by_id = relevance_scores or {row["problem_id"]: row["rank_score"] for row in ranked_rows}

    ordered_candidates = [by_id[row["problem_id"]] for row in ranked_rows if row["problem_id"] in by_id]

    # Pool diversity ("no single pool dominates the slate") only means
    # something when several pools actually have candidates. A cold-start
    # user typically only gets real output from 1-2 pools (course_path +
    # novelty -- weakness/review/stretch/vector all need history that
    # doesn't exist yet), so the default max_per_pool=3 caps the ONLY
    # pools with data down to a handful of picks and starves the slate
    # well short of k even when plenty of relevant candidates remain.
    # Relaxing the pool cap to k for cold start doesn't defeat diversity --
    # there's nothing to diversify AWAY from when only one pool is
    # contributing anyway.
    effective_max_per_pool = k if gen_result.difficulty_plan.is_cold_start else max_per_pool

    mixer = DiversityMixer(max_per_pool=effective_max_per_pool, max_per_topic=max_per_topic)
    final_candidates = mixer.mix(ordered_candidates, k=k, relevance_scores=scores_by_id)

    # Resolve real problem details from Qdrant -- the backend/frontend needs
    # actual titles, not opaque problem_id hashes.
    final_ids = [mc.problem_id for mc in final_candidates]
    payloads = _resolve_titles(final_ids, qdrant, collection)

    now_iso = datetime.now(timezone.utc).isoformat()
    final_rows = []
    for mc in final_candidates:
        payload = payloads.get(mc.problem_id, {})
        final_rows.append({
            "problem_id":       mc.problem_id,
            "title":            payload.get("title"),
            "title_slug":       payload.get("title_slug"),
            "difficulty_score": mc.difficulty_score,
            "topic_tags":       mc.topic_tags,
            "source":           _pool_sources_to_backend_source(mc.pool_sources),
            "recommended_at":   now_iso,
        })

    return RecommendationResult(
        user_id=user_id,
        recommendations=final_rows,
        candidate_set_id=staged.set_id,
    )


# ------------------------------------------------------------------ helpers

def _get_graph(user_id: str, db, redis, neo4j, bkt_store, hlr_store) -> UserGraph:
    """
    Fetch the user's graph. ALWAYS goes through UserGraphService.get() first
    -- which checks Redis, then Neo4j (durable, no TTL), before ever
    touching the database -- so a graph mutated by
    StateUpdateService.apply_update() is actually seen here, even
    when db=None.

    Only falls back to a brand-new cold-start graph if every tier is
    empty (genuinely the user's first-ever call, or the ValueError/
    AttributeError path when db=None and nothing is cached anywhere yet).
    """
    service = UserGraphService(db=db, redis=redis, neo4j=neo4j,
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

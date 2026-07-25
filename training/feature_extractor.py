"""
training/feature_extractor.py

Populates every column declared in training/feature_registry.py for one
(UserGraph, DifficultyPlan, MergedCandidate) triple, by calling directly
into the existing recommender objects -- no recommendation formula is
reimplemented here except difficulty_alignment, which is re-derived because
HeuristicRanker.score_one() never persists it on RankedCandidate (see that
function's docstring note). The re-derivation below is a byte-identical
copy of that one local computation, not a new formula.

Column names produced here match training/feature_registry.py exactly --
see ExtractedFeatures.to_flat_dict().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pipeline.recommender.models.user_graph import EdgeType, UserGraph
from pipeline.recommender.services.adaptive_difficulty import DifficultyPlan
from pipeline.recommender.services.candidate_filtering import (
    CandidateFilteringLayer,
    MergedCandidate,
)
from pipeline.recommender.services.heuristic_ranker import HeuristicRanker, RankedCandidate


# ============================================================================
# Feature-group dataclasses -- field names match the registry's feature names.
# ============================================================================

@dataclass(frozen=True)
class UserFeatures:
    cold_start: bool
    seeded_user: bool
    user_level: str
    solved_count: int
    average_mastery: float
    mastery_variance: float
    mastered_topic_count: int
    weak_topic_count: int
    urgent_topic_count: int
    overdue_review_count: int


@dataclass(frozen=True)
class GraphFeatures:
    # None when the graph has zero PREREQ edges at all -- undefined, not 0%.
    prerequisite_completion_ratio: Optional[float]


@dataclass(frozen=True)
class ProblemFeatures:
    difficulty_score: Optional[float]
    topic_tag_count: int
    # native NaN (None) unless catalog_metadata was supplied -- these are
    # not in the live Qdrant payload (registry: inference_available=False).
    company_tag_count: Optional[float]
    frequency: Optional[float]
    rating: Optional[float]
    asked_by_faang: Optional[bool]


@dataclass(frozen=True)
class PairFeatures:
    predicted_success: Optional[float]
    avg_mastery_pair: Optional[float]
    max_urgency_pair: Optional[float]
    proximity: float
    pool_agreement: float
    urgency_boost: float
    similarity_score: float
    difficulty_alignment: float
    mastery_difficulty_gap: Optional[float]
    best_pool_score: float


@dataclass(frozen=True)
class PoolFeatures:
    pool_count: int
    from_pool_A: bool
    from_pool_B_C: bool
    from_pool_D: bool
    from_pool_E: bool
    from_pool_F: bool
    from_pool_G: bool
    from_pool_vector: bool
    max_pool_weight: float


@dataclass(frozen=True)
class ExtractedFeatures:
    """The complete feature vector for one (User State, Candidate Problem)
    pair, plus the identifier/diagnostic columns the registry declares
    alongside it."""
    query_id: Optional[str]
    candidate_id: str
    user_id: str
    recommended_at: Optional[float]
    current_heuristic_rank_score: float

    user:    UserFeatures
    problem: ProblemFeatures
    pair:    PairFeatures
    pool:    PoolFeatures
    graph:   GraphFeatures

    def to_flat_dict(self) -> dict:
        """One row, column names matching feature_registry.py exactly."""
        return {
            "query_id": self.query_id,
            "candidate_id": self.candidate_id,
            "user_id": self.user_id,
            "recommended_at": self.recommended_at,
            "current_heuristic_rank_score": self.current_heuristic_rank_score,
            "cold_start": self.user.cold_start,
            "seeded_user": self.user.seeded_user,
            "user_level": self.user.user_level,
            "solved_count": self.user.solved_count,
            "average_mastery": self.user.average_mastery,
            "mastery_variance": self.user.mastery_variance,
            "mastered_topic_count": self.user.mastered_topic_count,
            "weak_topic_count": self.user.weak_topic_count,
            "urgent_topic_count": self.user.urgent_topic_count,
            "overdue_review_count": self.user.overdue_review_count,
            "prerequisite_completion_ratio": self.graph.prerequisite_completion_ratio,
            "difficulty_score": self.problem.difficulty_score,
            "topic_tag_count": self.problem.topic_tag_count,
            "company_tag_count": self.problem.company_tag_count,
            "frequency": self.problem.frequency,
            "rating": self.problem.rating,
            "asked_by_faang": self.problem.asked_by_faang,
            "predicted_success": self.pair.predicted_success,
            "avg_mastery_pair": self.pair.avg_mastery_pair,
            "max_urgency_pair": self.pair.max_urgency_pair,
            "proximity": self.pair.proximity,
            "pool_agreement": self.pair.pool_agreement,
            "urgency_boost": self.pair.urgency_boost,
            "similarity_score": self.pair.similarity_score,
            "difficulty_alignment": self.pair.difficulty_alignment,
            "mastery_difficulty_gap": self.pair.mastery_difficulty_gap,
            "best_pool_score": self.pair.best_pool_score,
            "pool_count": self.pool.pool_count,
            "from_pool_A": self.pool.from_pool_A,
            "from_pool_B_C": self.pool.from_pool_B_C,
            "from_pool_D": self.pool.from_pool_D,
            "from_pool_E": self.pool.from_pool_E,
            "from_pool_F": self.pool.from_pool_F,
            "from_pool_G": self.pool.from_pool_G,
            "from_pool_vector": self.pool.from_pool_vector,
            "max_pool_weight": self.pool.max_pool_weight,
        }


# ============================================================================
# Per-group extraction -- each function only reads existing recommender
# objects; none recomputes a formula the recommender already owns.
# ============================================================================

def extract_user_features(graph: UserGraph, plan: DifficultyPlan) -> UserFeatures:
    """
    All fields reused directly from UserGraph/DifficultyPlan except
    seeded_user and mastery_variance, which are new aggregations over
    data both objects already hold (no new query, no new graph traversal).
    """
    mastery_scores = [e.mastery_score for e in graph.concept_edges.values()]
    if mastery_scores:
        mean = sum(mastery_scores) / len(mastery_scores)
        variance = sum((s - mean) ** 2 for s in mastery_scores) / len(mastery_scores)
    else:
        variance = 0.0

    # Same condition the seeded-user cold-start bugfix (adaptive_difficulty.py)
    # targeted: real concept_edges (seeded mastery/HLR) but zero platform
    # solved_ids (solved_ids only ever populated from the `submission` table).
    seeded = len(graph.concept_edges) > 0 and len(graph.solved_ids) == 0

    return UserFeatures(
        cold_start=plan.is_cold_start,
        seeded_user=seeded,
        user_level=plan.level,
        solved_count=len(graph.solved_ids),
        average_mastery=round(plan.avg_mastery, 6),
        mastery_variance=round(variance, 6),
        mastered_topic_count=len(graph.mastered_concepts()),
        weak_topic_count=plan.n_weak,
        urgent_topic_count=plan.n_urgent,
        overdue_review_count=plan.n_overdue,
    )


def extract_graph_features(graph: UserGraph) -> GraphFeatures:
    """
    prerequisite_completion_ratio: mastered-source PREREQ edges / total
    PREREQ edges in graph.cc_edges. A forward scan over cc_edges (source ->
    targets it unlocks), which is the opposite direction from
    UserGraph._prereq_index()'s cached REVERSE index (target -> required
    prereqs, built for is_locked()'s lookup) -- that cache isn't the right
    shape for this ratio, so this is a fresh (not duplicated) aggregation
    over cc_edges, a field the graph already loaded.
    """
    mastered = set(graph.mastered_concepts())
    total = 0
    completed = 0
    for source_slug, edges in graph.cc_edges.items():
        for edge in edges:
            if edge.edge_type != EdgeType.PREREQ:
                continue
            total += 1
            if source_slug in mastered:
                completed += 1

    ratio = round(completed / total, 6) if total > 0 else None
    return GraphFeatures(prerequisite_completion_ratio=ratio)


def extract_problem_features(
    ranker_row: dict,
    catalog_metadata: Optional[dict] = None,
) -> ProblemFeatures:
    """
    ranker_row is one row of CandidateFilteringLayer.to_ranker_input()'s
    output -- difficulty_score/topic_tags are read from there, not
    recomputed. catalog_metadata is an optional offline-ingestion-schema-
    shaped dict (pipeline/ingestion/schema.py's RELATIONAL_COLUMNS/
    SIGNAL_COLUMNS) for the fields not present in the live Qdrant payload;
    None (native NaN) for all of them if not supplied, matching the
    registry's inference_available=False flag for this group.
    """
    topic_tags = ranker_row.get("topic_tags") or []
    meta = catalog_metadata or {}

    companies = meta.get("companies")
    company_tag_count = float(len(companies)) if companies is not None else None

    return ProblemFeatures(
        difficulty_score=ranker_row.get("difficulty_score"),
        topic_tag_count=len(topic_tags),
        company_tag_count=company_tag_count,
        frequency=meta.get("frequency"),
        rating=meta.get("rating"),
        asked_by_faang=meta.get("asked_by_faang"),
    )


def _difficulty_alignment(difficulty: Optional[float], avg_mastery: Optional[float]) -> float:
    """
    Byte-identical to the local computation inside
    HeuristicRanker.score_one() -- re-derived here (not imported) only
    because that value is never attached to RankedCandidate/persisted by
    to_ranker_input(). Must be kept in sync if score_one()'s formula ever
    changes; not an independent formula.
    """
    if difficulty is None or avg_mastery is None:
        return 0.0
    return 1.0 - min(1.0, abs(difficulty - avg_mastery))


def extract_pair_features(ranker_row: dict, ranked: RankedCandidate) -> PairFeatures:
    """
    proximity/pool_agreement/urgency_boost/similarity_score come directly
    from HeuristicRanker.score_one()'s own RankedCandidate output -- not
    recomputed. difficulty_alignment is re-derived (see _difficulty_alignment
    docstring). mastery_difficulty_gap is the one new engineered feature
    here (signed gap the existing symmetric difficulty_alignment can't
    express), reusing the same two already-computed inputs at zero extra cost.
    """
    difficulty = ranker_row.get("difficulty_score")
    avg_mastery = ranker_row.get("avg_mastery")

    gap = None
    if difficulty is not None and avg_mastery is not None:
        gap = round(avg_mastery - difficulty, 6)

    return PairFeatures(
        predicted_success=ranker_row.get("predicted_success"),
        avg_mastery_pair=avg_mastery,
        max_urgency_pair=ranker_row.get("max_urgency"),
        proximity=round(ranked.proximity_score, 6),
        pool_agreement=round(ranked.pool_agreement, 6),
        urgency_boost=round(ranked.urgency_boost, 6),
        similarity_score=round(ranked.similarity_score, 6),
        difficulty_alignment=round(_difficulty_alignment(difficulty, avg_mastery), 6),
        mastery_difficulty_gap=gap,
        best_pool_score=ranker_row.get("best_pool_score", 0.0),
    )


def extract_pool_features(candidate: MergedCandidate, plan: DifficultyPlan) -> PoolFeatures:
    """
    pool_count/pool_sources reused directly from MergedCandidate.
    from_pool_* booleans are the standard one-hot encoding of the same
    pool_sources list already on the candidate (not a new computation --
    see feature_registry.py's from_pool_* engineering_justification).
    max_pool_weight reuses DifficultyPlan.weight_of(), already computed
    once per user by AdaptiveDifficultyController.
    """
    sources = set(candidate.pool_sources)
    weights = [plan.weight_of(p) for p in candidate.pool_sources]
    return PoolFeatures(
        pool_count=candidate.pool_count,
        from_pool_A="A" in sources,
        from_pool_B_C="B_C" in sources,
        from_pool_D="D" in sources,
        from_pool_E="E" in sources,
        from_pool_F="F" in sources,
        from_pool_G="G" in sources,
        from_pool_vector="vector" in sources,
        max_pool_weight=round(max(weights), 6) if weights else 0.0,
    )


# ============================================================================
# Umbrella entry point.
# ============================================================================

def extract_features(
    graph: UserGraph,
    plan: DifficultyPlan,
    candidate: MergedCandidate,
    *,
    ranker: Optional[HeuristicRanker] = None,
    filtering_layer: Optional[CandidateFilteringLayer] = None,
    catalog_metadata: Optional[dict] = None,
    query_id: Optional[str] = None,
    recommended_at: Optional[float] = None,
) -> ExtractedFeatures:
    """
    Full feature vector for one (UserGraph, DifficultyPlan, MergedCandidate)
    triple -- reuses CandidateFilteringLayer.to_ranker_input() and
    HeuristicRanker.score_one() exactly as the live pipeline does.

    If candidate.predicted_success is still None (i.e. it was built without
    going through CandidateFilteringLayer.run()'s ZPD pass), this computes
    it via the layer's own _default_success_estimator -- the same call
    _apply_zpd_filter makes -- so this function is usable standalone in
    tests without requiring the full pipeline to have run first.
    """
    ranker = ranker or HeuristicRanker()
    layer = filtering_layer or CandidateFilteringLayer(graph)

    if candidate.predicted_success is None:
        candidate.predicted_success = layer._default_success_estimator(candidate, graph)

    row = layer.to_ranker_input([candidate])[0]
    ranked = ranker.score_one(row)

    return ExtractedFeatures(
        query_id=query_id,
        candidate_id=candidate.problem_id,
        user_id=graph.user.user_id,
        recommended_at=recommended_at,
        current_heuristic_rank_score=round(ranked.score, 6),
        user=extract_user_features(graph, plan),
        problem=extract_problem_features(row, catalog_metadata),
        pair=extract_pair_features(row, ranked),
        pool=extract_pool_features(candidate, plan),
        graph=extract_graph_features(graph),
    )

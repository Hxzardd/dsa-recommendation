"""
pipeline/recommender/services/topic_recommend.py

Single best-next-topic selection -- "what ONE topic should this user work
on next", independent of full problem-level /recommend. Reuses the same
UserGraph everything else reads (mastered/urgent/weak concepts, cc_edges
PREREQ/COOCCURS), just picks a topic slug instead of a candidate list.

Priority (momentum first, then recovery, then progression, then explore --
mirrors the pool priority order in recommend.py's _SOURCE_PRIORITY):
  1. Most urgent topic due for spaced review (forgetting curve)
  2. In-progress topic closest to mastery (finish what's nearly done)
  3. Newly unlocked topic (its prereq was just mastered, not started yet)
  4. Novel topic reachable from a mastered concept (same reachability
     NoveltyPool uses)
  5. Cold start: no graph data at all yet -- first STARTER_CONCEPTS entry
"""

from __future__ import annotations

import math

from pipeline.recommender.models.user_graph import UserGraph, EdgeType
from pipeline.recommender.pools.pools import STARTER_CONCEPTS

# Same productive-struggle band candidate_filtering.py's ZPD filter uses --
# kept identical here (not imported) so this module has no dependency on
# Shraddha's filtering layer; the two are independently free to tune.
_ZPD_OPTIMAL = 0.68


def recommend_topic(graph: UserGraph) -> tuple[str, str]:
    """
    Returns (topic_id, reason). reason is one of:
    "spaced_review", "in_progress", "unlocked", "novelty", "cold_start".
    """
    urgent = graph.urgent_concepts()
    if urgent:
        best = max(urgent, key=lambda s: graph.concept_edges[s].urgency)
        return best, "spaced_review"

    mastered = set(graph.mastered_concepts())

    in_progress = [
        (s, e.mastery_score) for s, e in graph.concept_edges.items()
        if s not in mastered
    ]
    if in_progress:
        best = max(in_progress, key=lambda pair: pair[1])[0]
        return best, "in_progress"

    seen = set(graph.concept_edges.keys())
    unlocked = []
    for src, edges in graph.cc_edges.items():
        if src not in mastered:
            continue
        for e in edges:
            if e.edge_type == EdgeType.PREREQ and e.target_slug not in seen:
                unlocked.append(e.target_slug)
    if unlocked:
        return unlocked[0], "unlocked"

    novel = []
    for src, edges in graph.cc_edges.items():
        if src not in mastered:
            continue
        for e in edges:
            if e.edge_type == EdgeType.COOCCURS and e.target_slug not in seen:
                novel.append(e.target_slug)
    if novel:
        return novel[0], "novelty"

    return STARTER_CONCEPTS[0], "cold_start"


def _predicted_success(mastery: float, difficulty: float) -> float:
    """Same sigmoid candidate_filtering.py's _default_success_estimator
    uses -- mastery == difficulty gives ~0.5, higher mastery relative to
    difficulty pushes success upward. Kept as a small local copy (see
    module docstring) rather than importing Shraddha's filtering layer."""
    delta = (mastery - difficulty) * 6.0
    return 1.0 / (1.0 + math.exp(-delta))


def recommend_problems_for_topic(
    graph: UserGraph,
    qdrant,
    collection: str,
    topic_id: str,
    n: int = 10,
) -> list[dict]:
    """
    "Backend's demand" endpoint: caller already knows which topic they want
    (topic_id) -- this returns problems FOR THAT TOPIC ranked by relevance
    to the user's own level, not a fixed difficulty band. Relevance =
    predicted_success closest to the ZPD sweet spot (_ZPD_OPTIMAL), using
    the user's real current mastery on this exact topic (graph.concept_mastery,
    the same BKT P(L) everything else reads) against each candidate's real
    Qdrant difficulty_score -- so a 0.25-mastery user and a 0.75-mastery
    user asking for the SAME topic get genuinely different, level-
    appropriate problems, not the same list.

    Respects the same prereq gating as every pool (graph.is_locked) and
    never re-serves solved/deprioritised problems.
    """
    if not qdrant or not topic_id:
        return []

    from qdrant_client.models import Filter, FieldCondition, MatchAny

    exclude = set(graph.solved_ids) | set(graph.deprioritised_ids)
    mastery = graph.concept_mastery(topic_id)

    try:
        points, _ = qdrant.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="topic_tags", match=MatchAny(any=[topic_id])),
            ]),
            limit=max(n * 10, 100), with_payload=True, with_vectors=False,
        )
    except Exception:
        return []

    candidates = []
    for p in points:
        pl = p.payload or {}
        pid = str(pl.get("problem_id", p.id))
        if pid in exclude:
            continue
        topic_tags = pl.get("topic_tags") or []
        if graph.is_locked(topic_tags):
            continue
        difficulty = pl.get("difficulty_score")
        if difficulty is None:
            continue
        predicted_success = _predicted_success(mastery, difficulty)
        candidates.append({
            "problem_id": pid,
            "topic_tags": topic_tags,
            "difficulty_score": difficulty,
            "predicted_success": round(predicted_success, 4),
        })

    candidates.sort(key=lambda c: abs(c["predicted_success"] - _ZPD_OPTIMAL))
    return candidates[:n]

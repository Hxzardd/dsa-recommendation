import logging

from pipeline.recommender.bkt import process_submission
from pipeline.recommender.hlr import process_hlr
from pipeline.recommender.scoring import compute_submission_score
from database.postgres.db import mark_recommendation_attempted

log = logging.getLogger(__name__)


_state_update_service = None


def _get_state_update_service():
    """Create the production graph projection service lazily.

    BKT/HLR persistence remains owned by the existing /update flow. This
    service only mirrors /update's computed result into the recommendation
    graph and its cache/durable stores.
    """
    global _state_update_service
    if _state_update_service is None:
        import db_env
        from pipeline.recommender.services.neo4j_graph_store import Neo4jGraphStore
        from pipeline.recommender.services.state_update_service import StateUpdateService
        from pipeline.recommender.services.user_graph_service import UserGraphService

        neo4j = Neo4jGraphStore(
            db_env.neo4j_driver(), database=db_env.NEO4J_DATABASE
        )
        _state_update_service = StateUpdateService(
            UserGraphService(db=None, redis=None, neo4j=neo4j)
        )
    return _state_update_service


def _split_current_state(submission):
    """Reshape problemTopics [{topicId, currentMastery, currentHlr}] into the
    flat {topic: value} dicts process_submission / process_hlr expect."""
    current_mastery = {}
    current_hlr = {}
    for t in submission.problemTopics:
        if t.currentMastery is not None:
            current_mastery[t.topicId] = t.currentMastery
        if t.currentHlr is not None:
            current_hlr[t.topicId] = t.currentHlr
    return current_mastery, current_hlr


def _merge_to_updated_topics(updated_mastery, updated_hlr):
    """Reshape the two flat {topic: value} dicts back into a single
    updatedTopics: [{topicId, updatedMastery, updatedHlr}] array."""
    all_topics = set(updated_mastery.keys()) | set(updated_hlr.keys())
    return [
        {
            "topicId": t,
            "updatedMastery": updated_mastery.get(t),
            "updatedHlr": updated_hlr.get(t),
        }
        for t in all_topics
    ]


def handle_update(submission):
    """
    Stateless calculator: computes the submission score, the updated BKT
    mastery and the updated HLR state, and RETURNS them for the backend to
    persist. ML is the single source of truth for these numbers (Bayesian
    P(L) with proximity dampening isn't reproducible by a simple rolling
    average, so a caller inventing its own math during an outage would
    silently corrupt the model's invariants) but it does NOT write the
    backend-owned `user_topic_mastery` / `user_hlr_state` tables itself --
    the backend owns all mastery persistence (see the backend's
    integrations/ml/apply-topic-mastery.ts). This keeps a single writer for
    those tables and matches the documented /update contract.

    ML still maintains its OWN recommendation state below (the feedback-loop
    marker and the graph projection into Redis/Neo4j); those are ML-internal
    stores, not the backend's mastery tables.
    """
    submission_dict = submission.model_dump()
    current_mastery, current_hlr = _split_current_state(submission)

    updated_mastery, mastered_topics, bkt_results = process_submission(
        submission_dict, current_mastery
    )
    updated_hlr, hlr_results = process_hlr(
        submission_dict, current_hlr
    )

    # Score formulation from the forwarded telemetry (see scoring.py). Uses
    # the prior average topic mastery as the smoothing baseline.
    score = compute_submission_score(submission_dict)

    # Feedback loop, write half: mark this problem's most recent
    # not-yet-attempted recommendation_log row as attempted, so the ranker
    # can eventually learn from recommend -> attempt outcomes. submission.
    # problemId is the LeetCode title slug (same join key
    # save_recommendation_log resolves recommendations by -- see
    # database.postgres.db.resolve_problem_ids_by_title_slugs). Best-effort:
    # most submissions won't correspond to a prior recommendation at all,
    # and a lookup miss there is a no-op, not an error.
    mark_recommendation_attempted(submission.userId, submission.problemId)

    updated_topics = _merge_to_updated_topics(updated_mastery, updated_hlr)

    # BKT and HLR were computed exactly once above. Project those results into
    # the recommendation graph without recalculating or persisting either.
    _get_state_update_service().apply_update(
        submission=submission.model_dump(),
        updated_mastery=updated_mastery,
        mastered_topics=mastered_topics,
        bkt_results=bkt_results,
        updated_hlr=updated_hlr,
        hlr_results=hlr_results,
    )

    return {
        "userId": submission.userId,
        "problemId": submission.problemId,
        "updatedTopics": updated_topics,
        "masteredTopics": mastered_topics,
        # ML-owned score formulation -- `score` is 0..1, `normalisedScore` is
        # 0..100. The backend persists these as the submission's finalScore /
        # normalisedScore (falling back to its local calculateScore only when
        # this service is unreachable).
        "score": score["finalScore"],
        "normalisedScore": score["normalisedScore"],
        "results": {"bkt": bkt_results, "hlr": hlr_results, "score": score},
    }

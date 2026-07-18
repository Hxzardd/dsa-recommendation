import logging

from pipeline.recommender.bkt import process_submission
from pipeline.recommender.hlr import process_hlr
from database.postgres.db import save_user_mastery_live, save_user_hlr, mark_recommendation_attempted

log = logging.getLogger(__name__)


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
    Computes AND persists the updated BKT/HLR state -- ML is the single
    source of truth for mastery (Bayesian P(L) with proximity dampening
    isn't reproducible by a simple rolling average, so a caller falling
    back to its own math during an outage would silently corrupt the
    mastery model's invariants -- see save_user_mastery/save_user_hlr,
    which translate the ML topic slug to the backend's topic.id via
    database.postgres.topic_taxonomy before writing).

    Persistence failures propagate (not swallowed) so a service caller
    (e.g. the backend's judge0 submission webhook, see middlewares/auth.py's
    ML_SERVICE_TOKEN) sees a real error and can retry/queue the submission,
    rather than getting a 200 that silently didn't save anything.
    """
    current_mastery, current_hlr = _split_current_state(submission)

    updated_mastery, mastered_topics, bkt_results = process_submission(
        submission.model_dump(), current_mastery
    )
    updated_hlr, hlr_results = process_hlr(
        submission.model_dump(), current_hlr
    )

    if updated_mastery:
        save_user_mastery_live(submission.userId, updated_mastery)
    if updated_hlr:
        save_user_hlr(submission.userId, updated_hlr)

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

    return {
        "userId": submission.userId,
        "problemId": submission.problemId,
        "updatedTopics": updated_topics,
        "masteredTopics": mastered_topics,
        "results": {"bkt": bkt_results, "hlr": hlr_results},
    }
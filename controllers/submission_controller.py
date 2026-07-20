import logging

from pipeline.recommender.bkt import process_submission
from pipeline.recommender.hlr import process_hlr
from database.postgres.db import (
    save_user_mastery_live, save_user_hlr, mark_recommendation_attempted,
    release_connection,
)
from pipeline.recommender.services.user_graph_service import UserGraphService
from pipeline.recommender.services.state_update_service import StateUpdateService

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


def _mirror_submission_to_graph(submission, current_mastery, current_hlr):
    """
    Best-effort: mirror this submission into the Neo4j/Redis-backed
    recommendation graph via StateUpdateService, so /recommend and
    /topic/recommend/* stop serving a graph frozen at whatever mastery
    state existed the first time this user's graph was cached.

    Without this call, StateUpdateService.process_submission() -- which
    UserGraphService.get() is explicitly built to check for ("a graph
    mutated by StateUpdateService.process_submission() is actually seen
    here", see recommend.py::_get_graph) -- was never invoked from
    anywhere in the live request path. The first /recommend-family call
    for a user builds and caches a fresh graph from Postgres; every call
    after that returns the same cached graph forever, because nothing
    ever mutated or re-saved it.

    Imported lazily (not at module level) to avoid constructing Qdrant/
    Neo4j clients before they're needed, and to keep this file free of a
    hard import-time dependency on recommendation_controller.py's lazy
    singletons.

    Deliberately never raises: a Neo4j/Qdrant hiccup here must not turn
    into a failed /update response -- the core mastery/HLR persistence
    above already happened and must not be rolled back or reported as
    failed because of a side-channel cache-mirroring problem.
    """
    from controllers.recommendation_controller import (
        _get_qdrant, _get_neo4j_store, _get_db_wrapper,
    )

    db_wrapper = None
    try:
        db_wrapper = _get_db_wrapper()
        bkt_store = {submission.userId: dict(current_mastery)}
        hlr_store = {submission.userId: dict(current_hlr)}
        graph_service = UserGraphService(
            db=db_wrapper, redis=None, neo4j=_get_neo4j_store(),
            bkt=bkt_store, hlr=hlr_store,
        )
        state_service = StateUpdateService(
            graph_service, qdrant=_get_qdrant(),
            bkt_store=bkt_store, hlr_store=hlr_store,
        )
        state_service.process_submission(
            submission.userId, submission.model_dump(), rebuild_vector=False
        )
    except Exception as exc:
        log.warning(
            "[handle_update] graph mirroring failed for user %s, problem %s "
            "(%s: %s) -- mastery/HLR persistence above is unaffected.",
            submission.userId, submission.problemId,
            exc.__class__.__name__, exc,
        )
    finally:
        if db_wrapper is not None:
            release_connection(db_wrapper.conn)


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

    # Best-effort: keep the Neo4j/Redis recommendation graph in sync with
    # this submission. See _mirror_submission_to_graph's docstring -- this
    # was previously never called from anywhere, leaving /recommend and
    # /topic/recommend/* serving a graph frozen at first-cache time.
    _mirror_submission_to_graph(submission, current_mastery, current_hlr)

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
        "results": {"bkt": bkt_results, "hlr": hlr_results},
    }

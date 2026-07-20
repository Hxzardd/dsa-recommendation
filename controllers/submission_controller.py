from pipeline.recommender.bkt import process_submission
from pipeline.recommender.hlr import process_hlr


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
    Stateless combined handler. Backend sends current mastery/HLR per topic
    in submission.problemTopics; ML computes and returns the updated state.
    ML never touches the database -- backend owns persistence.
    """
    current_mastery, current_hlr = _split_current_state(submission)

    updated_mastery, mastered_topics, bkt_results = process_submission(
        submission.model_dump(), current_mastery
    )
    updated_hlr, hlr_results = process_hlr(
        submission.model_dump(), current_hlr
    )

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

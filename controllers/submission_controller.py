from pipeline.recommender.bkt import process_submission
from pipeline.recommender.hlr import process_hlr


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

    return {
        "userId": submission.userId,
        "problemId": submission.problemId,
        "updatedTopics": updated_topics,
        "masteredTopics": mastered_topics,
        "results": {"bkt": bkt_results, "hlr": hlr_results},
    }
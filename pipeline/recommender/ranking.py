from datetime import datetime, timezone
from database.postgres.db import get_connection, release_connection
from pipeline.recommender.hlr import calculate_urgency

WEIGHTS = {
    "bkt_mastery":    0.35,
    "hlr_urgency":    0.25,
    "similarity":     0.25,
    "variety":        0.15,
}

def calculate_variety_score(problem_topics, recent_topics, window=10):
    if not recent_topics:
        return 1.0
    recent_count = sum(
        1 for t in problem_topics if t in recent_topics[-window:]
    )
    variety_score = 1.0 - (recent_count / max(1, len(problem_topics)))
    return round(max(0.0, variety_score), 4)

def load_prerequisites() -> dict:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT topic_id, prerequisite_id FROM topic_prerequisite'
            )
            rows = cur.fetchall()
            prereqs = {}
            for topic_id, prereq_id in rows:
                if topic_id not in prereqs:
                    prereqs[topic_id] = []
                prereqs[topic_id].append(prereq_id)
            return prereqs
    finally:
       release_connection(conn)

_PREREQ_TABLE = None

def _get_prereq_table():
    """
    FIX (Greptile P1 "Transient Failure Disables Prerequisites"): the old
    version cached {} on ANY failure (including a transient one during
    Postgres startup), and since {} is falsy-but-not-None, `_PREREQ_TABLE
    is None` was False on every subsequent call -- meaning it never
    retried, ever, even after Postgres recovered. Every prerequisite
    check silently passed for the rest of the process lifetime.

    Fixed: only a SUCCESSFUL load is cached. A failed attempt leaves
    _PREREQ_TABLE as None, so the next call retries instead of being
    stuck with a stale empty table forever.
    """
    global _PREREQ_TABLE
    if _PREREQ_TABLE is None:
        try:
            _PREREQ_TABLE = load_prerequisites()
        except Exception:
            return {}   # NOT cached -- next call retries
    return _PREREQ_TABLE

def prerequisite_check(problem_topics, user_bkt_mastery, threshold=0.75):
    prereqs = _get_prereq_table()
    for topic in problem_topics:
        for prereq in prereqs.get(topic, []):
            if user_bkt_mastery.get(prereq, 0) < threshold:
                return False
    return True

def rank_candidates(
    candidates,
    user_bkt_mastery,
    user_hlr_state,
    recent_topics,
    current_timestamp=None
):
    if current_timestamp is None:
        current_timestamp = datetime.now(timezone.utc).timestamp()

    ranked = []

    for candidate in candidates:
        problem_topics = candidate.get("topics", [])
        if not problem_topics:
            continue
        similarity_score = candidate.get("score", 0.0)

        if not prerequisite_check(problem_topics, user_bkt_mastery):
            continue

        topic_masteries = [user_bkt_mastery.get(t, 0.15) for t in problem_topics]
        avg_mastery = sum(topic_masteries) / max(1, len(topic_masteries))
        bkt_score = 1.0 - avg_mastery

        topic_urgencies = [
            calculate_urgency(user_hlr_state.get(t, {}), current_timestamp)
            for t in problem_topics
        ]
        hlr_score = sum(topic_urgencies) / max(1, len(topic_urgencies))

        sim_score = similarity_score
        variety_score = calculate_variety_score(problem_topics, recent_topics)

        final_score = (
            WEIGHTS["bkt_mastery"] * bkt_score +
            WEIGHTS["hlr_urgency"] * hlr_score +
            WEIGHTS["similarity"]  * sim_score +
            WEIGHTS["variety"]     * variety_score
        )

        ranked.append({
            "title_slug": candidate.get("title_slug"),
            "title": candidate.get("title"),
            "description": candidate.get("description"),
            "topics": problem_topics,
            "difficulty_score": candidate.get("difficulty_score"),
            "category": candidate.get("category", "general"),
            "bkt_score": round(bkt_score, 4),
            "hlr_score": round(hlr_score, 4),
            "similarity_score": round(sim_score, 4),
            "variety_score": round(variety_score, 4),
            "final_score": round(final_score, 4)
        })

    ranked.sort(key=lambda x: x["final_score"], reverse=True)
    return ranked
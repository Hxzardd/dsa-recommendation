import requests
from collections import defaultdict
from pipeline.recommender.hlr import seed_half_life_from_cf
from pipeline.recommender.bkt import process_submission
from database.postgres.db import get_connection, save_user_hlr
from psycopg2.extras import RealDictCursor


def get_user_cf_handle(user_id: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT linked_codeforces FROM "user" WHERE id = %s',
                (user_id,)
            )
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()


def get_existing_hlr_topics(user_id: str) -> set:
    """Return set of topic_ids that already have HLR rows for this user."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT topic_id FROM user_hlr_state WHERE user_id = %s",
                (user_id,)
            )
            rows = cur.fetchall()
            return {row["topic_id"] for row in rows}
    finally:
        conn.close()


def get_cf_submissions(cf_handle: str):
    try:
        response = requests.get(
            f"https://codeforces.com/api/user.status?handle={cf_handle}",
            timeout=10
        )
        data = response.json()
        if data.get("status") == "OK":
            return data.get("result", [])
        return []
    except Exception:
        return []


def handle_seed_hlr(user_id: str):
    cf_handle = get_user_cf_handle(user_id)
    if not cf_handle:
        return {"message": "No Codeforces handle linked for this user"}

    submissions = get_cf_submissions(cf_handle)
    if not submissions:
        return {"message": "No Codeforces submissions found"}

    converted = [
        {
            "problemId": sub.get("problem", {}).get("name", ""),
            "verdict": "OK" if sub.get("verdict") == "OK" else "FAILED",
            "timestamp": sub.get("creationTimeSeconds")
        }
        for sub in submissions
    ]

    problem_to_topics = defaultdict(list)
    for sub in submissions:
        problem = sub.get("problem", {})
        name = problem.get("name", "")
        tags = problem.get("tags", [])
        problem_to_topics[name] = tags

    half_lives = seed_half_life_from_cf(converted, problem_to_topics)

    # Only seed topics the user doesn't already have HLR data for.
    # This prevents reseeding from wiping out existing review schedules.
    existing_topics = get_existing_hlr_topics(user_id)
    new_topics = {t: hl for t, hl in half_lives.items() if t not in existing_topics}

    if not new_topics:
        return {
            "userId": user_id,
            "cf_handle": cf_handle,
            "topics_seeded": 0,
            "message": "All topics already have HLR data — nothing to seed"
        }

    hlr_state = {
        topic: {
            "half_life": hl,
            "last_review": None,
            "p_recall": 0.5,
            "next_review_days": 1.0
        }
        for topic, hl in new_topics.items()
    }

    save_user_hlr(user_id, hlr_state)

    return {
        "userId": user_id,
        "cf_handle": cf_handle,
        "topics_seeded": len(new_topics),
        "half_lives": new_topics
    }


def get_user_lc_handle(user_id: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT linked_leetcode FROM "user" WHERE id = %s',
                (user_id,)
            )
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()


def get_lc_submissions(lc_handle: str):
    try:
        query = """
        {
          recentSubmissionList(username: "%s", limit: 100) {
            title
            statusDisplay
            timestamp
            topicTags {
              slug
            }
          }
        }
        """ % lc_handle
        response = requests.post(
            "https://leetcode.com/graphql",
            json={"query": query},
            timeout=10
        )
        data = response.json()
        return data.get("data", {}).get("recentSubmissionList", [])
    except Exception:
        return []


def save_user_mastery(user_id: str, mastery: dict):
    """Write seeded BKT mastery to user_topic_mastery table."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for topic_id, mastery_score in mastery.items():
                cur.execute("""
                    INSERT INTO user_topic_mastery (user_id, topic_id, mastery_score, updated_at)
                    VALUES (%s, %s, %s, NOW())
                    ON CONFLICT (user_id, topic_id) DO NOTHING
                """, (user_id, topic_id, mastery_score))
        conn.commit()
    finally:
        conn.close()


def handle_seed_bkt(user_id: str):
    lc_handle = get_user_lc_handle(user_id)
    if not lc_handle:
        return {"message": "No LeetCode handle linked for this user"}

    submissions = get_lc_submissions(lc_handle)
    if not submissions:
        return {"message": "No LeetCode submissions found"}

    # Build mastery from LC submissions using BKT.
    # Start from empty mastery and process each submission in order.
    current_mastery = {}
    for sub in submissions:
        title = sub.get("title", "")
        status = sub.get("statusDisplay", "")
        tags = [t.get("slug", "") for t in sub.get("topicTags", [])]

        verdict = "OK" if status == "Accepted" else "FAILED"

        submission_dict = {
            "problemId": title,
            "verdict": verdict,
            "hintsUsed": 0,
            "testCasesPassed": 1 if verdict == "OK" else 0,
            "totalTestCases": 1,
            "submissionCount": 1,
            "normalisedScore": 1.0 if verdict == "OK" else 0.0,
            "timestamp": sub.get("timestamp", 0)
        }

        updated_mastery, _, _ = process_submission(submission_dict, current_mastery)
        current_mastery = updated_mastery

    if not current_mastery:
        return {"message": "No topics found in LeetCode submissions"}

    # Only write topics not already in the mastery table (don't overwrite real data).
    save_user_mastery(user_id, current_mastery)

    return {
        "userId": user_id,
        "lc_handle": lc_handle,
        "submissions_processed": len(submissions),
        "topics_seeded": len(current_mastery),
        "mastery": current_mastery
    }
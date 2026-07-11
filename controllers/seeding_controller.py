import requests
from collections import defaultdict
from pipeline.recommender.hlr import seed_half_life_from_cf
from database.postgres.db import get_connection, save_user_hlr

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

    hlr_state = {
        topic: {
            "half_life": hl,
            "last_review": None,
            "p_recall": 0.5,
            "next_review_days": 1.0
        }
        for topic, hl in half_lives.items()
    }

    save_user_hlr(user_id, hlr_state)

    return {
        "userId": user_id,
        "cf_handle": cf_handle,
        "topics_seeded": len(half_lives),
        "half_lives": half_lives
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

def handle_seed_bkt(user_id: str):
    lc_handle = get_user_lc_handle(user_id)
    if not lc_handle:
        return {"message": "No LeetCode handle linked for this user"}

    submissions = get_lc_submissions(lc_handle)
    if not submissions:
        return {"message": "No LeetCode submissions found"}

    return {
        "userId": user_id,
        "lc_handle": lc_handle,
        "submissions_found": len(submissions)
    }



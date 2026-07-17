import requests
from collections import defaultdict
from pipeline.recommender.hlr import seed_half_life_from_cf
from pipeline.recommender.bkt import (
    process_submission, calculate_observed, update_bkt,
    MASTERY_THRESHOLD, DEFAULT_P_L,
)
from database.postgres.db import get_connection, release_connection, save_user_hlr, save_user_mastery
from database.postgres.topic_taxonomy import canonical_tags_for_leetcode_slug
from psycopg2.extras import RealDictCursor

def user_exists(user_id: str) -> bool:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT 1 FROM "user" WHERE id = %s',
                (user_id,)
            )
            return cur.fetchone() is not None
    finally:
        release_connection(conn)

class ProviderError(Exception):
    """
    FIX (Greptile P2 "Provider Failures Become Empty History"): raised
    when Codeforces/LeetCode actually fail (timeout, non-JSON, HTTP
    error) -- distinct from a genuinely empty submission history, so
    callers can tell a retryable outage apart from "this user really
    has no submissions" instead of both collapsing into the same
    silent empty-list response.
    """
    pass


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
        release_connection(conn)


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
        release_connection(conn)


def get_cf_submissions(cf_handle: str):
    """
    Raises ProviderError on an actual fetch failure (timeout, non-JSON,
    non-OK status from Codeforces) instead of silently returning [].
    A real empty submission history still returns [] normally -- only
    genuine failures raise now.
    """
    try:
        response = requests.get(
            f"https://codeforces.com/api/user.status?handle={cf_handle}",
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise ProviderError(f"Codeforces request failed: {exc}") from exc
    except ValueError as exc:
        raise ProviderError(f"Codeforces returned non-JSON response: {exc}") from exc

    if data.get("status") != "OK":
        # A malformed/unknown handle still returns status != "OK" from CF's
        # API but is a legitimate "no such user" case, not a fetch failure
        # -- treat as empty history, not ProviderError.
        return []
    return data.get("result", [])


def handle_seed_hlr(user_id: str):
    if not user_exists(user_id):
        return {"message": "User not found"}
    cf_handle = get_user_cf_handle(user_id)
    if not cf_handle:
        return {"message": "No Codeforces handle linked for this user"}

    try:
        submissions = get_cf_submissions(cf_handle)
    except ProviderError as exc:
        return {
            "userId": user_id,
            "cf_handle": cf_handle,
            "error": "provider_unavailable",
            "message": f"Could not fetch Codeforces submissions: {exc}",
        }

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
        tags = [t.lower().replace(" ", "_") for t in problem.get("tags", [])]
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
        release_connection(conn)


def _fetch_question_topic_tags(title_slug: str) -> list[str]:
    """
    LeetCode's own tag slugs (e.g. "hash-table") for one problem, translated
    to our canonical ML tags via topic_taxonomy.canonical_tags_for_leetcode_slug
    (e.g. ["hash_map"]). Best-effort: any fetch/parse failure returns []
    rather than raising -- one problem's tags being unavailable shouldn't
    abort the whole history import, unlike get_lc_submissions' own errors.
    """
    query = """
    query questionTopicTags($titleSlug: String!) {
      question(titleSlug: $titleSlug) {
        topicTags { slug }
      }
    }
    """
    try:
        response = requests.post(
            "https://leetcode.com/graphql",
            json={"query": query, "variables": {"titleSlug": title_slug}},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return []
    if "errors" in data:
        return []
    question = (data.get("data") or {}).get("question") or {}
    lc_slugs = [t["slug"] for t in question.get("topicTags", []) if t.get("slug")]

    canonical: list[str] = []
    for lc_slug in lc_slugs:
        for tag in canonical_tags_for_leetcode_slug(lc_slug):
            if tag not in canonical:
                canonical.append(tag)
    return canonical


def get_lc_submissions(lc_handle: str):
    """
    Raises ProviderError on an actual fetch failure -- same fix as get_cf_submissions.

    FIX (backend review "seed_bkt topicTags:[] bug"): LeetCode's
    recentSubmissionList does NOT return topicTags -- confirmed against
    their live GraphQL API (the field is silently empty, not an error).
    Tags only come back from the question(titleSlug) query. Fetches them
    with a second round-trip per UNIQUE problem (not per submission -- the
    same problem can appear many times in a user's history) and attaches
    them under the same "topicTags": [{"slug": ...}] shape the original
    (broken) inline query claimed to return, so handle_seed_bkt's
    downstream parsing needs no changes. Tags are translated from
    LeetCode's own taxonomy to our canonical ML tags before attaching --
    see _fetch_question_topic_tags.
    """
    try:
        query = """
        query userRecentSubmissions($username: String!) {
          recentSubmissionList(username: $username, limit: 100) {
            title
            titleSlug
            statusDisplay
            timestamp
          }
        }
        """
        response = requests.post(
            "https://leetcode.com/graphql",
            json={"query": query, "variables": {"username": lc_handle}},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise ProviderError(f"LeetCode request failed: {exc}") from exc
    except ValueError as exc:
        raise ProviderError(f"LeetCode returned non-JSON response: {exc}") from exc

    if "errors" in data:
        # GraphQL-level errors (bad handle, rate limited, etc) come back
        # as HTTP 200 with an "errors" key -- also a real failure, not
        # legitimately empty history.
        raise ProviderError(f"LeetCode GraphQL error: {data['errors']}")

    submissions = data.get("data", {}).get("recentSubmissionList", [])

    tag_cache: dict[str, list[str]] = {}
    for sub in submissions:
        title_slug = sub.get("titleSlug")
        if not title_slug:
            sub["topicTags"] = []
            continue
        if title_slug not in tag_cache:
            tag_cache[title_slug] = _fetch_question_topic_tags(title_slug)
        sub["topicTags"] = [{"slug": tag} for tag in tag_cache[title_slug]]

    return submissions


def _apply_bkt_update(current_mastery: dict, topics: list, verdict: str,
                      hints_used: int, test_cases_passed: int,
                      total_test_cases: int, submission_count: int,
                      normalised_score: float) -> tuple:
    """
    FIX (Greptile P1 "Fetched Topic Tags Are Discarded"): process_submission()
    looks up topics via the module-level problem_to_topics mapping keyed
    by problemId -- but LeetCode's API already RETURNS the real topic
    tags per submission directly (sub["topicTags"]). Routing through
    process_submission() meant a valid recent problem absent from (or
    normalizing differently in) that mapping produced ZERO mastery
    update despite LC handing us its actual tags right there in the
    response. This applies BKT directly against the topics LC gave us,
    bypassing the internal title-to-slug lookup entirely for this path.

    Mirrors process_submission()'s per-topic logic exactly (same
    calculate_observed + update_bkt + MASTERY_THRESHOLD), just topic-source
    swapped.
    """
    if not topics:
        return current_mastery, []

    observed = calculate_observed(
        verdict=verdict, hints_taken=hints_used,
        test_cases_passed=test_cases_passed, total_test_cases=total_test_cases,
        submission_count=submission_count, normalised_score=normalised_score,
    )

    updated = dict(current_mastery)
    for topic in topics:
        current_p_l = current_mastery.get(topic, DEFAULT_P_L["branch"])
        updated[topic] = update_bkt(current_p_l, observed)

    return updated, topics


def handle_seed_bkt(user_id: str):
    if not user_exists(user_id):
        return {"message": "User not found"}
    lc_handle = get_user_lc_handle(user_id)
    if not lc_handle:
        return {"message": "No LeetCode handle linked for this user"}

    try:
        submissions = get_lc_submissions(lc_handle)
    except ProviderError as exc:
        return {
            "userId": user_id,
            "lc_handle": lc_handle,
            "error": "provider_unavailable",
            "message": f"Could not fetch LeetCode submissions: {exc}",
        }

    if not submissions:
        return {"message": "No LeetCode submissions found"}

    # FIX (Greptile P1 "History Is Applied Newest First"): LeetCode's
    # recentSubmissionList returns newest-first. BKT is a SEQUENTIAL
    # update -- each result feeds into the next as current_mastery -- so
    # processing newest-first applies the user's attempts in REVERSE
    # chronological order, producing a materially different (wrong)
    # final mastery state than processing oldest-to-newest. Sorted here
    # by numeric timestamp ascending before the fold. LC's timestamp
    # field is a string in the GraphQL response, hence int(...).
    submissions = sorted(submissions, key=lambda s: int(s.get("timestamp", 0) or 0))

    current_mastery = {}
    for sub in submissions:
        status = sub.get("statusDisplay", "")
        # FIX: use the REAL tags LeetCode returned for this submission,
        # not a title-to-slug guess routed through an unrelated mapping.
        tags = [t.get("slug", "") for t in sub.get("topicTags", []) if t.get("slug")]
        verdict = "OK" if status == "Accepted" else "FAILED"
        current_mastery, _ = _apply_bkt_update(
            current_mastery, tags, verdict,
            hints_used=0,
            test_cases_passed=1 if verdict == "OK" else 0,
            total_test_cases=1,
            submission_count=1,
            normalised_score=1.0 if verdict == "OK" else 0.0,
        )

    if not current_mastery:
        return {"message": "No topics found in LeetCode submissions"}

    save_user_mastery(user_id, current_mastery)

    return {
        "userId": user_id,
        "lc_handle": lc_handle,
        "submissions_processed": len(submissions),
        "topics_seeded": len(current_mastery),
        "mastery": current_mastery
    }
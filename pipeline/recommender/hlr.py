import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone

from pipeline.recommender.telemetry import compute_telemetry_signal_from_submission

# Load problem to topics mapping. Absolute path so this works regardless of
# the working directory the process is launched from. Falls back to an empty
# mapping (with a warning) instead of crashing at import time if the file is
# missing -- see bkt.py's identical fix for why this must not be a hard crash.
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_pt_edges_path = os.path.join(_BASE_DIR, "data", "problem_topic_edges_normalized.json")
try:
    with open(_pt_edges_path) as f:
        pt_edges = json.load(f)
except FileNotFoundError:
    print(f"[!] {_pt_edges_path} not found -- hlr.py starting with an EMPTY "
          f"problem->topic mapping.")
    pt_edges = []

problem_to_topics = defaultdict(list)
for edge in pt_edges:
    problem_to_topics[edge["source"]].append(edge["target"])

MIN_HALF_LIFE = 1.0
MAX_HALF_LIFE = 180.0
MASTERY_THRESHOLD = 0.75
RECALL_THRESHOLD = 0.5


def _parse_aware(iso_str):
    """
    Parse an ISO datetime string and guarantee a UTC-aware result.
    Stored states or client payloads can arrive as either:
        "2026-06-30T12:00:00+00:00"   (aware)
        "2026-06-30T12:00:00"         (naive -- no timezone)
    datetime.fromisoformat() on the naive form returns tzinfo=None, and
    subtracting it from a UTC-aware "now" raises:
        TypeError: can't subtract offset-naive and offset-aware datetimes
    This wrapper normalises naive timestamps to UTC so every caller in
    this module (and controllers/mastery_controller.py's proficiency
    calculation) can safely subtract two datetimes without checking first.
    """
    dt = datetime.fromisoformat(iso_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _get_problem_id(sub):
    """
    Accept either snake_case (problem_id) or camelCase (problemId) keys.
    process_hlr()'s documented submission schema uses problemId, but CF
    history payloads / older callers may use problem_id. Checking only
    one form means the other silently returns no topics and the user
    gets no seeded half-life data at all.
    """
    return sub.get("problemId") or sub.get("problem_id")


def seed_half_life_from_cf(cf_submissions, problem_to_topics):
    """
    Calculate initial half life per topic from CF history.
    Called once when user connects their CF account.
    """
    topic_stats = defaultdict(lambda: {"solved": 0, "attempted": 0, "last_seen": None})

    for sub in cf_submissions:
        problem_id = _get_problem_id(sub)
        topics = problem_to_topics.get(problem_id, [])
        for topic in topics:
            topic_stats[topic]["attempted"] += 1
            if sub.get("verdict") == "OK":
                topic_stats[topic]["solved"] += 1
            if topic_stats[topic]["last_seen"] is None:
                topic_stats[topic]["last_seen"] = sub.get("timestamp")

    half_lives = {}
    for topic, stats in topic_stats.items():
        solve_rate = stats["solved"] / max(1, stats["attempted"])
        solve_count = stats["solved"]
        base_h = MIN_HALF_LIFE * (2 ** (solve_count / 5))
        rate_factor = 0.5 + solve_rate
        half_life = min(MAX_HALF_LIFE, base_h * rate_factor)
        half_lives[topic] = round(half_life, 3)

    return half_lives

def recall_probability(half_life, days_since_review):
    """
    Calculate probability user still remembers topic.
    At t=0: p=1.0, at t=half_life: p=0.5
    """
    if days_since_review <= 0:
        return 1.0
    return round(2 ** (-days_since_review / half_life), 4)

def update_half_life(current_half_life, performance, days_since_review):
    """
    Update half life based on performance.
    Good performance increases half life.
    Poor performance decreases half life.
    """
    theta = 0.5
    scale = 2 ** (performance - theta)

    if days_since_review > 0:
        retention = recall_probability(current_half_life, days_since_review)
        if retention > 0.5 and performance > 0.6:
            scale *= 1.2

    new_half_life = current_half_life * scale
    new_half_life = max(MIN_HALF_LIFE, min(MAX_HALF_LIFE, new_half_life))
    return round(new_half_life, 3)

# =============================================================================
# URGENCY SCORE FOR RANKING
# urgency = 1 - p(recall)
# =============================================================================
def calculate_urgency(hlr_state, current_timestamp):
    """
    Calculate urgency score for ranking engine.
    Low recall = high urgency = topic needs review soon.
    Returns value between 0.0 and 1.0.
    """
    last_review = hlr_state.get("last_review")
    half_life = hlr_state.get("half_life", MIN_HALF_LIFE)

    if last_review is None:
        return 0.5

    # _parse_aware guarantees a UTC-aware datetime even if last_review
    # was stored without timezone info -- fixes the naive/aware subtraction
    # TypeError that previously crashed this function.
    last_review_dt = _parse_aware(last_review)
    now_dt = datetime.fromtimestamp(current_timestamp, tz=timezone.utc)
    days_since = (now_dt - last_review_dt).total_seconds() / 86400

    p_recall = recall_probability(half_life, days_since)
    urgency = 1.0 - p_recall

    if p_recall < RECALL_THRESHOLD:
        urgency = min(1.0, urgency * 1.5)

    return round(urgency, 4)

def process_hlr(submission, user_hlr_state):
    """
    Process a submission and update HLR state for all related topics.

    Args:
        submission: dict with userId, problemId, problemTopics,
                    verdict, hintsUsed, submissionCount,
                    normalisedScore, timestamp
        user_hlr_state: dict of {topic_slug: hlr_state} for this user

    Returns:
        updated_hlr_state, results
    """

    # Use the topics supplied by the backend when available.
    # Fall back to the static mapping for legacy callers.
    problem_topics = submission.get("problemTopics")

    if problem_topics:
        topics = [
            topic["topicId"]
            for topic in problem_topics
        ]
    else:
        problem_id = _get_problem_id(submission)
        topics = problem_to_topics.get(problem_id, [])

    if not topics:
        return user_hlr_state, []

    # Shared telemetry signal (also consumed by bkt.py::process_submission
    # for the same submission) -- see telemetry.py for the confidence-
    # penalty logic. Previously this module computed its own divergent
    # "performance" score (calculate_performance) from the same telemetry
    # bkt.py used for "observed" -- two different opinions of how well the
    # learner did on the exact same submission. Both now agree.
    performance = compute_telemetry_signal_from_submission(submission).value

    current_time = submission.get(
        "timestamp",
        datetime.now(timezone.utc).timestamp()
    )
    now_dt = datetime.fromtimestamp(current_time, tz=timezone.utc)

    updated_state = dict(user_hlr_state)
    results = []
    for topic in topics:
     current_state = user_hlr_state.get(topic, {})
     current_half_life = current_state.get("half_life", MIN_HALF_LIFE)
     last_review = current_state.get("last_review")

     if last_review:
        last_review_dt = _parse_aware(last_review)
        days_since = (now_dt - last_review_dt).total_seconds() / 86400
     else:
        days_since = 0

     new_half_life = update_half_life(
        current_half_life,
        performance,
        days_since,
     )

     p_recall = recall_probability(current_half_life, days_since)

     next_review_days = round(-new_half_life * math.log2(0.7), 1)

     new_state = {
        "half_life": new_half_life,
        "last_review": now_dt.isoformat(),
        "performance": performance,
        "p_recall": p_recall,
        "next_review_days": next_review_days,
     }

     updated_state[topic] = new_state

     results.append({
        "topic": topic,
        "performance": performance,
        "previous_half_life": current_half_life,
        "new_half_life": new_half_life,
        "p_recall": p_recall,
        "next_review_days": next_review_days,
     })
    return updated_state, results

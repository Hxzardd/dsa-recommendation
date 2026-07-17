import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone

from pipeline.recommender.telemetry import compute_telemetry_signal_from_submission

# Load problem to topics mapping. Neo4j FIRST (pipeline/graphs/
# neo4j_offline_writer.py's shared, centrally-updated copy), falling back
# to the local JSON file (absolute path so this works regardless of the
# working directory the process is launched from) if Neo4j is unavailable
# -- same graceful-degrade convention as every other Neo4j touchpoint in
# this repo, and the same fallback bkt.py uses. Falls back to an empty
# mapping (with a warning) instead of crashing at import time if both are
# unavailable.
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_pt_edges_path = os.path.join(_BASE_DIR, "data", "problem_topic_edges_normalized.json")

problem_to_topics = defaultdict(list)
try:
    from pipeline.graphs.neo4j_offline_writer import load_problem_topics
    _neo4j_pt = load_problem_topics()
except Exception:
    _neo4j_pt = {}

if _neo4j_pt:
    for slug, topics in _neo4j_pt.items():
        problem_to_topics[slug] = list(topics)
else:
    try:
        with open(_pt_edges_path) as f:
            pt_edges = json.load(f)
    except FileNotFoundError:
        print(f"[!] {_pt_edges_path} not found -- hlr.py starting with an EMPTY "
              f"problem->topic mapping.")
        pt_edges = []

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

# FIX (backend report: "spammed 4-5 solved within seconds, the forgetting
# curve never triggers, half life only increasing purely from performance"):
# `scale` below was applied at FULL strength regardless of days_since_review
# -- reviewing the exact same topic 5 times in a few seconds multiplied
# half_life by 2**(performance-0.5) every single time, since nothing in
# the formula reduced the update's weight for a near-zero gap. A rapid
# re-review isn't a real spaced-repetition data point (it doesn't test
# whether the user actually RETAINED anything over time, just whether they
# remembered what they did 3 seconds ago) and should barely move half_life
# at all -- reviewing right at or past the scheduled half_life is much
# more informative and should get close to full effect.
# _REVIEW_INFORMATIVENESS_TAU is the elapsed-time scale (in days) at which
# a review starts counting as meaningfully spaced -- 0.02 days ≈ 29
# minutes. Saturating (1 - e^-x) curve: ~0 for a same-second resubmit,
# ~63% by one tau, ~95% by 3 tau, asymptotically 1.0 for a well-spaced
# review. This does NOT touch recall_probability/urgency's own math --
# only how much a single update_half_life() call is allowed to move the
# half-life for a too-soon review.
_REVIEW_INFORMATIVENESS_TAU = 0.02   # days


def update_half_life(current_half_life, performance, days_since_review, weight=1.0):
    """
    Update half life based on performance.
    Good performance increases half life.
    Poor performance decreases half life.

    The update itself is scaled by how informative this review actually is
    given elapsed time (see _REVIEW_INFORMATIVENESS_TAU docstring above) --
    a review seconds after the last one barely moves half_life; a review
    genuinely spaced out gets close to the full performance-driven effect.

    weight: this topic's relevance to the problem just solved (0-1, default
    1.0 = full relevance, matching prior behavior exactly when omitted).
    A problem backend tags as "70% graphs, 30% dfs" should move the graphs
    half-life much more than the dfs one on the same submission -- without
    this every co-tagged topic got an identical update regardless of how
    central it actually was to the problem.
    """
    theta = 0.5
    scale = 2 ** (performance - theta)

    if days_since_review > 0:
        retention = recall_probability(current_half_life, days_since_review)
        if retention > 0.5 and performance > 0.6:
            scale *= 1.2

    informativeness = 1.0 - math.exp(-max(0.0, days_since_review) / _REVIEW_INFORMATIVENESS_TAU)
    effective_scale = 1.0 + (scale - 1.0) * informativeness * max(0.0, min(1.0, weight))

    new_half_life = current_half_life * effective_scale
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

    # weight: how central this topic is to the problem (0-1, e.g. a
    # "70% graphs, 30% dfs" problem) -- defaults to 1.0 (full effect, prior
    # behavior unchanged) when the backend doesn't send one, and for the
    # static-mapping fallback path, which has no per-topic weight concept.
    topic_weights: dict = {}
    if problem_topics:
        topics = [topic["topicId"] for topic in problem_topics]
        for topic in problem_topics:
            w = topic.get("weight")
            topic_weights[topic["topicId"]] = 1.0 if w is None else max(0.0, min(1.0, w))
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
        weight=topic_weights.get(topic, 1.0),
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

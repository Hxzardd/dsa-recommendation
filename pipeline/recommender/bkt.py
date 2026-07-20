import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone

from pipeline.recommender.telemetry import (
    MASTERY_THRESHOLD,
    compute_telemetry_signal,
    compute_telemetry_signal_from_submission,
)

# Load problem->topic mapping -- this is only the FALLBACK path
# process_submission() uses when the caller doesn't send problemTopics
# directly in the request body (see process_submission's docstring); the
# primary path never touches this. Neo4j FIRST (pipeline/graphs/
# neo4j_offline_writer.py's shared, centrally-updated copy), falling back
# to the local JSON file if Neo4j is unavailable -- same graceful-degrade
# convention as every other Neo4j touchpoint in this repo.
_BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_pt_edges_path = os.path.join(
    _BASE_DIR,
    "data",
    "problem_topic_edges_normalized.json",
)

problem_to_topics = defaultdict(list)
try:
    from pipeline.graphs.neo4j_offline_writer import load_problem_topics
    _neo4j_pt = load_problem_topics()
except Exception:
    _neo4j_pt = {}

if _neo4j_pt:
    for slug, topics in _neo4j_pt.items():
        problem_to_topics[slug] = list(topics)
    print(f"Loaded topic mappings for {len(problem_to_topics)} problems (from Neo4j)")
else:
    try:
        with open(_pt_edges_path) as f:
            pt_edges = json.load(f)
    except FileNotFoundError:
        print(
            f"[!] {_pt_edges_path} not found -- bkt.py starting with an EMPTY "
            f"problem->topic mapping."
        )
        pt_edges = []

    for edge in pt_edges:
        problem_to_topics[edge["source"]].append(edge["target"])

    print(f"Loaded topic mappings for {len(problem_to_topics)} problems "
          f"(Neo4j unavailable/empty -- from local JSON)")

BKT_PARAMS = {
    "P_T": 0.1,   # probability of learning after one attempt
    "P_G": 0.3,   # probability of guessing correctly without knowing
    "P_S": 0.1,   # probability of slipping even if they know
}

# Default initial P(L) per topic type
# Root topics start higher since user likely has some base knowledge
# Branch topics start lower since they are more specific
DEFAULT_P_L = {
    "root": 0.14,   # arrays, strings, math
    "branch": 0.12, # sliding window, two pointers etc
    "unknown": 0.1  # topic we have no info about
}

# Observed score below this is treated as a failed attempt -- the BKT
# learning transition (P_T) is skipped for failures, since "learning from
# a failed attempt" should not move mastery upward.
LEARNING_TRANSITION_THRESHOLD = 0.5

# Hard ceiling on how much a SINGLE submission can move mastery, regardless
# of how large the raw Bayesian update computes to. The uncapped formula can
# swing a cold-start topic (P_L=0.15) to ~0.6+ off one strong submission --
# that's a step-function jump, not the gradual, incremental leveling this
# system is meant to produce (Duolingo-style: many small confirmations, not
# one lucky submission maxing out a skill). Deterministic and explainable --
# no ML, just a clamp on the delta.
#
# This is the NEUTRAL (difficulty=0.5 or unknown) cap. When a problem's
# difficulty is known, the effective cap itself scales by difficulty (see
# _CAP_SCALE_MIN/_MAX below) -- without this, a flat cap silently erases the
# difficulty signal for any submission whose raw Bayesian delta already
# exceeds it, which cold-start submissions almost always do.
MAX_MASTERY_DELTA = 0.12

# Same 0.7x-1.3x range telemetry.py's difficulty_credit uses, applied to the
# cap instead of (in addition to) raw_perf -- so a trivial problem tops out
# lower (0.7 * 0.12 = 0.084) and a hard one tops out higher (1.3 * 0.12 =
# 0.156), even from a cold start where the raw delta would otherwise
# saturate either cap identically.
_CAP_SCALE_MIN = 0.7
_CAP_SCALE_MAX = 1.3

# Difficulty-aware dampening: a solve on a problem well below the user's
# current mastery on this topic is weaker evidence of growth than one at or
# above it -- they likely already knew it. Only dampens (never boosts) and
# never zeroes out a solve entirely (a solve is always some evidence).
_TRIVIAL_GAP_THRESHOLD = -0.15
_TRIVIAL_DAMPEN_FLOOR = 0.4

# Mastery-proximity dampening: the closer the user already is to fully
# knowing a topic, the less a single additional solve should move the
# needle -- diminishing returns as mastery grows, on top of (not instead
# of) the cold-start cap above. Only applied to positive deltas -- a poor
# submission should still be able to pull a highly-mastered topic back down
# at full strength, since forgetting/regression isn't subject to
# diminishing returns the same way growth is.
#
# EXPONENTIAL, not linear: proximity_scale(p_l) = floor + (1-floor)*e^(-k*p_l).
# A straight-line falloff (1-p_l) dampens 0.3 and 0.7 mastery almost the
# same amount relative to each other (0.7x vs 0.3x -- a flat 2.3x ratio
# everywhere). The exponential shape drops off MUCH faster in the low-to-
# mid range and then flattens as it approaches the floor, so a 0.7-mastery
# learner's gain is visibly, disproportionately more gradual than a
# 0.3-mastery learner's on the exact same problem -- not just a uniform
# scale-down -- while never fully vanishing (floors at
# _PROXIMITY_DAMPEN_FLOOR) so even a near-mastered topic still credits a
# genuine solve with SOME growth.
#
# At current_p_l=0.0: scale=1.0 (no dampening, matches the old linear
# formula's start). At current_p_l=1.0: scale≈floor (asymptotic, never
# fully bottoms out for any finite p_l<1).
_PROXIMITY_DAMPEN_FLOOR = 0.1
_PROXIMITY_DECAY_RATE = 4.0

# FIX (backend report: "if the same user re-submits the same problem
# they've already solved, their topic mastery keeps going up a little
# each time... is that expected?"): it was -- nothing in update_bkt
# accounted for how recently this topic was last reviewed, so a rapid
# resubmit of an already-solved problem got the exact same learning
# credit as a genuinely new solve. Re-solving something you solved 3
# seconds ago is weak-to-no evidence of NEW learning (you already knew
# it); re-solving it after real time has passed is closer to a fresh
# data point. Mirrors hlr.py's _REVIEW_INFORMATIVENESS_TAU (same
# principle, same 0.02-day/~29min time constant, kept as a separate
# constant since BKT and HLR are free to tune independently) -- only
# dampens POSITIVE deltas; a poor resubmit is still valid negative
# evidence regardless of timing.
_REPEAT_SOLVE_TAU = 0.02   # days


def calculate_observed(verdict, hints_taken, test_cases_passed,
                        total_test_cases, submission_count, normalised_score,
                        difficulty=None, weights=None):
    """
    Back-compat wrapper around telemetry.compute_telemetry_signal for
    callers (seeding_controller.py's CF/LC history replay) that still call
    this by its old name/signature directly instead of going through
    process_submission(). Delegates to the same shared signal bkt.py/hlr.py
    now use, so history-seeded mastery and live-submission mastery are
    computed from one formula instead of two divergent ones. `weights` is
    accepted for signature compatibility but unused -- telemetry.py owns
    the weighting now.
    """
    return compute_telemetry_signal(
        verdict=verdict, hints_taken=hints_taken,
        test_cases_passed=test_cases_passed, total_test_cases=total_test_cases,
        submission_count=submission_count, normalised_score=normalised_score,
        difficulty=difficulty,
    ).value


def update_bkt(current_p_l, observed, difficulty=None, days_since_last_review=None, weight=1.0):
    """
    Update knowledge probability using Bayes theorem.

    Args:
        current_p_l: current probability user knows this topic (0 to 1)
        observed: performance score from telemetry.compute_telemetry_signal (0 to 1)
        difficulty: optional 0-1 difficulty score of the solved problem. When
            provided, a solve well below the user's current mastery on this
            topic (see _TRIVIAL_GAP_THRESHOLD) has its positive delta
            dampened -- trivial practice teaches less than appropriately
            challenging practice. Omit (default) to skip this adjustment.
        days_since_last_review: optional elapsed time (days) since this
            topic's last recorded review (from the caller's currentHlr.
            last_review). When provided, a positive delta from a review
            too soon after the last one is heavily dampened (see
            _REPEAT_SOLVE_TAU docstring above) -- re-solving the same
            problem seconds after solving it isn't new evidence of
            learning. None (default, e.g. no prior HLR state at all --
            genuinely first-ever attempt) skips this adjustment entirely,
            same as every other None-means-"no signal yet" convention in
            this function.
        weight: this topic's relevance to the problem just solved (0-1,
            default 1.0 = full relevance, matching prior behavior exactly
            when omitted). A problem tagged "70% graphs, 30% dfs" should
            move the graphs mastery much more than the dfs one on the same
            submission.

    Returns:
        new_p_l: updated probability (0 to 1). A single call can move
            current_p_l by at most MAX_MASTERY_DELTA (difficulty=None) or
            MAX_MASTERY_DELTA scaled by _CAP_SCALE_MIN.._CAP_SCALE_MAX
            (difficulty given), further shrunk by mastery-proximity,
            trivial-gap, repeat-solve, and topic-weight dampening for
            positive deltas -- see the module docstring constants above.
    """
    P_T = BKT_PARAMS["P_T"]
    P_G = BKT_PARAMS["P_G"]
    P_S = BKT_PARAMS["P_S"]

    # P(L | correct observation)
    p_l_correct = (current_p_l * (1 - P_S)) / (
        (current_p_l * (1 - P_S)) + ((1 - current_p_l) * P_G)
    )

    # P(L | wrong observation)
    p_l_wrong = (current_p_l * P_S) / (
        (current_p_l * P_S) + ((1 - current_p_l) * (1 - P_G))
    )

    # Blend using observed score as weight
    p_l_given_obs = observed * p_l_correct + (1 - observed) * p_l_wrong

    # Account for learning from this attempt -- ONLY on a sufficiently
    # successful observation. Applying the learning transition (P_T)
    # unconditionally lets a fully failed attempt (observed=0.0) still
    # push mastery upward, since (1 - p_l_given_obs) * P_T > 0 regardless
    # of how bad the observation was. A failed attempt should not be
    # treated as evidence of learning.
    if observed >= LEARNING_TRANSITION_THRESHOLD:
        new_p_l_raw = p_l_given_obs + (1 - p_l_given_obs) * P_T
    else:
        new_p_l_raw = p_l_given_obs

    # Smoothing cap -- bound how far this ONE submission can move mastery.
    # The cap itself scales with difficulty (see MAX_MASTERY_DELTA's
    # docstring) so difficulty keeps differentiating outcomes even when the
    # raw delta is large enough to saturate a flat cap.
    if difficulty is not None:
        cap_scale = max(_CAP_SCALE_MIN, min(_CAP_SCALE_MAX,
                         _CAP_SCALE_MIN + (_CAP_SCALE_MAX - _CAP_SCALE_MIN) * max(0.0, min(1.0, difficulty))))
        effective_cap = MAX_MASTERY_DELTA * cap_scale
    else:
        effective_cap = MAX_MASTERY_DELTA
    delta = new_p_l_raw - current_p_l

    # Soft-saturate instead of hard-clipping: a plain min/max clip means any
    # raw delta beyond the cap (which cold-start submissions almost always
    # produce -- see MAX_MASTERY_DELTA's docstring) lands on the EXACT same
    # final value, an abrupt flat wall with no texture between "confidently
    # solved" and "barely solved" once both exceed it. tanh approaches the
    # cap asymptotically instead of hitting it outright -- small deltas pass
    # through almost unchanged (tanh(x)~=x near 0), large ones taper off
    # smoothly toward +-effective_cap, so the whole raw-delta -> final-delta
    # mapping is one continuous curve rather than linear-then-flat.
    if effective_cap > 0:
        delta = effective_cap * math.tanh(delta / effective_cap)

    if delta > 0:
        # Mastery-proximity dampening -- diminishing returns as the topic
        # approaches full mastery. Exponential falloff (see
        # _PROXIMITY_DAMPEN_FLOOR/_PROXIMITY_DECAY_RATE docstring above):
        # 1.0x at current_p_l=0, decaying toward the floor as current_p_l
        # grows, with most of the drop happening early rather than spread
        # evenly -- a 0.7-mastery learner's gain is visibly more gradual
        # than a 0.3-mastery learner's on the same problem, not just a
        # uniformly-smaller fraction of it.
        proximity_scale = _PROXIMITY_DAMPEN_FLOOR + (1.0 - _PROXIMITY_DAMPEN_FLOOR) * math.exp(
            -_PROXIMITY_DECAY_RATE * current_p_l)
        delta *= proximity_scale

        # Difficulty dampening for trivial solves (never amplifies a
        # decrease from a failed/weak attempt).
        if difficulty is not None:
            gap = difficulty - current_p_l
            if gap < _TRIVIAL_GAP_THRESHOLD:
                delta *= max(_TRIVIAL_DAMPEN_FLOOR, 1.0 + gap)

        # Repeat-solve dampening -- see _REPEAT_SOLVE_TAU docstring above.
        # Only dampens growth; a rapid-fire WRONG resubmit is still valid
        # negative evidence regardless of timing (handled by this being
        # inside the `delta > 0` branch only).
        if days_since_last_review is not None:
            informativeness = 1.0 - math.exp(
                -max(0.0, days_since_last_review) / _REPEAT_SOLVE_TAU)
            delta *= informativeness

        # Topic-weight dampening -- see `weight` docstring above.
        delta *= max(0.0, min(1.0, weight))
    elif observed >= LEARNING_TRANSITION_THRESHOLD:
        # A credited success (verdict=OK, strong enough to cross the
        # learning-transition threshold) can still produce a NEGATIVE raw
        # Bayesian delta at high prior mastery -- e.g. current_p_l=0.85,
        # observed=0.73 nets a small pull-down purely from the math (0.73
        # reads as "weaker than what 0.85 mastery would predict"), even
        # though the attempt was correct and got learning credit. A correct
        # solve should never actively PUNISH mastery, only dampen the
        # reward down to (but not below) zero -- that's a distinct case
        # from forgetting/regression, which only applies to attempts that
        # did NOT cross the threshold (handled by the trivial-gap block
        # above being skipped, delta staying whatever the raw negative
        # Bayesian pull was).
        delta = max(delta, 0.0)

    new_p_l = current_p_l + delta
    return round(min(1.0, max(0.0, new_p_l)), 4)

def _days_since_last_review(current_hlr, now_ts):
    """
    Elapsed days between a topic's currentHlr.last_review (sent by the
    caller per-topic, same field HLR itself already tracks) and this
    submission's timestamp. None if there's no prior review to compare
    against -- a genuinely first-ever attempt shouldn't be dampened by
    "repeat solve" logic; there's nothing to repeat yet.
    """
    if not current_hlr:
        return None
    last_review = current_hlr.get("last_review")
    if not last_review:
        return None
    try:
        last_review_dt = datetime.fromisoformat(last_review)
        if last_review_dt.tzinfo is None:
            last_review_dt = last_review_dt.replace(tzinfo=timezone.utc)
        now_dt = datetime.fromtimestamp(now_ts, tz=timezone.utc)
        return (now_dt - last_review_dt).total_seconds() / 86400
    except (ValueError, TypeError, OSError):
        return None


def process_submission(submission, user_mastery):
    """
    Process a submission and update BKT mastery for all related topics.

    Args:
        submission: dict with userId, problemId, problemTopics, verdict,
                    testCasesPassed, totalTestCases, hintsUsed,
                    submissionCount, normalisedScore, problemDifficulty
        user_mastery: dict of {topic_slug: p_l} for this user

    Returns:
        updated_mastery: dict of {topic_slug: new_p_l}
        mastered_topics: list of topics that crossed mastery threshold
        results: detailed results per topic
    """
    # Use the topics supplied by the backend instead of looking them up
    # from the static mapping.
    problem_topics = submission.get("problemTopics")

    # weight/currentHlr per topic -- see update_bkt's docstring for both.
    # Empty dicts (static-mapping fallback path) mean every topic gets the
    # prior default behavior: weight=1.0, no repeat-solve dampening.
    topic_weights: dict = {}
    topic_hlr: dict = {}
    if problem_topics:
       topics = [
        topic["topicId"]
        for topic in problem_topics
    ]
       for topic in problem_topics:
           w = topic.get("weight")
           topic_weights[topic["topicId"]] = 1.0 if w is None else max(0.0, min(1.0, w))
           topic_hlr[topic["topicId"]] = topic.get("currentHlr")
    else:
       problem_id = submission["problemId"]
       topics = problem_to_topics.get(problem_id, [])
    if not topics:
       return user_mastery, [], []

    # Shared telemetry signal (also consumed by hlr.py::process_hlr for the
    # same submission) -- see telemetry.py for the confidence-penalty logic.
    observed = compute_telemetry_signal_from_submission(submission).value
    difficulty = submission.get("problemDifficulty")
    now_ts = submission.get("timestamp") or datetime.now(timezone.utc).timestamp()

    updated_mastery = dict(user_mastery)
    mastered_topics = []
    results = []

    for topic in topics:
        # Get current P(L) or use default
        current_p_l = user_mastery.get(topic, DEFAULT_P_L["branch"])

        days_since = _days_since_last_review(topic_hlr.get(topic), now_ts)

        # Update BKT
        new_p_l = update_bkt(
            current_p_l, observed, difficulty=difficulty,
            days_since_last_review=days_since,
            weight=topic_weights.get(topic, 1.0),
        )
        updated_mastery[topic] = new_p_l

        # Check if topic just got mastered
        was_mastered = current_p_l >= MASTERY_THRESHOLD
        now_mastered = new_p_l >= MASTERY_THRESHOLD

        if now_mastered and not was_mastered:
            mastered_topics.append(topic)

        results.append({
            "topic": topic,
            "previous_p_l": current_p_l,
            "new_p_l": new_p_l,
            "mastered": now_mastered,
            "observed_score": observed,
        })

    return updated_mastery, mastered_topics, results

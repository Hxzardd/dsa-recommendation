import json
import os
from collections import defaultdict
import numpy as np

# Load problem->topic mapping
_BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_pt_edges_path = os.path.join(
    _BASE_DIR,
    "data",
    "problem_topic_edges_normalized.json",
)

try:
    with open(_pt_edges_path) as f:
        pt_edges = json.load(f)
except FileNotFoundError:
    print(
        f"[!] {_pt_edges_path} not found -- bkt.py starting with an EMPTY "
        f"problem->topic mapping."
    )
    pt_edges = []

problem_to_topics = defaultdict(list)
for edge in pt_edges:
    problem_to_topics[edge["source"]].append(edge["target"])

print(f"Loaded topic mappings for {len(problem_to_topics)} problems")

BKT_PARAMS = {
    "P_T": 0.1,   # probability of learning after one attempt
    "P_G": 0.3,   # probability of guessing correctly without knowing
    "P_S": 0.1,   # probability of slipping even if they know
}

# Default initial P(L) per topic type
# Root topics start higher since user likely has some base knowledge
# Branch topics start lower since they are more specific
DEFAULT_P_L = {
    "root": 0.2,    # arrays, strings, math
    "branch": 0.15, # sliding window, two pointers etc
    "unknown": 0.1  # topic we have no info about
}

# Mastery threshold — above this topic is considered mastered
MASTERY_THRESHOLD = 0.75

# Observed score below this is treated as a failed attempt -- the BKT
# learning transition (P_T) is skipped for failures, since "learning from
# a failed attempt" should not move mastery upward. Matches the original
# intent of the `if observed >= 0.5` branch before it was dropped.
LEARNING_TRANSITION_THRESHOLD = 0.5

# STEP 3 — OBSERVED SCORE CALCULATION
# Phase 1: weighted combination with default weights
# Phase 2: replace with XGBoost once real user data is available

def calculate_observed(verdict, hints_taken, test_cases_passed,
                       total_test_cases, submission_count, normalised_score,
                       weights=None):
    """
    Calculate observed performance score from submission signals.
    Returns a value between 0.0 and 1.0.
    
    Phase 1: weighted combination
    Phase 2: swap in XGBoost model here once training data is available
    """
    if total_test_cases == 0:
        return 0.0

    # Default weights — will be learned from data in Phase 2
    if weights is None:
        weights = {
            "normalised_score": 0.40,
            "pass_rate":        0.25,
            "hint_penalty":     0.20,
            "attempt_penalty":  0.15
        }

    # Component 1 — normalised score from backend
    # Failed attempts get 30% credit at most
    w1 = normalised_score if verdict == "OK" else normalised_score * 0.3

    # Component 2 — pass rate (test cases passed / total)
    w2 = test_cases_passed / total_test_cases

    # Component 3 — hint penalty (more hints = lower score)
    max_hints = 10
    w3 = max(0.0, 1 - (hints_taken / max_hints))

    # Component 4 — attempt penalty (more attempts = lower score)
    max_attempts = 10
    w4 = max(0.0, 1 - ((submission_count - 1) / max_attempts))

    # Weighted combination
    observed = (
        weights["normalised_score"] * w1 +
        weights["pass_rate"]        * w2 +
        weights["hint_penalty"]     * w3 +
        weights["attempt_penalty"]  * w4
    )

    # Failed attempts capped at 0.35 max
    if verdict != "OK":
        observed = min(0.35, observed)

    return round(min(1.0, max(0.0, observed)), 4)

# import xgboost as xgb
#
# def calculate_observed_xgb(model, verdict, hints_taken, test_cases_passed,
#                             total_test_cases, submission_count, normalised_score):
#     pass_rate = test_cases_passed / total_test_cases if total_test_cases > 0 else 0
#     hint_penalty = max(0.0, 1 - (hints_taken / 10))
#     attempt_penalty = max(0.0, 1 - ((submission_count - 1) / 10))
#     solved = 1 if verdict == "OK" else 0
#
#     # Interaction features — captures non linear relationships
#     hints_x_passrate = hint_penalty * pass_rate
#     attempts_x_hints = attempt_penalty * hint_penalty
#
#     X = np.array([[normalised_score, pass_rate, hint_penalty,
#                    attempt_penalty, hints_x_passrate, attempts_x_hints, solved]])
#
#     observed = model.predict_proba(X)[0][1]
#     return round(float(observed), 4)


def update_bkt(current_p_l, observed):
    """
    Update knowledge probability using Bayes theorem.

    Args:
        current_p_l: current probability user knows this topic (0 to 1)
        observed: performance score from calculate_observed (0 to 1)

    Returns:
        new_p_l: updated probability (0 to 1)
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
        new_p_l = p_l_given_obs + (1 - p_l_given_obs) * P_T
    else:
        new_p_l = p_l_given_obs
    return round(min(1.0, max(0.0, new_p_l)), 4)

def process_submission(submission, user_mastery):
    """
    Process a submission and update BKT mastery for all related topics.

    Args:
        submission: dict with userId, problemId, problemTopics,
                    verdict, testCasesPassed, totalTestCases,
                    hintsUsed, submissionCount, normalisedScore
        user_mastery: dict of {topic_slug: p_l} for this user

    Returns:
        updated_mastery: dict of {topic_slug: new_p_l}
        mastered_topics: list of topics that crossed mastery threshold
        results: detailed results per topic
    """
    # Use the topics supplied by the backend instead of looking them up
    # from the static mapping.
    problem_topics = submission.get("problemTopics")

    if problem_topics:
       topics = [
        topic["topicId"]
        for topic in problem_topics
    ]
    else:
       problem_id = submission["problemId"]
       topics = problem_to_topics.get(problem_id, [])
    if not topics:
       return user_mastery, [], []

    # Calculate observed score
    observed = calculate_observed(
        verdict=submission["verdict"],
        hints_taken=submission.get("hintsUsed", 0),
        test_cases_passed=submission.get("testCasesPassed", 0),
        total_test_cases=submission.get("totalTestCases", 1),
        submission_count=submission.get("submissionCount", 1),
        normalised_score=submission.get("normalisedScore", 0.0),
    )

    updated_mastery = dict(user_mastery)
    mastered_topics = []
    results = []

    for topic in topics:
        # Get current P(L) or use default
        current_p_l = user_mastery.get(topic, DEFAULT_P_L["branch"])

        # Update BKT
        new_p_l = update_bkt(current_p_l, observed)
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
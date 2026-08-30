"""
Submission score formulation (ML-owned).

The backend forwards the full behavioural telemetry for a submission and this
module turns it into the single 0..1 submission score the platform displays and
awards XP from. It is the Python counterpart of the backend's
`src/server/scoring.ts::calculateScore` and MUST stay in weight-parity with it
(see `src/shared/data/const.ts` for the canonical weight table) -- the backend
keeps its local implementation only as an offline fallback for when this service
is unreachable; when ML is up, the score it returns here is authoritative.

This is deliberately SEPARATE from `telemetry.py::compute_telemetry_signal`
(the BKT/HLR learning signal). Mastery and the displayed submission score are
two different questions over the same telemetry, exactly as the backend splits
them, so changing one never silently perturbs the other.

    raw_score =
        0.25 * first_pass
      + 0.15 * approach_score
      + 0.15 * fluency_score
      + 0.15 * edge_case_score
      + 0.10 * speed_score
      + 0.10 * (1 - hint_penalty)
      + 0.10 * solution_optimality

    final_score = 0.8 * previous_score + 0.2 * raw_score
"""

from __future__ import annotations

from typing import Optional

_W_FIRST_PASS = 0.25
_W_APPROACH = 0.15
_W_FLUENCY = 0.15
_W_EDGE_CASE = 0.15
_W_SPEED = 0.10
_W_HINT = 0.10
_W_SOLUTION_OPTIMALITY = 0.10

_MAX_MAJOR_REWRITES = 3
_HINT_COUNT_CAP = 3

_SMOOTHING_OLD = 0.8
_SMOOTHING_NEW = 0.2

_NEUTRAL = 0.5


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return lo
    if v != v:  # NaN
        return lo
    return max(lo, min(hi, v))


def _round(value: float, decimals: int = 4) -> float:
    return round(value, decimals)


def compute_submission_score(
    submission: dict,
    previous_score: Optional[float] = None,
) -> dict:
    """
    Formulate the 0..1 submission score from the submission + its telemetry.

    Args:
        submission: the /update request body (dict). Reads `verdict`,
            `submissionCount` and the optional `telemetry` sub-object.
        previous_score: the user's prior average mastery on this problem's
            topics (0..1), used for the final-score smoothing. Falls back to
            the average of the request's `problemTopics[].currentMastery` when
            not supplied, mirroring the backend's "average prior mastery".

    Returns a dict with the raw/final/normalised score plus the component
    breakdown (for explainability / debugging), all neutral-safe: absent
    telemetry yields neutral component scores, never a penalty.
    """
    t = submission.get("telemetry") or {}

    def num(key: str, fallback: float = 0.0) -> float:
        val = t.get(key)
        try:
            return float(val)
        except (TypeError, ValueError):
            return fallback

    passed = submission.get("verdict") == "OK"
    submission_count = int(submission.get("submissionCount", 1) or 1)

    major_rewrite_count = num("majorRewriteCount", 0.0)
    backspace_count = num("backspaceCount", 0.0)
    total_keystrokes = num("totalKeystrokes", 0.0)
    session_duration = num("sessionDurationSeconds", 0.0)
    hints_used = num("hintsUsed", float(submission.get("hintsUsed", 0) or 0))
    first_hint_at = t.get("firstHintOpenedAtSeconds")
    edge_cases_passed = num(
        "edgeCasesPassed", float(submission.get("testCasesPassed", 0) or 0)
    )
    total_edge_cases = num(
        "totalEdgeCases", float(submission.get("totalTestCases", 0) or 0)
    )
    runtime_percentile = num("runtimePercentile", _NEUTRAL)
    memory_percentile = num("memoryPercentile", _NEUTRAL)
    # Speed needs a benchmark average the backend doesn't yet send; stay neutral
    # exactly like the backend does today (averageTimeSeconds defaults to 0).
    time_taken = session_duration
    average_time = num("averageTimeSeconds", 0.0)

    # first_pass = 1 iff accepted on the very first submit
    first_pass = 1.0 if (passed and submission_count == 1) else 0.0

    # approach_score = 1 - major_rewrites / MAX, floored at 0
    approach_score = _clamp(1 - max(0.0, major_rewrite_count) / _MAX_MAJOR_REWRITES)

    # fluency_score = 1 - backspaces / keystrokes (neutral if no keystroke data)
    fluency_score = (
        _clamp(1 - max(0.0, backspace_count) / total_keystrokes)
        if total_keystrokes > 0
        else _NEUTRAL
    )

    # edge_case_score = edge cases passed / total (neutral if untagged)
    edge_case_score = (
        _clamp(edge_cases_passed / total_edge_cases)
        if total_edge_cases > 0
        else _NEUTRAL
    )

    # speed_score = 1 - user_time / avg_time (neutral if no timing benchmark)
    speed_score = (
        _clamp(1 - time_taken / average_time)
        if average_time > 0 and time_taken > 0
        else _NEUTRAL
    )

    # hint_penalty: earlier/heavier hint use costs more; 0 if no hints
    hint_penalty = 0.0
    if hints_used > 0:
        if first_hint_at is None or session_duration <= 0:
            hint_penalty = 0.5
        else:
            timing_penalty = 1 - float(first_hint_at) / session_duration
            hint_count_multiplier = min(1.0, hints_used / _HINT_COUNT_CAP)
            hint_penalty = _clamp(timing_penalty * hint_count_multiplier)

    # solution_optimality: lower percentile is better
    runtime_score = _clamp(1 - runtime_percentile)
    memory_score = _clamp(1 - memory_percentile)
    solution_optimality = _clamp(0.7 * runtime_score + 0.3 * memory_score)

    raw_score = _clamp(
        _W_FIRST_PASS * first_pass
        + _W_APPROACH * approach_score
        + _W_FLUENCY * fluency_score
        + _W_EDGE_CASE * edge_case_score
        + _W_SPEED * speed_score
        + _W_HINT * (1 - hint_penalty)
        + _W_SOLUTION_OPTIMALITY * solution_optimality
    )

    if previous_score is None:
        masteries = [
            topic.get("currentMastery")
            for topic in submission.get("problemTopics", [])
            if topic.get("currentMastery") is not None
        ]
        previous_score = sum(masteries) / len(masteries) if masteries else 0.0

    prev = _clamp(previous_score)
    final_score = _round(_clamp(_SMOOTHING_OLD * prev + _SMOOTHING_NEW * raw_score))

    return {
        "firstPass": first_pass,
        "approachScore": _round(approach_score),
        "fluencyScore": _round(fluency_score),
        "edgeCaseScore": _round(edge_case_score),
        "speedScore": _round(speed_score),
        "hintPenalty": _round(hint_penalty),
        "runtimeScore": _round(runtime_score),
        "memoryScore": _round(memory_score),
        "solutionOptimality": _round(solution_optimality),
        "rawScore": _round(raw_score),
        "previousScore": _round(prev),
        "finalScore": final_score,
        "normalisedScore": round(final_score * 100),
    }

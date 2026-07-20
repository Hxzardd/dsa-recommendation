"""
Shared telemetry -> performance signal, consumed by both bkt.py and hlr.py.

Previously bkt.py::calculate_observed and hlr.py::calculate_performance were
two independent, divergent formulas over the same submission telemetry (one
a weighted sum, the other multiplicative) -- BKT and HLR could end up being
updated from two different opinions of "how well did the learner do" on the
exact same submission. This module replaces both with one shared signal.

The signal has two multiplied parts:
    raw_perf   -- the direct evidence of competence (score, pass rate, hint
                  use, attempt count) -- same weights as the old
                  calculate_observed, which were already reasonable.
    confidence -- a separate penalty for noisy/thrashing behavior (excessive
                  submissions, heavy hint reliance) that raw_perf's own
                  diminishing-returns terms don't fully capture. This is what
                  makes "excessive wrong attempts" actively hurt the signal
                  rather than merely fail to help it: two learners who both
                  eventually pass can still end up with different signal
                  values depending on how much they thrashed to get there.

Only fields already present on the Submission schema are used (hintsUsed,
submissionCount, testCasesPassed/totalTestCases, normalisedScore, verdict) --
no wire-contract changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Single canonical mastery threshold -- previously duplicated as 0.75 in
# bkt.py/hlr.py/state_update_service.py but drifted to 0.7 in
# user_graph_service.py and UserGraph.mastered_concepts()'s default arg.
MASTERY_THRESHOLD = 0.75

# Weights for the direct-evidence component. Unchanged from the old
# calculate_observed -- normalised_score and pass_rate dominate because
# they're the most direct signal of competence; hints/attempts are
# behavioral proxies, not correctness.
_W_SCORE = 0.40
_W_PASS_RATE = 0.25
_W_HINT = 0.20
_W_ATTEMPT = 0.15

# Confidence-penalty tuning. These kick in on top of raw_perf's own
# diminishing-returns terms, specifically to punish thrashing (lots of
# submissions, heavy hint reliance) rather than just failing to reward it.
_SUBMISSION_FREE_ATTEMPTS = 3     # no confidence penalty for the first N attempts
_SUBMISSION_PENALTY_PER_EXTRA = 0.08
_SUBMISSION_PENALTY_FLOOR = 0.5

_HINT_FREE_HINTS = 2              # no confidence penalty for the first N hints
_HINT_PENALTY_PER_EXTRA = 0.06
_HINT_PENALTY_FLOOR = 0.6

_FAILED_VERDICT_CAP = 0.35

# Difficulty-credit tuning: harder problems solved well count as slightly
# stronger evidence of mastery, trivial ones slightly weaker. Deliberately
# modest (+/-30% at the extremes) -- this is a secondary adjustment on top
# of raw_perf, not a replacement for it. difficulty=None (the default, and
# what every existing caller passes) -> credit=1.0, i.e. no change at all
# from previous behavior.
_DIFFICULTY_CREDIT_MIN = 0.7
_DIFFICULTY_CREDIT_MAX = 1.3


@dataclass
class TelemetrySignal:
    """The one shared per-submission performance signal fed to both BKT and HLR."""
    raw_perf: float
    confidence: float
    difficulty_credit: float
    value: float   # raw_perf * confidence * difficulty_credit, capped for non-OK verdicts -- what callers use


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_telemetry_signal(
    verdict: str,
    hints_taken: int,
    test_cases_passed: int,
    total_test_cases: int,
    submission_count: int,
    normalised_score: float,
    difficulty: Optional[float] = None,
) -> TelemetrySignal:
    """
    Turn raw submission telemetry into one shared performance signal.
    Returns a value between 0.0 and 1.0 (in `.value`), plus the raw/
    confidence/difficulty-credit breakdown for debugging/explainability.

    `difficulty` (0-1, the solved problem's difficulty score) is optional --
    omitting it (the default) leaves the signal exactly as it was before
    difficulty-awareness existed (difficulty_credit=1.0).
    """
    if total_test_cases == 0:
        return TelemetrySignal(raw_perf=0.0, confidence=1.0, difficulty_credit=1.0, value=0.0)

    # Direct-evidence component (same shape as the old calculate_observed).
    w1 = normalised_score if verdict == "OK" else normalised_score * 0.3
    w2 = test_cases_passed / total_test_cases
    w3 = max(0.0, 1 - (hints_taken / 10))
    w4 = max(0.0, 1 - ((submission_count - 1) / 10))
    raw_perf = _W_SCORE * w1 + _W_PASS_RATE * w2 + _W_HINT * w3 + _W_ATTEMPT * w4

    # Confidence penalty -- multiplicative, separate from raw_perf, so
    # thrashing behavior actively pulls the signal down instead of just
    # capping how much it can go up.
    excess_attempts = max(0, submission_count - _SUBMISSION_FREE_ATTEMPTS)
    attempt_confidence = _clamp(
        1 - _SUBMISSION_PENALTY_PER_EXTRA * excess_attempts,
        _SUBMISSION_PENALTY_FLOOR, 1.0,
    )
    excess_hints = max(0, hints_taken - _HINT_FREE_HINTS)
    hint_confidence = _clamp(
        1 - _HINT_PENALTY_PER_EXTRA * excess_hints,
        _HINT_PENALTY_FLOOR, 1.0,
    )
    confidence = attempt_confidence * hint_confidence

    # Difficulty credit: linear from 0.7x (difficulty=0.0, trivial) to 1.3x
    # (difficulty=1.0, hard) around a neutral 1.0x at difficulty=0.5.
    # difficulty=None -> 1.0x, i.e. no-op for every caller that doesn't pass it.
    if difficulty is None:
        difficulty_credit = 1.0
    else:
        difficulty_credit = _clamp(
            0.7 + 0.6 * _clamp(difficulty, 0.0, 1.0),
            _DIFFICULTY_CREDIT_MIN, _DIFFICULTY_CREDIT_MAX,
        )

    value = raw_perf * confidence * difficulty_credit
    if verdict != "OK":
        value = min(_FAILED_VERDICT_CAP, value)

    value = round(min(1.0, max(0.0, value)), 4)
    return TelemetrySignal(
        raw_perf=round(raw_perf, 4), confidence=round(confidence, 4),
        difficulty_credit=round(difficulty_credit, 4), value=value,
    )


def compute_telemetry_signal_from_submission(submission: dict) -> TelemetrySignal:
    """Convenience wrapper matching the submission dict shape bkt.py/hlr.py already use."""
    return compute_telemetry_signal(
        verdict=submission["verdict"],
        hints_taken=submission.get("hintsUsed", 0),
        test_cases_passed=submission.get("testCasesPassed", 0),
        total_test_cases=submission.get("totalTestCases", 1),
        submission_count=submission.get("submissionCount", 1),
        normalised_score=submission.get("normalisedScore", 0.0),
        difficulty=submission.get("problemDifficulty"),
    )

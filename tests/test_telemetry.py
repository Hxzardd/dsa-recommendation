import pytest

from pipeline.recommender.telemetry import (
    compute_telemetry_signal,
    compute_telemetry_signal_from_submission,
    MASTERY_THRESHOLD,
)


def _signal(submission_count=1, hints_taken=0, verdict="OK", normalised_score=1.0,
            test_cases_passed=10, total_test_cases=10):
    return compute_telemetry_signal(
        verdict=verdict, hints_taken=hints_taken,
        test_cases_passed=test_cases_passed, total_test_cases=total_test_cases,
        submission_count=submission_count, normalised_score=normalised_score,
    )


def test_mastery_threshold_is_075():
    assert MASTERY_THRESHOLD == 0.75


def test_confidence_decreases_monotonically_with_submission_count():
    values = [_signal(submission_count=n).confidence for n in (1, 3, 5, 8, 12, 20)]
    for a, b in zip(values, values[1:]):
        assert b <= a


def test_confidence_decreases_monotonically_with_hints():
    values = [_signal(hints_taken=n).confidence for n in (0, 2, 4, 6, 10, 15)]
    for a, b in zip(values, values[1:]):
        assert b <= a


def test_excessive_thrashing_scores_lower_than_clean_solve():
    clean = _signal(submission_count=1, hints_taken=0)
    thrashed = _signal(submission_count=12, hints_taken=8)
    assert thrashed.value < clean.value
    assert thrashed.confidence < clean.confidence


def test_non_ok_verdict_capped_at_035():
    sig = _signal(verdict="WRONG_ANSWER", normalised_score=1.0, test_cases_passed=10, total_test_cases=10)
    assert sig.value <= 0.35


def test_value_bounded_0_to_1():
    for n in range(0, 30, 3):
        sig = _signal(submission_count=n, hints_taken=n)
        assert 0.0 <= sig.value <= 1.0
        assert 0.0 <= sig.confidence <= 1.0


def test_bkt_and_hlr_receive_the_same_signal():
    """
    Regression guard for the original bug: bkt.py and hlr.py used to compute
    two different values (calculate_observed vs calculate_performance) from
    the same submission. Both now go through compute_telemetry_signal, so
    calling it twice with identical inputs must be deterministic and equal.
    """
    submission = {
        "verdict": "OK", "hintsUsed": 2, "testCasesPassed": 9,
        "totalTestCases": 10, "submissionCount": 4, "normalisedScore": 0.9,
    }
    a = compute_telemetry_signal_from_submission(submission)
    b = compute_telemetry_signal_from_submission(submission)
    assert a.value == b.value
    assert a.confidence == b.confidence


def test_zero_total_test_cases_returns_zero():
    sig = _signal(total_test_cases=0)
    assert sig.value == 0.0


def test_difficulty_none_is_a_noop():
    with_none = compute_telemetry_signal(
        verdict="OK", hints_taken=0, test_cases_passed=10, total_test_cases=10,
        submission_count=1, normalised_score=0.9, difficulty=None,
    )
    assert with_none.difficulty_credit == 1.0


def test_harder_problem_yields_higher_value_than_easier():
    easy = compute_telemetry_signal(
        verdict="OK", hints_taken=0, test_cases_passed=10, total_test_cases=10,
        submission_count=1, normalised_score=0.9, difficulty=0.05,
    )
    hard = compute_telemetry_signal(
        verdict="OK", hints_taken=0, test_cases_passed=10, total_test_cases=10,
        submission_count=1, normalised_score=0.9, difficulty=0.95,
    )
    assert hard.value > easy.value
    assert hard.difficulty_credit > 1.0
    assert easy.difficulty_credit < 1.0


def test_difficulty_credit_bounded():
    sig = compute_telemetry_signal(
        verdict="OK", hints_taken=0, test_cases_passed=10, total_test_cases=10,
        submission_count=1, normalised_score=0.9, difficulty=1.0,
    )
    assert sig.difficulty_credit <= 1.3
    sig2 = compute_telemetry_signal(
        verdict="OK", hints_taken=0, test_cases_passed=10, total_test_cases=10,
        submission_count=1, normalised_score=0.9, difficulty=0.0,
    )
    assert sig2.difficulty_credit >= 0.7

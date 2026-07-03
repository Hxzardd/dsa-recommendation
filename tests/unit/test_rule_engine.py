"""Rule engine tests."""

from app.models.request_schemas import AnalyzeRequest
from app.parser.normalizer import normalize
from app.rule_engine.engine import run_rules
from app.rule_engine.rules import (
    rule_accepted,
    rule_compilation_error,
    rule_empty_output,
    rule_memory_limit_exceeded,
    rule_runtime_error_signature,
    rule_time_limit_exceeded,
    rule_wrong_answer,
)
from tests.fixtures.sample_payloads import VALID_WRONG_ANSWER_PAYLOAD


def _normalized(payload: dict) -> object:
    """Build a normalized submission from a payload dictionary."""

    return normalize(AnalyzeRequest.model_validate(payload))


def _wrong_answer_payload() -> dict:
    """Build a wrong-answer payload for rule behavior tests."""

    return VALID_WRONG_ANSWER_PAYLOAD | {
        "verdict": "wrong_answer",
        "test_summary": {
            "total_test_cases": 30,
            "passed_test_cases": 24,
            "failed_test_cases": 6,
        },
        "sample_failed_cases": [
            {
                "stdin": "nums=[1]\ntarget=1",
                "expected_output": "0",
                "actual_output": "-1",
            },
        ],
    }


def test_accepted_rule() -> None:
    """Accepted submissions do not need error analysis."""

    submission = _normalized(
        VALID_WRONG_ANSWER_PAYLOAD
        | {
            "verdict": "accepted",
            "test_summary": {
                "total_test_cases": 30,
                "passed_test_cases": 30,
                "failed_test_cases": 0,
            },
            "sample_failed_cases": [],
        },
    )

    result = rule_accepted(submission)

    assert result is not None
    assert result.error_category is None
    assert result.confidence == "high"
    assert result.deterministic_feedback == (
        "Submission passed all test cases — no error analysis needed."
    )
    assert result.deterministic_hint is not None
    assert "accepted" in result.note


def test_compilation_error_rule() -> None:
    """Compilation errors produce deterministic feedback and hint."""

    submission = _normalized(
        VALID_WRONG_ANSWER_PAYLOAD
        | {
            "verdict": "compilation_error",
            "test_summary": {
                "total_test_cases": 0,
                "passed_test_cases": 0,
                "failed_test_cases": 0,
            },
            "sample_failed_cases": [],
            "compile_output": "SyntaxError: invalid syntax\n",
            "stdout": "",
        },
    )

    result = rule_compilation_error(submission)

    assert result is not None
    assert result.error_category == "compilation_error"
    assert result.deterministic_feedback is not None
    assert result.deterministic_hint is not None
    assert result.confidence == "high"


def test_empty_output_rule() -> None:
    """Empty actual output in a representative failed case is detected."""

    submission = _normalized(
        VALID_WRONG_ANSWER_PAYLOAD
        | {
            "sample_failed_cases": [
                {
                    "stdin": "1\n",
                    "expected_output": "42\n",
                    "actual_output": "",
                },
            ],
        },
    )

    result = rule_empty_output(submission)

    assert result is not None
    assert result.confidence == "medium"
    assert result.deterministic_hint is not None


def test_time_limit_rule() -> None:
    """TLE verdicts are categorized deterministically."""

    submission = _normalized(
        VALID_WRONG_ANSWER_PAYLOAD
        | {
            "verdict": "time_limit_exceeded",
            "test_summary": VALID_WRONG_ANSWER_PAYLOAD["test_summary"],
        },
    )

    result = rule_time_limit_exceeded(submission)

    assert result is not None
    assert result.error_category == "time_limit_exceeded"


def test_memory_limit_rule() -> None:
    """MLE verdicts are categorized deterministically."""

    submission = _normalized(
        VALID_WRONG_ANSWER_PAYLOAD
        | {
            "verdict": "memory_limit_exceeded",
            "test_summary": VALID_WRONG_ANSWER_PAYLOAD["test_summary"],
        },
    )

    result = rule_memory_limit_exceeded(submission)

    assert result is not None
    assert result.error_category == "memory_limit_exceeded"


def test_runtime_error_signature_rule() -> None:
    """Known runtime error signatures produce deterministic hints."""

    submission = _normalized(
        VALID_WRONG_ANSWER_PAYLOAD
        | {
            "verdict": "runtime_error",
            "stderr": "IndexError: list index out of range\n",
        },
    )

    result = rule_runtime_error_signature(submission)

    assert result is not None
    assert result.error_category == "runtime_error"
    assert "index" in (result.deterministic_hint or "").lower()


def test_wrong_answer_rule() -> None:
    """Wrong answer mismatches are recorded without category guessing."""

    payload = _wrong_answer_payload()
    submission = _normalized(payload)
    summary = payload["test_summary"]
    first_failed_case = payload["sample_failed_cases"][0]

    result = rule_wrong_answer(submission)

    assert result is not None
    assert result.error_category is None
    assert f"{summary['passed_test_cases']}/{summary['total_test_cases']}" in result.note
    assert f"expected {first_failed_case['expected_output']!r}" in result.note
    assert f"got {first_failed_case['actual_output']!r}" in result.note


def test_compilation_error_short_circuits_llm() -> None:
    """High-confidence compilation errors do not need the LLM."""

    submission = _normalized(
        VALID_WRONG_ANSWER_PAYLOAD
        | {
            "verdict": "compilation_error",
            "test_summary": {
                "total_test_cases": 0,
                "passed_test_cases": 0,
                "failed_test_cases": 0,
            },
            "sample_failed_cases": [],
            "compile_output": "SyntaxError: invalid syntax\n",
            "stdout": "",
        },
    )

    outcome = run_rules(submission)

    assert outcome.needs_llm is False
    assert outcome.error_category == "compilation_error"
    assert outcome.deterministic_feedback is not None
    assert outcome.deterministic_hint is not None


def test_wrong_answer_requires_llm() -> None:
    """Wrong answer sample mismatches still need LLM explanation."""

    submission = _normalized(_wrong_answer_payload())

    outcome = run_rules(submission)

    assert outcome.needs_llm is True
    assert outcome.error_category is None
    assert any("wrong_answer" in note for note in outcome.rule_notes)


def test_accepted_short_circuits_llm() -> None:
    """Accepted submissions short-circuit before other rules."""

    submission = _normalized(
        VALID_WRONG_ANSWER_PAYLOAD
        | {
            "verdict": "accepted",
            "test_summary": {
                "total_test_cases": 30,
                "passed_test_cases": 30,
                "failed_test_cases": 0,
            },
            "sample_failed_cases": [
                {
                    "stdin": "1\n",
                    "expected_output": "42\n",
                    "actual_output": "",
                },
            ],
        },
    )

    outcome = run_rules(submission)

    assert outcome.needs_llm is False
    assert outcome.error_category is None
    assert outcome.deterministic_feedback == (
        "Submission passed all test cases — no error analysis needed."
    )
    assert outcome.rule_notes == [
        "accepted: Submission passed all test cases — no error analysis needed."
    ]


def test_compilation_error_has_highest_priority():
    payload = VALID_WRONG_ANSWER_PAYLOAD | {
        "verdict": "compilation_error",
        "compile_output": "SyntaxError",
        "sample_failed_cases": [],
    }

    request = AnalyzeRequest.model_validate(payload)
    submission = normalize(request)
    outcome = run_rules(submission)

    assert outcome.error_category == "compilation_error"
    assert outcome.needs_llm is False

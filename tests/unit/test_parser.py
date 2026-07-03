"""Parser normalization tests."""
import pytest
from pydantic import ValidationError

from app.models.request_schemas import AnalyzeRequest
from app.parser.normalizer import normalize
from tests.fixtures.sample_payloads import VALID_WRONG_ANSWER_PAYLOAD


def test_normalizes_sample_fixture() -> None:
    """The canonical sample payload normalizes into the domain model."""

    request = AnalyzeRequest.model_validate(VALID_WRONG_ANSWER_PAYLOAD)

    submission = normalize(request)

    assert submission.submission_id == request.submission_id
    assert (
        submission.test_summary.failed_test_cases
        == VALID_WRONG_ANSWER_PAYLOAD["test_summary"]["failed_test_cases"]
    )
    assert len(submission.sample_failed_cases) == len(
        VALID_WRONG_ANSWER_PAYLOAD["sample_failed_cases"],
    )
    assert submission.output_diff_summary is not None


def test_trims_output_whitespace() -> None:
    """Judge text fields and sample case outputs are trimmed consistently."""

    payload = VALID_WRONG_ANSWER_PAYLOAD | {
        "stdout": "8\n\n",
        "stderr": "Traceback\n",
        "sample_failed_cases": [
            {
                "stdin": "5\n1 2 3 4 5\n\n",
                "expected_output": "9\n",
                "actual_output": "8\n",
            },
        ],
    }
    request = AnalyzeRequest.model_validate(payload)

    submission = normalize(request)

    assert submission.stdout == "8"
    assert submission.stderr == "Traceback"
    assert submission.sample_failed_cases[0].stdin == "5\n1 2 3 4 5"
    assert submission.sample_failed_cases[0].expected_output == "9"
    assert submission.sample_failed_cases[0].actual_output == "8"


def test_output_diff_summary_none_for_accepted() -> None:
    """Accepted submissions do not get a mismatch summary."""

    payload = VALID_WRONG_ANSWER_PAYLOAD | {
        "verdict": "accepted",
        "test_summary": {
            "total_test_cases": VALID_WRONG_ANSWER_PAYLOAD["test_summary"]["total_test_cases"],
            "passed_test_cases": VALID_WRONG_ANSWER_PAYLOAD["test_summary"]["total_test_cases"],
            "failed_test_cases": 0,
        },
        "sample_failed_cases": [],
        "actual_output": "",
        "stdout": "9\n",
    }
    request = AnalyzeRequest.model_validate(payload)

    submission = normalize(request)

    assert submission.output_diff_summary is None


def test_output_diff_summary_handles_length_mismatch() -> None:
    """Output length mismatches are summarized deterministically."""

    payload = VALID_WRONG_ANSWER_PAYLOAD | {
        "verdict": "wrong_answer",
        "sample_failed_cases": [
            {
                "stdin": "1\n",
                "expected_output": "1\n2\n",
                "actual_output": "1\n",
            },
        ],
    }
    request = AnalyzeRequest.model_validate(payload)

    submission = normalize(request)

    assert submission.output_diff_summary == (
        "case 1: output length mismatch: expected 2 line(s), got 1 line(s)"
    )

def test_compilation_error_normalizes_correctly() -> None:
    payload = VALID_WRONG_ANSWER_PAYLOAD | {
        "verdict": "compilation_error",
        "compile_output": "SyntaxError: invalid syntax",
        "stdout": "",
        "stderr": "",
        "test_summary": {
            "total_test_cases": 0,
            "passed_test_cases": 0,
            "failed_test_cases": 0,
        },
        "sample_failed_cases": [],
    }

    request = AnalyzeRequest.model_validate(payload)
    submission = normalize(request)

    assert submission.verdict == "compilation_error"
    assert submission.compile_output == "SyntaxError: invalid syntax"
    assert submission.sample_failed_cases == []
    assert submission.output_diff_summary is None

def test_runtime_error_normalizes_correctly() -> None:
    payload = VALID_WRONG_ANSWER_PAYLOAD | {
        "verdict": "runtime_error",
        "stderr": "IndexError: list index out of range",
        "sample_failed_cases": [],
    }

    request = AnalyzeRequest.model_validate(payload)
    submission = normalize(request)

    assert submission.verdict == "runtime_error"
    assert "IndexError" in submission.stderr

def test_multiple_failed_cases_are_preserved() -> None:
    """Multiple representative failed test cases are preserved during normalization."""

    payload = VALID_WRONG_ANSWER_PAYLOAD | {
        "sample_failed_cases": [
            {
                "stdin": "1\n",
                "expected_output": "2\n",
                "actual_output": "1\n",
            },
            {
                "stdin": "2\n",
                "expected_output": "4\n",
                "actual_output": "3\n",
            },
            {
                "stdin": "3\n",
                "expected_output": "6\n",
                "actual_output": "5\n",
            },
        ],
        "test_summary": {
            "total_test_cases": 10,
            "passed_test_cases": 7,
            "failed_test_cases": 3,
        },
    }

    request = AnalyzeRequest.model_validate(payload)
    submission = normalize(request)

    assert len(submission.sample_failed_cases) == 3

    assert submission.sample_failed_cases[0].expected_output == "2"
    assert submission.sample_failed_cases[1].expected_output == "4"
    assert submission.sample_failed_cases[2].expected_output == "6"

def test_empty_failed_case_list_is_supported() -> None:
    """Normalization succeeds when no representative failed cases are present."""

    payload = VALID_WRONG_ANSWER_PAYLOAD | {
        "sample_failed_cases": [],
        "test_summary": {
            "total_test_cases": 10,
            "passed_test_cases": 10,
            "failed_test_cases": 0,
        },
        "verdict": "accepted",
    }

    request = AnalyzeRequest.model_validate(payload)
    submission = normalize(request)

    assert submission.sample_failed_cases == []
    assert submission.output_diff_summary is None

def test_missing_required_field_raises_validation_error() -> None:
    """Missing required request fields should fail validation."""

    payload = VALID_WRONG_ANSWER_PAYLOAD.copy()
    payload.pop("source_code")

    with pytest.raises(ValidationError):
        AnalyzeRequest.model_validate(payload)

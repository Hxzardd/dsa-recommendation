"""Prompt builder tests."""

from types import SimpleNamespace

from app.models.request_schemas import AnalyzeRequest
from app.parser.normalizer import normalize
from app.prompt_builder import builder
from app.prompt_builder.builder import build_prompt
from app.rule_engine.engine import run_rules
from tests.fixtures.sample_payloads import VALID_WRONG_ANSWER_PAYLOAD


def test_prompt_contains_source_code_verbatim() -> None:
    """The prompt includes the submitted source code unchanged."""

    request = AnalyzeRequest.model_validate(VALID_WRONG_ANSWER_PAYLOAD)
    submission = normalize(request)
    outcome = run_rules(submission)

    prompt = build_prompt(submission, outcome)

    assert VALID_WRONG_ANSWER_PAYLOAD["source_code"] in prompt.user


def test_prompt_contains_json_schema_instruction() -> None:
    """The system prompt includes the required JSON response schema."""

    request = AnalyzeRequest.model_validate(VALID_WRONG_ANSWER_PAYLOAD)
    submission = normalize(request)
    outcome = run_rules(submission)

    prompt = build_prompt(submission, outcome)

    assert '"feedback_text": "string"' in prompt.system
    assert '"hint_text": "string"' in prompt.system
    assert '"error_category": "one of:' in prompt.system


def test_prompt_forbids_full_solution() -> None:
    """The system prompt explicitly prevents solution leakage."""

    request = AnalyzeRequest.model_validate(VALID_WRONG_ANSWER_PAYLOAD)
    submission = normalize(request)
    outcome = run_rules(submission)

    prompt = build_prompt(submission, outcome)

    assert "Never provide a full corrected solution" in prompt.system
    assert "working code that solves the problem" in prompt.system


def test_prompt_includes_test_summary_and_failed_cases() -> None:
    """The user prompt includes aggregate and representative failure context."""

    payload = VALID_WRONG_ANSWER_PAYLOAD | {
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
    request = AnalyzeRequest.model_validate(payload)
    submission = normalize(request)
    outcome = run_rules(submission)
    summary = payload["test_summary"]

    prompt = build_prompt(submission, outcome)

    assert f"total_test_cases={summary['total_test_cases']}" in prompt.user
    assert f"passed_test_cases={summary['passed_test_cases']}" in prompt.user
    assert f"failed_test_cases={summary['failed_test_cases']}" in prompt.user
    assert "expected_output:" in prompt.user
    assert "actual_output:" in prompt.user


def test_truncation_preserves_source_over_sample_stdin(monkeypatch) -> None:
    """Large sample stdin is removed before source code is truncated."""

    source_code = "def solve():\n    print(42)"
    large_stdin = "x" * 5000
    payload = VALID_WRONG_ANSWER_PAYLOAD | {
        "verdict": "wrong_answer",
        "source_code": source_code,
        "sample_failed_cases": [
            {
                "stdin": large_stdin,
                "expected_output": "42\n",
                "actual_output": "41\n",
            },
        ],
    }
    request = AnalyzeRequest.model_validate(payload)
    submission = normalize(request)
    outcome = run_rules(submission)
    monkeypatch.setattr(
        builder,
        "get_settings",
        lambda: SimpleNamespace(prompt_max_chars=1800),
    )

    prompt = build_prompt(submission, outcome)

    assert source_code in prompt.user
    assert large_stdin not in prompt.user
    assert len(prompt.system) + len(prompt.user) <= 1800

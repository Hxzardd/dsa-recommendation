"""Normalize validated analyze requests into internal domain objects."""

from app.models.domain import NormalizedSubmission
from app.models.request_schemas import AnalyzeRequest, SampleFailedCase


def _trim_text(value: str) -> str:
    """Trim trailing whitespace that commonly appears in judge output."""

    return value.rstrip()


def _summarize_case_diff(case: SampleFailedCase, index: int) -> str | None:
    """Create a deterministic diff summary for one representative failed case."""

    expected = case.expected_output
    actual = case.actual_output

    if expected == actual:
        return None

    expected_lines = expected.splitlines()
    actual_lines = actual.splitlines()

    if len(expected_lines) != len(actual_lines):
        return (
            f"case {index}: output length mismatch: expected {len(expected_lines)} "
            f"line(s), got {len(actual_lines)} line(s)"
        )

    for line_number, (expected_line, actual_line) in enumerate(
        zip(expected_lines, actual_lines, strict=True),
        start=1,
    ):
        if expected_line != actual_line:
            return (
                f"case {index}: expected {expected_line!r} but got {actual_line!r} "
                f"on line {line_number}"
            )

    return f"case {index}: expected output differs from actual output"


def _build_output_diff_summary(
    verdict: str,
    sample_failed_cases: list[SampleFailedCase],
) -> str | None:
    """Build a deterministic summary across representative failed cases."""

    if verdict == "accepted":
        return None

    summaries = [
        summary
        for index, failed_case in enumerate(sample_failed_cases, start=1)
        if (summary := _summarize_case_diff(failed_case, index)) is not None
    ]
    if not summaries:
        return None

    return "; ".join(summaries)


def normalize(request: AnalyzeRequest) -> NormalizedSubmission:
    """Normalize a validated backend request into a pipeline domain object."""

    normalized_failed_cases = [
        SampleFailedCase(
            stdin=_trim_text(failed_case.stdin),
            expected_output=_trim_text(failed_case.expected_output),
            actual_output=_trim_text(failed_case.actual_output),
        )
        for failed_case in request.sample_failed_cases
    ]
    normalized_stderr_source = request.stderr
    if not normalized_stderr_source and request.verdict == "compilation_error":
        normalized_stderr_source = request.compile_output

    return NormalizedSubmission(
        submission_id=request.submission_id,
        problem_id=request.problem_id,
        user_id=request.user_id,
        language=request.language,
        verdict=request.verdict,
        source_code=request.source_code,
        test_summary=request.test_summary,
        sample_failed_cases=normalized_failed_cases,
        stdout=_trim_text(request.stdout),
        stderr=_trim_text(request.stderr),
        compile_output=_trim_text(request.compile_output),
        execution_time_ms=request.execution_time_ms,
        memory_kb=request.memory_kb,
        submitted_at=request.submitted_at,
        normalized_stderr=_trim_text(normalized_stderr_source) or None,
        output_diff_summary=_build_output_diff_summary(
            request.verdict,
            normalized_failed_cases,
        ),
        is_deterministic_case=False,
    )

"""Deterministic rules for common submission outcomes."""

from collections.abc import Callable

from app.models.domain import NormalizedSubmission, RuleResult

Rule = Callable[[NormalizedSubmission], RuleResult | None]


def rule_accepted(sub: NormalizedSubmission) -> RuleResult | None:
    """Short-circuit accepted submissions because no failure analysis is needed."""

    if sub.verdict != "accepted":
        return None

    return RuleResult(
        matched=True,
        error_category=None,
        deterministic_feedback="Submission passed all test cases — no error analysis needed.",
        deterministic_hint="No changes are needed for this accepted submission.",
        confidence="high",
        note="accepted: Submission passed all test cases — no error analysis needed.",
    )


def rule_compilation_error(sub: NormalizedSubmission) -> RuleResult | None:
    """Short-circuit clear compilation errors without an LLM call."""

    if sub.verdict != "compilation_error":
        return None

    compiler_message = sub.normalized_stderr or sub.compile_output or "The code did not compile."
    feedback = (
        "Your submission failed before it could run because the compiler reported an error. "
        f"Compiler output: {compiler_message}"
    )
    return RuleResult(
        matched=True,
        error_category="compilation_error",
        deterministic_feedback=feedback,
        deterministic_hint="Start by fixing the first compiler error, then run the sample again.",
        confidence="high",
        note="compilation_error: compile output explains the failure deterministically",
    )


def rule_empty_output(sub: NormalizedSubmission) -> RuleResult | None:
    """Detect failed cases where the program produced no output."""

    for index, failed_case in enumerate(sub.sample_failed_cases, start=1):
        if not failed_case.actual_output.strip() and failed_case.expected_output.strip():
            return RuleResult(
                matched=True,
                error_category=None,
                deterministic_feedback=None,
                deterministic_hint=(
                    f"Case {index} expected output, but your program produced no output. "
                    "Check whether the result is printed on every path."
                ),
                confidence="medium",
                note=f"empty_output: sample failed case {index} produced no output",
            )

    return None


def rule_time_limit_exceeded(sub: NormalizedSubmission) -> RuleResult | None:
    """Categorize time limit exceeded verdicts."""

    if sub.verdict != "time_limit_exceeded":
        return None

    return RuleResult(
        matched=True,
        error_category="time_limit_exceeded",
        deterministic_feedback=None,
        deterministic_hint=(
            "The submission exceeded the time limit. Revisit the loop bounds and the "
            "overall time complexity."
        ),
        confidence="medium",
        note="time_limit_exceeded: verdict deterministically identifies the category",
    )


def rule_memory_limit_exceeded(sub: NormalizedSubmission) -> RuleResult | None:
    """Categorize memory limit exceeded verdicts."""

    if sub.verdict != "memory_limit_exceeded":
        return None

    return RuleResult(
        matched=True,
        error_category="memory_limit_exceeded",
        deterministic_feedback=None,
        deterministic_hint=(
            "The submission exceeded the memory limit. Check whether large structures "
            "can be avoided or streamed."
        ),
        confidence="medium",
        note="memory_limit_exceeded: verdict deterministically identifies the category",
    )


def rule_runtime_error_signature(sub: NormalizedSubmission) -> RuleResult | None:
    """Detect common runtime error signatures from stderr."""

    if sub.verdict != "runtime_error":
        return None

    stderr = (sub.normalized_stderr or sub.stderr).lower()
    signatures = {
        "indexerror": "An index went outside the valid range.",
        "zerodivisionerror": "A division by zero occurred.",
        "recursionerror": "The recursion depth grew too large.",
        "keyerror": "A dictionary key was accessed but was not present.",
        "typeerror": "An operation used a value of an incompatible type.",
    }
    for signature, hint in signatures.items():
        if signature in stderr:
            return RuleResult(
                matched=True,
                error_category="runtime_error",
                deterministic_feedback=None,
                deterministic_hint=hint,
                confidence="medium",
                note=f"runtime_error_signature: detected {signature}",
            )

    return None


def rule_wrong_answer(sub: NormalizedSubmission) -> RuleResult | None:
    """Record observed wrong-answer mismatches without inferring root cause."""

    if sub.verdict != "wrong_answer":
        return None

    notes = [
        "wrong_answer: "
        f"{sub.test_summary.passed_test_cases}/{sub.test_summary.total_test_cases} "
        "test cases passed"
    ]
    if sub.output_diff_summary:
        notes.append(sub.output_diff_summary)

    return RuleResult(
        matched=True,
        error_category=None,
        deterministic_feedback=None,
        deterministic_hint=None,
        confidence="low",
        note="; ".join(notes),
    )


REGISTERED_RULES: tuple[Rule, ...] = (
    rule_accepted,
    rule_compilation_error,
    rule_empty_output,
    rule_time_limit_exceeded,
    rule_memory_limit_exceeded,
    rule_runtime_error_signature,
    rule_wrong_answer,
)

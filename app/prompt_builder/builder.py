"""Build mentor-style prompts for LLM analysis."""

from app.config.settings import get_settings
from app.logging.logger import get_logger
from app.models.domain import LLMPrompt, NormalizedSubmission, RuleEngineOutcome

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a patient coding mentor for a learner.
Never provide a full corrected solution or working code that solves the problem.
Explain the mistake conceptually and give a nudge or hint, not the answer.
Always respond in strict JSON only, matching this exact schema:
{
  "feedback_text": "string",
  "hint_text": "string",
  "error_category": "one of: wrong_answer_logic | off_by_one | edge_case_missing | \
wrong_algorithm | time_limit_exceeded | memory_limit_exceeded | runtime_error | \
compilation_error | unknown",
  "reasoning_quality": "one of: strong | partial | weak | unknown",
  "concept_gaps": ["array", "of", "short", "strings"]
}
No markdown fences, no prose before or after the JSON."""


def _format_failed_cases(sub: NormalizedSubmission, include_stdin: bool) -> str:
    """Format representative failed cases for the user prompt."""

    if not sub.sample_failed_cases:
        return "No representative failed cases were provided."

    sections: list[str] = []
    for index, failed_case in enumerate(sub.sample_failed_cases[:3], start=1):
        lines = [f"Case {index}:"]
        if include_stdin and failed_case.stdin:
            lines.append(f"stdin:\n{failed_case.stdin}")
        lines.append(f"expected_output:\n{failed_case.expected_output}")
        lines.append(f"actual_output:\n{failed_case.actual_output}")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def _compose_user_prompt(
    sub: NormalizedSubmission,
    rule_outcome: RuleEngineOutcome,
    include_case_stdin: bool,
    include_stderr: bool,
) -> str:
    """Compose the user prompt with optional non-essential sections."""

    parts = [
        f"Language: {sub.language}",
        f"Verdict: {sub.verdict}",
        "Submitted source code:",
        sub.source_code,
        "Test summary:",
        (
            f"total_test_cases={sub.test_summary.total_test_cases}, "
            f"passed_test_cases={sub.test_summary.passed_test_cases}, "
            f"failed_test_cases={sub.test_summary.failed_test_cases}"
        ),
        "Representative failed cases:",
        _format_failed_cases(sub, include_stdin=include_case_stdin),
    ]

    if sub.output_diff_summary:
        parts.extend(["Output diff summary:", sub.output_diff_summary])

    diagnostic_output = sub.normalized_stderr or sub.stderr or sub.compile_output
    if include_stderr and diagnostic_output:
        parts.extend(["stderr / compile_output:", diagnostic_output])

    if rule_outcome.rule_notes:
        parts.extend(
            [
                "Deterministic analysis notes:",
                "\n".join(f"- {note}" for note in rule_outcome.rule_notes),
            ],
        )

    parts.append("Respond with the JSON object only.")
    return "\n\n".join(parts)


def build_prompt(
    sub: NormalizedSubmission,
    rule_outcome: RuleEngineOutcome,
) -> LLMPrompt:
    """Build a strict JSON prompt from a normalized submission and rule outcome."""

    settings = get_settings()
    prompt = _compose_user_prompt(
        sub,
        rule_outcome,
        include_case_stdin=True,
        include_stderr=True,
    )

    if len(SYSTEM_PROMPT) + len(prompt) <= settings.prompt_max_chars:
        return LLMPrompt(system=SYSTEM_PROMPT, user=prompt)

    prompt = _compose_user_prompt(
        sub,
        rule_outcome,
        include_case_stdin=False,
        include_stderr=True,
    )
    if len(SYSTEM_PROMPT) + len(prompt) <= settings.prompt_max_chars:
        logger.warning("prompt truncated sample failed case stdin")
        return LLMPrompt(system=SYSTEM_PROMPT, user=prompt)

    prompt = _compose_user_prompt(
        sub,
        rule_outcome,
        include_case_stdin=False,
        include_stderr=False,
    )
    if len(SYSTEM_PROMPT) + len(prompt) > settings.prompt_max_chars:
        logger.warning("prompt remains over budget after non-essential truncation")
    else:
        logger.warning("prompt truncated stderr and compile output")

    return LLMPrompt(system=SYSTEM_PROMPT, user=prompt)

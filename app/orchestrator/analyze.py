"""Analyze orchestrator — the deep module behind POST /analyze.

Interface: ``analyze_submission(request, llm=None) -> AnalyzeResponse``.
Hidden behind it: input safety, normalization, deterministic rules, prompt
building, the LLM round-trip, output validation, and all degrade/invalid
handling. Callers (the route) and tests cross only this seam; the LLM client is
injected so tests pass a fake.
"""

from __future__ import annotations

from time import perf_counter
from typing import get_args

from app.config.settings import get_settings
from app.llm.client import LLMClient, LLMUnavailableError, get_llm_client
from app.models.domain import NormalizedSubmission, RuleEngineOutcome
from app.models.response_schemas import AnalyzeResponse, ErrorCategory
from app.parser.normalizer import normalize
from app.prompt_builder.builder import build_prompt
from app.rule_engine.engine import run_rules
from app.security.sanitizer import validate_input_safety
from app.validator.response_validator import parse_analyze_json

_VALID_ERROR_CATEGORIES = set(get_args(ErrorCategory))

_FALLBACK_FEEDBACK = "Automated feedback is temporarily unavailable for this submission."
_FALLBACK_HINT = (
    "Re-check your logic against the failing cases; try the smallest counterexample by hand."
)


def _elapsed_ms(start: float) -> int:
    return max(0, round((perf_counter() - start) * 1000))


def _safe_category(value: str | None) -> ErrorCategory:
    return value if value in _VALID_ERROR_CATEGORIES else "unknown"  # type: ignore[return-value]


def _degraded_response(
    sub: NormalizedSubmission,
    outcome: RuleEngineOutcome,
    start: float,
    status: str,
) -> AnalyzeResponse:
    """Build a valid response from deterministic hints when the LLM can't help."""

    return AnalyzeResponse(
        submission_id=sub.submission_id,
        feedback_text=outcome.deterministic_feedback or _FALLBACK_FEEDBACK,
        hint_text=outcome.deterministic_hint or _FALLBACK_HINT,
        error_category=_safe_category(outcome.error_category),
        reasoning_quality="unknown",
        concept_gaps=[],
        processing_status=status,  # type: ignore[arg-type]
        processing_ms=_elapsed_ms(start),
        model_used="rule_engine",
    )


async def analyze_submission(request, llm: LLMClient | None = None) -> AnalyzeResponse:
    """Run the full analysis pipeline for one submission.

    Raises ``UnsafeInputError`` for unsafe/oversized input (mapped to 4xx by the
    route). Every other failure mode (LLM down, invalid model JSON) degrades to a
    valid response rather than raising.
    """

    start = perf_counter()
    settings = get_settings()
    llm = llm or get_llm_client()

    validate_input_safety(request)
    sub = normalize(request)
    outcome = run_rules(sub)

    # Deterministic short-circuit — no LLM needed (accepted, clear compile error…).
    if not outcome.needs_llm:
        return AnalyzeResponse(
            submission_id=sub.submission_id,
            feedback_text=outcome.deterministic_feedback or _FALLBACK_FEEDBACK,
            hint_text=outcome.deterministic_hint or _FALLBACK_HINT,
            error_category=_safe_category(outcome.error_category),
            reasoning_quality="unknown",
            concept_gaps=[],
            processing_status="rule_only",
            processing_ms=_elapsed_ms(start),
            model_used="rule_engine",
        )

    prompt = build_prompt(sub, outcome)
    try:
        raw = await llm.complete(prompt)
    except LLMUnavailableError:
        return _degraded_response(sub, outcome, start, status="error")

    validated = parse_analyze_json(raw, max_concept_gaps=settings.max_concept_gaps)
    if validated is None:
        return _degraded_response(sub, outcome, start, status="llm_output_invalid")

    return AnalyzeResponse(
        submission_id=sub.submission_id,
        feedback_text=validated.feedback_text,
        hint_text=validated.hint_text,
        error_category=validated.error_category,
        reasoning_quality=validated.reasoning_quality,
        concept_gaps=validated.concept_gaps,
        processing_status="completed",
        processing_ms=_elapsed_ms(start),
        model_used=settings.vllm_model,
    )

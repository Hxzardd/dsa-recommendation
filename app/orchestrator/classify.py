"""Classify orchestrator — the deep module behind POST /classify-approach.

Interface: ``classify_approach(request, llm=None) -> ClassifyApproachResponse``.
Builds the classify prompt, runs the injected LLM, validates the JSON (filtering
to the candidate sets), and degrades to confidence 0 when the LLM is unavailable
or returns unusable output — so the backend simply stays weights-only.
"""

from __future__ import annotations

from time import perf_counter

from app.config.settings import get_settings
from app.llm.client import LLMClient, LLMUnavailableError, get_llm_client
from app.models.request_schemas import ClassifyApproachRequest
from app.models.response_schemas import ClassifyApproachResponse
from app.prompt_builder.classify_builder import build_classify_prompt
from app.validator.response_validator import parse_classify_json


def _elapsed_ms(start: float) -> int:
    return max(0, round((perf_counter() - start) * 1000))


def _degraded(
    request: ClassifyApproachRequest,
    start: float,
    status: str,
) -> ClassifyApproachResponse:
    """confidence 0 → backend keeps structural-weights-only crediting."""

    return ClassifyApproachResponse(
        submission_id=request.submission_id,
        matched_approach="unknown",
        used_topics=[],
        used_data_structures=[],
        used_patterns=[],
        confidence=0.0,
        processing_status=status,  # type: ignore[arg-type]
        processing_ms=_elapsed_ms(start),
        model_used="none",
    )


async def classify_approach(
    request: ClassifyApproachRequest,
    llm: LLMClient | None = None,
) -> ClassifyApproachResponse:
    """Classify which candidate topics/patterns the submission actually used."""

    start = perf_counter()
    settings = get_settings()
    llm = llm or get_llm_client()

    prompt = build_classify_prompt(request)
    try:
        raw = await llm.complete(prompt)
    except LLMUnavailableError:
        return _degraded(request, start, "error")

    validated = parse_classify_json(
        raw,
        candidate_topics=request.candidate_topics,
        candidate_patterns=request.candidate_patterns,
        max_used=settings.max_used_items,
    )
    if validated is None:
        return _degraded(request, start, "llm_output_invalid")

    return ClassifyApproachResponse(
        submission_id=request.submission_id,
        matched_approach=validated.matched_approach,
        used_topics=validated.used_topics,
        used_data_structures=validated.used_data_structures,
        used_patterns=validated.used_patterns,
        confidence=validated.confidence,
        processing_status="completed",
        processing_ms=_elapsed_ms(start),
        model_used=settings.vllm_model,
    )

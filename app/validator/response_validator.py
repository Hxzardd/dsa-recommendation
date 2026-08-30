"""Pure validation of raw LLM text into structured, schema-safe fields.

No side effects, no I/O — given raw model text it returns a validated result or
None (when the text can't be parsed at all). Bad enum values are coerced to
"unknown" and list fields are clamped, so a slightly-off model never crashes the
pipeline; a completely unparseable response returns None so the orchestrator can
fall back deterministically.
"""

from __future__ import annotations

import json
import re
from typing import Any, get_args

from pydantic import BaseModel

from app.models.response_schemas import ErrorCategory, ReasoningQuality

_VALID_ERROR_CATEGORIES = set(get_args(ErrorCategory))
_VALID_REASONING = set(get_args(ReasoningQuality))
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


class ValidatedAnalysis(BaseModel):
    """Schema-safe analyze fields extracted from the model output."""

    feedback_text: str
    hint_text: str
    error_category: ErrorCategory
    reasoning_quality: ReasoningQuality
    concept_gaps: list[str]


class ValidatedClassification(BaseModel):
    """Schema-safe approach-classification fields extracted from the model output."""

    matched_approach: str
    used_topics: list[str]
    used_data_structures: list[str]
    used_patterns: list[str]
    confidence: float


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    """Return the first JSON object embedded in ``raw`` (tolerating code fences)."""

    if not raw or not raw.strip():
        return None

    stripped = _FENCE_RE.sub("", raw.strip())
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None

    return parsed if isinstance(parsed, dict) else None


def _as_str(value: Any, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _as_str_list(value: Any, *, limit: int, allowed: set[str] | None = None) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        if allowed is not None and item.lower() not in allowed:
            continue
        out.append(item)
        if len(out) >= limit:
            break
    return out


def parse_analyze_json(raw: str, *, max_concept_gaps: int) -> ValidatedAnalysis | None:
    """Parse/validate LLM analyze output. None when unparseable."""

    data = _extract_json_object(raw)
    if data is None:
        return None

    error_category = _as_str(data.get("error_category")).lower()
    if error_category not in _VALID_ERROR_CATEGORIES:
        error_category = "unknown"

    reasoning_quality = _as_str(data.get("reasoning_quality")).lower()
    if reasoning_quality not in _VALID_REASONING:
        reasoning_quality = "unknown"

    feedback = _as_str(data.get("feedback_text")).strip()
    hint = _as_str(data.get("hint_text")).strip()
    if not feedback and not hint:
        # A response with neither field carries no usable analysis.
        return None

    return ValidatedAnalysis(
        feedback_text=feedback or "No feedback was produced for this submission.",
        hint_text=hint or "Re-check your approach against the failing cases.",
        error_category=error_category,  # type: ignore[arg-type]
        reasoning_quality=reasoning_quality,  # type: ignore[arg-type]
        concept_gaps=_as_str_list(data.get("concept_gaps"), limit=max_concept_gaps),
    )


def parse_classify_json(
    raw: str,
    *,
    candidate_topics: list[str],
    candidate_patterns: list[str],
    max_used: int,
) -> ValidatedClassification | None:
    """Parse/validate LLM classify output, filtering used_* to the candidates."""

    data = _extract_json_object(raw)
    if data is None:
        return None

    topics_allowed = {t.lower() for t in candidate_topics}
    patterns_allowed = {p.lower() for p in candidate_patterns}

    confidence = data.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = 0.0
    confidence = max(0.0, min(1.0, float(confidence)))

    return ValidatedClassification(
        matched_approach=_as_str(data.get("matched_approach"), "other") or "other",
        used_topics=_as_str_list(
            data.get("used_topics"), limit=max_used, allowed=topics_allowed or None
        ),
        used_data_structures=_as_str_list(data.get("used_data_structures"), limit=max_used),
        used_patterns=_as_str_list(
            data.get("used_patterns"), limit=max_used, allowed=patterns_allowed or None
        ),
        confidence=confidence,
    )

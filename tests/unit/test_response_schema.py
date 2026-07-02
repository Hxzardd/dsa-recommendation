"""AnalyzeResponse schema tests."""

import pytest
from pydantic import ValidationError

from app.models.response_schemas import AnalyzeResponse


def test_valid_response_constructs() -> None:
    """A valid response contract constructs successfully."""

    response = AnalyzeResponse(
        submission_id="sub_9f2a1b7c",
        feedback_text="Check how you combine the values before printing.",
        hint_text="Trace the sample by hand and compare each intermediate step.",
        error_category="wrong_answer_logic",
        reasoning_quality="partial",
        concept_gaps=[],
        processing_status="completed",
        processing_ms=12,
        model_used="qwen2.5-coder:7b",
    )

    assert response.processing_status == "completed"


def test_invalid_enum_value_rejected() -> None:
    """Response enum fields reject undocumented values."""

    with pytest.raises(ValidationError):
        AnalyzeResponse(
            submission_id="sub_9f2a1b7c",
            feedback_text="Feedback",
            hint_text="Hint",
            error_category="not_a_category",
            reasoning_quality="partial",
            concept_gaps=[],
            processing_status="completed",
            processing_ms=12,
            model_used="qwen2.5-coder:7b",
        )


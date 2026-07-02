"""Response schemas for the AI analysis service."""

from typing import Literal

from pydantic import BaseModel, Field

ErrorCategory = Literal[
    "wrong_answer_logic",
    "off_by_one",
    "edge_case_missing",
    "wrong_algorithm",
    "time_limit_exceeded",
    "memory_limit_exceeded",
    "runtime_error",
    "compilation_error",
    "unknown",
]
ReasoningQuality = Literal["strong", "partial", "weak", "unknown"]
ProcessingStatus = Literal["completed", "rule_only", "llm_output_invalid", "timeout", "error"]


class AnalyzeResponse(BaseModel):
    """Structured analysis response returned to the backend."""

    submission_id: str
    feedback_text: str
    hint_text: str
    error_category: ErrorCategory
    reasoning_quality: ReasoningQuality
    concept_gaps: list[str] = Field(default_factory=list)
    processing_status: ProcessingStatus
    processing_ms: int = Field(ge=0)
    model_used: str


class ErrorResponse(BaseModel):
    """Stable error envelope for validation and unexpected server errors."""

    error_code: str
    message: str
    submission_id: str | None = None


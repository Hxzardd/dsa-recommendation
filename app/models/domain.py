"""Internal domain models shared between pipeline stages."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from app.models.request_schemas import SampleFailedCase, TestSummary, Verdict


class RuleResult(BaseModel):
    """Result returned by one deterministic rule."""

    matched: bool
    error_category: str | None = None
    deterministic_feedback: str | None = None
    deterministic_hint: str | None = None
    confidence: Literal["high", "medium", "low"]
    note: str


class RuleEngineOutcome(BaseModel):
    """Aggregated result of deterministic rule execution."""

    needs_llm: bool
    error_category: str | None = None
    deterministic_feedback: str | None = None
    deterministic_hint: str | None = None
    rule_notes: list[str]


class LLMPrompt(BaseModel):
    """Prompt parts sent to an LLM provider in later phases."""

    system: str
    user: str


class NormalizedSubmission(BaseModel):
    """Submission after parser normalization.

    This model is intentionally mutable so later pipeline stages can annotate it.
    """

    submission_id: str
    problem_id: str
    user_id: str
    language: str
    verdict: Verdict
    source_code: str
    test_summary: TestSummary
    sample_failed_cases: list[SampleFailedCase]
    stdout: str
    stderr: str
    compile_output: str
    execution_time_ms: int
    memory_kb: int
    submitted_at: datetime
    normalized_stderr: str | None = None
    output_diff_summary: str | None = None
    is_deterministic_case: bool = False

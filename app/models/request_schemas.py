"""Request schemas for the AI analysis service."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.config.settings import get_settings

Verdict = Literal[
    "wrong_answer",
    "time_limit_exceeded",
    "memory_limit_exceeded",
    "runtime_error",
    "compilation_error",
    "accepted",
]


class TestSummary(BaseModel):
    """Backend-provided aggregate test results for the submission."""

    total_test_cases: int = Field(ge=0)
    passed_test_cases: int = Field(ge=0)
    failed_test_cases: int = Field(ge=0)

    @model_validator(mode="after")
    def counts_must_be_consistent(self) -> "TestSummary":
        """Ensure passed and failed counts do not exceed total test cases."""

        if self.passed_test_cases + self.failed_test_cases > self.total_test_cases:
            msg = "passed_test_cases + failed_test_cases must not exceed total_test_cases"
            raise ValueError(msg)
        return self


class SampleFailedCase(BaseModel):
    """Representative failing test case provided by the backend."""

    stdin: str = ""
    expected_output: str = ""
    actual_output: str = ""


class AnalyzeRequest(BaseModel):
    """Request payload sent by the backend for a completed submission."""

    submission_id: str
    problem_id: str
    user_id: str
    language: str
    verdict: Verdict
    source_code: str
    test_summary: TestSummary
    sample_failed_cases: list[SampleFailedCase] = Field(max_length=3)
    stdout: str = ""
    stderr: str = ""
    compile_output: str = ""
    execution_time_ms: int = Field(ge=0)
    memory_kb: int = Field(ge=0)
    submitted_at: datetime

    @field_validator("source_code")
    @classmethod
    def source_code_must_be_safe_size(cls, value: str) -> str:
        """Ensure submitted source is present and within configured limits."""

        if not value.strip():
            msg = "source_code must not be empty"
            raise ValueError(msg)

        max_chars = get_settings().max_source_code_chars
        if len(value) > max_chars:
            msg = f"source_code exceeds maximum length of {max_chars} characters"
            raise ValueError(msg)

        return value

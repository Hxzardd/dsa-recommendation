"""Internal domain models shared between pipeline stages."""

from datetime import datetime

from pydantic import BaseModel

from app.models.request_schemas import Verdict


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
    stdin: str
    expected_output: str
    actual_output: str
    stdout: str
    stderr: str
    compile_output: str
    execution_time_ms: int
    memory_kb: int
    submitted_at: datetime
    normalized_stderr: str | None = None
    output_diff_summary: str | None = None
    is_deterministic_case: bool = False


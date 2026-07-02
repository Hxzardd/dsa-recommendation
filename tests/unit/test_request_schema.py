"""AnalyzeRequest schema tests."""

import pytest
from pydantic import ValidationError

from app.config.settings import get_settings
from app.models.request_schemas import AnalyzeRequest
from tests.fixtures.sample_payloads import VALID_WRONG_ANSWER_PAYLOAD


def test_valid_payload_parses() -> None:
    """The canonical backend payload parses without losing fields."""

    request = AnalyzeRequest.model_validate(VALID_WRONG_ANSWER_PAYLOAD)

    assert request.submission_id == VALID_WRONG_ANSWER_PAYLOAD["submission_id"]
    assert request.verdict == "wrong_answer"
    assert request.submitted_at.isoformat().startswith("2026-06-30T18:04:11")
    assert request.model_dump(mode="json").keys() == VALID_WRONG_ANSWER_PAYLOAD.keys()


def test_invalid_verdict_rejected() -> None:
    """Unknown verdicts are rejected with validation errors."""

    payload = VALID_WRONG_ANSWER_PAYLOAD | {"verdict": "mysterious_failure"}

    with pytest.raises(ValidationError):
        AnalyzeRequest.model_validate(payload)


def test_oversized_source_code_rejected() -> None:
    """Source code longer than the configured cap is rejected."""

    max_chars = get_settings().max_source_code_chars
    payload = VALID_WRONG_ANSWER_PAYLOAD | {"source_code": "x" * (max_chars + 1)}

    with pytest.raises(ValidationError):
        AnalyzeRequest.model_validate(payload)


def test_missing_required_field_rejected() -> None:
    """Required backend contract fields must be present."""

    payload = VALID_WRONG_ANSWER_PAYLOAD.copy()
    payload.pop("submission_id")

    with pytest.raises(ValidationError):
        AnalyzeRequest.model_validate(payload)


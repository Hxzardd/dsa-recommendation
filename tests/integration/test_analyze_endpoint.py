"""POST /analyze endpoint tests (fake LLM injected via monkeypatch)."""

import json

from fastapi.testclient import TestClient

from app.llm.client import FakeLLMClient, LLMUnavailableError
from app.main import app
from app.orchestrator import analyze as analyze_mod
from tests.fixtures.sample_payloads import VALID_WRONG_ANSWER_PAYLOAD

client = TestClient(app)

_GOOD_JSON = json.dumps(
    {
        "feedback_text": "Off-by-one in your bound.",
        "hint_text": "Check the right index.",
        "error_category": "off_by_one",
        "reasoning_quality": "partial",
        "concept_gaps": ["loop_bounds"],
    }
)


def test_analyze_returns_completed(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(analyze_mod, "get_llm_client", lambda: FakeLLMClient(_GOOD_JSON))
    response = client.post("/analyze", json=VALID_WRONG_ANSWER_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["processing_status"] == "completed"
    assert body["error_category"] == "off_by_one"
    assert body["submission_id"] == VALID_WRONG_ANSWER_PAYLOAD["submission_id"]
    for key in ("feedback_text", "hint_text", "reasoning_quality", "concept_gaps", "model_used"):
        assert key in body


def test_analyze_degrades_when_llm_down(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        analyze_mod,
        "get_llm_client",
        lambda: FakeLLMClient(raises=LLMUnavailableError("down")),
    )
    response = client.post("/analyze", json=VALID_WRONG_ANSWER_PAYLOAD)
    assert response.status_code == 200  # never 5xx on LLM outage
    assert response.json()["processing_status"] == "error"


def test_analyze_rejects_empty_source_code() -> None:
    bad = {**VALID_WRONG_ANSWER_PAYLOAD, "source_code": "   "}
    response = client.post("/analyze", json=bad)
    assert response.status_code == 422  # schema validation

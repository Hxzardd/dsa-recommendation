"""POST /classify-approach endpoint tests (fake LLM injected via monkeypatch)."""

import json

from fastapi.testclient import TestClient

from app.llm.client import FakeLLMClient, LLMUnavailableError
from app.main import app
from app.orchestrator import classify as classify_mod

client = TestClient(app)

_BRUTE_FORCE_TWO_SUM = {
    "submission_id": "sub_classify_1",
    "problem_id": "two-sum",
    "language": "python",
    "source_code": (
        "def two_sum(nums, t):\n"
        "    for i in range(len(nums)):\n"
        "        for j in range(i+1, len(nums)):\n"
        "            if nums[i]+nums[j]==t:\n"
        "                return [i, j]\n"
    ),
    "candidate_topics": ["array", "hash_map"],
    "candidate_patterns": ["simulation"],
    "data_structure_tags": ["integer_arrays", "hash_maps"],
}


def test_classify_returns_used_topics(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    raw = json.dumps(
        {
            "matched_approach": "brute_force_nested",
            "used_topics": ["array"],
            "used_data_structures": ["integer_array"],
            "used_patterns": [],
            "confidence": 0.9,
        }
    )
    monkeypatch.setattr(classify_mod, "get_llm_client", lambda: FakeLLMClient(raw))
    response = client.post("/classify-approach", json=_BRUTE_FORCE_TWO_SUM)
    assert response.status_code == 200
    body = response.json()
    assert body["used_topics"] == ["array"]  # brute force → hash_map NOT credited
    assert body["confidence"] == 0.9
    assert body["processing_status"] == "completed"


def test_classify_degrades_to_zero_confidence(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        classify_mod,
        "get_llm_client",
        lambda: FakeLLMClient(raises=LLMUnavailableError("down")),
    )
    response = client.post("/classify-approach", json=_BRUTE_FORCE_TWO_SUM)
    assert response.status_code == 200
    assert response.json()["confidence"] == 0.0

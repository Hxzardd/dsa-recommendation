"""Orchestrator tests — analyze + classify pipelines with an injected fake LLM."""

import asyncio
import json

from app.llm.client import FakeLLMClient, LLMUnavailableError
from app.models.request_schemas import AnalyzeRequest, ClassifyApproachRequest
from app.orchestrator.analyze import analyze_submission
from app.orchestrator.classify import classify_approach
from tests.fixtures.sample_payloads import VALID_WRONG_ANSWER_PAYLOAD

_GOOD_ANALYZE_JSON = json.dumps(
    {
        "feedback_text": "Your search skips the final index.",
        "hint_text": "Re-examine your right bound.",
        "error_category": "off_by_one",
        "reasoning_quality": "partial",
        "concept_gaps": ["loop_bounds"],
    }
)


def _wrong_answer_request() -> AnalyzeRequest:
    return AnalyzeRequest(**VALID_WRONG_ANSWER_PAYLOAD)


def test_analyze_completed_with_llm() -> None:
    req = _wrong_answer_request()
    fake = FakeLLMClient(_GOOD_ANALYZE_JSON)
    result = asyncio.run(analyze_submission(req, llm=fake))
    assert result.processing_status == "completed"
    assert result.error_category == "off_by_one"
    assert result.submission_id == req.submission_id
    assert fake.calls, "LLM should have been called for a wrong_answer"


def test_analyze_accepted_is_rule_only_without_llm() -> None:
    payload = {
        **VALID_WRONG_ANSWER_PAYLOAD,
        "verdict": "accepted",
        "test_summary": {"total_test_cases": 30, "passed_test_cases": 30, "failed_test_cases": 0},
        "sample_failed_cases": [],
    }
    fake = FakeLLMClient("should not be called")
    result = asyncio.run(analyze_submission(AnalyzeRequest(**payload), llm=fake))
    assert result.processing_status == "rule_only"
    assert fake.calls == [], "accepted submissions must not call the LLM"


def test_analyze_degrades_when_llm_down() -> None:
    fake = FakeLLMClient(raises=LLMUnavailableError("vllm down"))
    result = asyncio.run(analyze_submission(_wrong_answer_request(), llm=fake))
    assert result.processing_status == "error"
    assert result.feedback_text  # still a valid, non-empty response
    assert result.model_used == "rule_engine"


def test_analyze_degrades_on_invalid_llm_json() -> None:
    fake = FakeLLMClient("this is not json")
    result = asyncio.run(analyze_submission(_wrong_answer_request(), llm=fake))
    assert result.processing_status == "llm_output_invalid"
    assert result.feedback_text


def _classify_request() -> ClassifyApproachRequest:
    return ClassifyApproachRequest(
        submission_id="sub_1",
        problem_id="two-sum",
        language="python",
        source_code=(
            "def two_sum(nums, t):\n"
            "    for i in range(len(nums)):\n"
            "        for j in range(i + 1, len(nums)):\n"
            "            if nums[i] + nums[j] == t:\n"
            "                return [i, j]\n"
        ),
        candidate_topics=["array", "hash_map"],
        candidate_patterns=["simulation"],
        data_structure_tags=["integer_arrays"],
    )


def test_classify_completed_with_llm() -> None:
    raw = json.dumps(
        {
            "matched_approach": "brute_force_nested",
            "used_topics": ["array"],
            "used_data_structures": ["integer_array"],
            "used_patterns": [],
            "confidence": 0.88,
        }
    )
    fake = FakeLLMClient(raw)
    result = asyncio.run(classify_approach(_classify_request(), llm=fake))
    assert result.processing_status == "completed"
    assert result.used_topics == ["array"]  # brute force → no hash_map credit
    assert result.confidence == 0.88


def test_classify_degrades_to_zero_confidence_when_llm_down() -> None:
    fake = FakeLLMClient(raises=LLMUnavailableError("down"))
    result = asyncio.run(classify_approach(_classify_request(), llm=fake))
    assert result.processing_status == "error"
    assert result.confidence == 0.0
    assert result.used_topics == []

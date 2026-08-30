"""Response validator tests (pure parsing/coercion)."""

import json

from app.validator.response_validator import parse_analyze_json, parse_classify_json

_GOOD_ANALYZE = {
    "feedback_text": "Your loop misses the last element.",
    "hint_text": "Check your upper bound.",
    "error_category": "off_by_one",
    "reasoning_quality": "partial",
    "concept_gaps": ["loop_bounds", "binary_search"],
}


def test_parse_analyze_valid() -> None:
    result = parse_analyze_json(json.dumps(_GOOD_ANALYZE), max_concept_gaps=8)
    assert result is not None
    assert result.error_category == "off_by_one"
    assert result.reasoning_quality == "partial"
    assert result.concept_gaps == ["loop_bounds", "binary_search"]


def test_parse_analyze_strips_code_fences() -> None:
    raw = "```json\n" + json.dumps(_GOOD_ANALYZE) + "\n```"
    assert parse_analyze_json(raw, max_concept_gaps=8) is not None


def test_parse_analyze_coerces_bad_enums_to_unknown() -> None:
    bad = {**_GOOD_ANALYZE, "error_category": "banana", "reasoning_quality": "nope"}
    result = parse_analyze_json(json.dumps(bad), max_concept_gaps=8)
    assert result is not None
    assert result.error_category == "unknown"
    assert result.reasoning_quality == "unknown"


def test_parse_analyze_clamps_concept_gaps() -> None:
    many = {**_GOOD_ANALYZE, "concept_gaps": [f"g{i}" for i in range(20)]}
    result = parse_analyze_json(json.dumps(many), max_concept_gaps=3)
    assert result is not None
    assert len(result.concept_gaps) == 3


def test_parse_analyze_unparseable_returns_none() -> None:
    assert parse_analyze_json("not json at all", max_concept_gaps=8) is None
    assert parse_analyze_json("", max_concept_gaps=8) is None


def test_parse_analyze_requires_some_text() -> None:
    empty = {**_GOOD_ANALYZE, "feedback_text": "", "hint_text": ""}
    assert parse_analyze_json(json.dumps(empty), max_concept_gaps=8) is None


def test_parse_classify_filters_to_candidates() -> None:
    raw = json.dumps(
        {
            "matched_approach": "brute_force_nested",
            "used_topics": ["array", "hash_map", "not_a_candidate"],
            "used_data_structures": ["integer_array"],
            "used_patterns": ["simulation", "ghost_pattern"],
            "confidence": 0.9,
        }
    )
    result = parse_classify_json(
        raw,
        candidate_topics=["array", "hash_map"],
        candidate_patterns=["simulation"],
        max_used=32,
    )
    assert result is not None
    assert result.used_topics == ["array", "hash_map"]  # not_a_candidate filtered
    assert result.used_patterns == ["simulation"]  # ghost_pattern filtered
    assert result.matched_approach == "brute_force_nested"
    assert result.confidence == 0.9


def test_parse_classify_clamps_confidence_and_defaults() -> None:
    result = parse_classify_json(
        json.dumps({"confidence": 5}),
        candidate_topics=["array"],
        candidate_patterns=[],
        max_used=32,
    )
    assert result is not None
    assert result.confidence == 1.0
    assert result.matched_approach == "other"
    assert result.used_topics == []


def test_parse_classify_unparseable_returns_none() -> None:
    assert (
        parse_classify_json("garbage", candidate_topics=[], candidate_patterns=[], max_used=32)
        is None
    )

"""Build the approach-classification prompt.

Given the submitted code and the problem's candidate topics/patterns (plus the
optimal solution signature and known wrong approaches), ask the model which
topics/techniques the code ACTUALLY used. Strict-JSON output, mentor-neutral.
"""

from __future__ import annotations

import json

from app.config.settings import get_settings
from app.models.domain import LLMPrompt
from app.models.request_schemas import ClassifyApproachRequest

CLASSIFY_SYSTEM_PROMPT = """You classify which topics and techniques a code \
submission ACTUALLY uses. You are given a problem's candidate topics/patterns \
and the submitted code. Decide, from the code alone, which candidates the code \
genuinely uses (e.g. a brute-force nested loop does NOT use a hash map even if \
hash_map is a candidate).
Respond in strict JSON only, matching this exact schema:
{
  "matched_approach": "short label: the optimal approach pattern, one of the \
common wrong approaches, or \\"other\\"",
  "used_topics": ["subset of candidate_topics actually used"],
  "used_data_structures": ["data structures actually used in the code"],
  "used_patterns": ["subset of candidate_patterns actually used"],
  "confidence": 0.0
}
confidence is your certainty in [0,1]. Only include candidates you are \
confident the code uses. No markdown fences, no prose before or after the JSON."""


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [source truncated for length] ..."


def build_classify_prompt(request: ClassifyApproachRequest) -> LLMPrompt:
    """Compose the classify prompt from a request, within the char budget."""

    settings = get_settings()
    signature = (
        json.dumps(request.solution_signature, ensure_ascii=False)
        if request.solution_signature
        else "none"
    )
    wrong = (
        json.dumps(
            [
                {"pattern": w.get("pattern"), "description": w.get("description")}
                for w in request.common_wrong_approaches
            ],
            ensure_ascii=False,
        )
        if request.common_wrong_approaches
        else "none"
    )

    # Reserve budget for the fixed sections, spend the rest on source code.
    fixed = "\n".join(
        [
            f"Language: {request.language}",
            f"candidate_topics: {json.dumps(request.candidate_topics)}",
            f"candidate_patterns: {json.dumps(request.candidate_patterns)}",
            f"data_structure_tags: {json.dumps(request.data_structure_tags)}",
            f"optimal_solution_signature: {signature}",
            f"common_wrong_approaches: {wrong}",
            "Submitted source code:",
        ],
    )
    source_budget = max(
        0, settings.prompt_max_chars - len(CLASSIFY_SYSTEM_PROMPT) - len(fixed) - 64
    )
    user = (
        fixed
        + "\n"
        + _truncate(request.source_code, source_budget)
        + "\n\nRespond with the JSON object only."
    )
    return LLMPrompt(system=CLASSIFY_SYSTEM_PROMPT, user=user)

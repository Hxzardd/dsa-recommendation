# API reference — KNode AI Code Analysis

Base URL (local): `http://localhost:8099` — interactive docs at `/docs`.

## Auth

Optional shared-secret bearer (`app/api/deps.py::require_service_auth`).
Enforced only when `SERVICE_API_KEY` is configured; otherwise the service is
open (local/dev). When set, callers must send
`Authorization: Bearer <SERVICE_API_KEY>` (must equal the backend's
`AI_SERVICE_API_KEY`). `/health` and `/live` are always open.

---

## `POST /analyze` — mentor feedback for a submission

Consumed by the backend's `integrations/ai-feedback` (`AI_ANALYSIS_URL`),
typically for failed submissions.

**Request** (`AnalyzeRequest`):

```jsonc
{
  "submission_id": "sub_5ac8",
  "problem_id": "binary-search",
  "user_id": "user_558",
  "language": "python",
  "verdict": "wrong_answer",      // wrong_answer | time_limit_exceeded |
                                  // memory_limit_exceeded | runtime_error |
                                  // compilation_error | accepted
  "source_code": "def f(): ...",  // required, non-empty, <= MAX_SOURCE_CODE_CHARS
  "test_summary": { "total_test_cases": 30, "passed_test_cases": 24, "failed_test_cases": 6 },
  "sample_failed_cases": [        // <= 3
    { "stdin": "...", "expected_output": "0", "actual_output": "-1" }
  ],
  "stdout": "", "stderr": "", "compile_output": "",
  "execution_time_ms": 39, "memory_kb": 14208,
  "submitted_at": "2026-07-02T10:15:43Z"
}
```

**Response** (`AnalyzeResponse`):

```jsonc
{
  "submission_id": "sub_5ac8",
  "feedback_text": "Your loop misses the last index…",   // never a full solution
  "hint_text": "Re-check your upper bound.",
  "error_category": "off_by_one",   // wrong_answer_logic | off_by_one |
                                    // edge_case_missing | wrong_algorithm |
                                    // time_limit_exceeded | memory_limit_exceeded |
                                    // runtime_error | compilation_error | unknown
  "reasoning_quality": "partial",   // strong | partial | weak | unknown
  "concept_gaps": ["loop_bounds"],
  "processing_status": "completed", // completed | rule_only | llm_output_invalid | timeout | error
  "processing_ms": 812,
  "model_used": "Qwen/Qwen2.5-Coder-32B-Instruct"   // or "rule_engine" on degrade
}
```

- `422` — schema violation (empty/oversized `source_code`, bad `verdict`, …).
- `400` — unsafe input (`{ "error_code": "unsafe_input", … }`).
- Never `5xx` on an LLM outage — degrades to a valid response (see below).

---

## `POST /classify-approach` — which topics/techniques were used

Consumed by the backend's `integrations/approach-detection` (`AI_CLASSIFY_URL`)
to gate approach-aware mastery credit.

**Request** (`ClassifyApproachRequest`):

```jsonc
{
  "submission_id": "sub_1",
  "problem_id": "two-sum",
  "language": "python",
  "source_code": "def two_sum(...): ...",   // required, non-empty
  "problem_statement": "…",                  // optional context
  "candidate_topics": ["array", "hash-map"], // topic slugs (what may be returned)
  "candidate_patterns": ["simulation"],
  "data_structure_tags": ["integer_arrays", "hash_maps"],
  "solution_signature": { "approach_family": "hashing", "code_motifs": [...] },
  "common_wrong_approaches": [ { "pattern": "brute_force_nested", "description": "…" } ]
}
```

**Response** (`ClassifyApproachResponse`):

```jsonc
{
  "submission_id": "sub_1",
  "matched_approach": "brute_force_nested",  // optimal pattern | a wrong-approach | "other"
  "used_topics": ["array"],                  // subset of candidate_topics actually used
  "used_data_structures": ["integer_array"],
  "used_patterns": [],                       // subset of candidate_patterns
  "confidence": 0.9,                         // 0..1
  "processing_status": "completed",
  "processing_ms": 640,
  "model_used": "Qwen/Qwen2.5-Coder-32B-Instruct"
}
```

The backend reads `matched_approach` / `used_*` / `confidence`. On any degrade
(LLM down or unusable output) the service returns `confidence: 0` with empty
`used_*`, and the backend keeps structural-weights-only crediting.

`used_topics` / `used_patterns` are always filtered to the supplied candidate
sets, so the backend can map them straight back to `problem_topic` rows.

---

## `GET /health` — readiness
Returns `{"status": "ok"}` (the service has no hard external dependency; vLLM is
called lazily per request and degraded when absent).

## Degradation reference

| `processing_status` | Meaning |
|---|---|
| `completed` | LLM produced valid output |
| `rule_only` | deterministic rule answered; no LLM call (analyze only) |
| `error` | LLM unreachable/timeout → deterministic fallback / confidence 0 |
| `llm_output_invalid` | LLM replied but JSON was unusable → fallback / confidence 0 |
| `timeout` | reserved for explicit timeout classification |

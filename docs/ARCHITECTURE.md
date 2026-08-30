# Architecture — KNode AI Code Analysis

A small, stateless **FastAPI** service that reviews a learner's code submission.
Two jobs: mentor-style **feedback** (`/analyze`) and **approach classification**
(`/classify-approach`). LLM inference is **vLLM-only**; the service degrades
gracefully when vLLM is unavailable and never returns 5xx on an LLM outage.

## Where it sits (three services)

| Service | Repo / branch | Role |
|---|---|---|
| Backend | `dsa-website` (`personal`) | Next.js app + API; owns all persistence; calls this service |
| ML engine | `dsa-recommendation` (`personalML`) | score + BKT + HLR + recommendations |
| **AI analysis** | `dsa-recommendation` (`AI`) — *this* | vLLM `/analyze` + `/classify-approach` |

The backend calls `/classify-approach` to learn which topics/techniques a
submission actually used (feeding approach-aware mastery gating in the ML
service), and `/analyze` to show mentor feedback on failed submissions.

## Request pipeline

```
request → sanitizer → parser → rule_engine ─┬─ short-circuit (deterministic) ─────► response
                                            │
                                            └─ needs LLM → prompt_builder → LLM → validator → response
                                                                            │
                                                              injected LLMClient (vLLM / fake)
```

- **sanitizer** (`app/security/sanitizer.py`) — rejects oversized/unsafe input (→ 400).
- **parser** (`app/parser/normalizer.py`) — `AnalyzeRequest` → `NormalizedSubmission`
  (trims output, builds a deterministic diff summary).
- **rule_engine** (`app/rule_engine/`) — deterministic rules for clear cases
  (accepted, compilation errors). A high-confidence rule short-circuits without
  any LLM call (`processing_status: rule_only`).
- **prompt_builder** (`app/prompt_builder/`) — strict-JSON, mentor-style prompts
  that never leak a full solution; budget-aware truncation. A separate
  `classify_builder.py` builds the approach-classification prompt.
- **LLM client** (`app/llm/client.py`) — see the seam below.
- **validator** (`app/validator/response_validator.py`) — pure parse/validate of
  the model's JSON; strips code fences, coerces bad enums to `unknown`, clamps
  lists, filters classify results to the candidate sets. Returns `None` when
  unparseable so the orchestrator can degrade.
- **orchestrator** (`app/orchestrator/`) — the deep module gluing it all
  together; see below.

## Deep modules & seams

- **Orchestrator = the deep module.** Interface: `analyze_submission(request,
  llm=None)` and `classify_approach(request, llm=None)`. Everything above is
  hidden behind these two functions — routes and tests cross only this seam.
- **LLM client = an injected internal seam.** Interface `complete(prompt) -> str`.
  Two adapters: `VLLMClient` (real, httpx → OpenAI-compatible
  `/v1/chat/completions`) and `FakeLLMClient` (in-memory, for tests). The
  orchestrator accepts an `LLMClient` and never constructs one itself, so tests
  inject the fake and production injects vLLM. `get_llm_client()` returns the
  production client.
- **Validator = a pure function** (no I/O, no side effects), so its parsing/
  coercion rules are trivially testable.

## Graceful degradation

The service is built so all three services can run together without errors even
with no vLLM. Every non-input failure maps to a valid response via
`processing_status`:

| Situation | `/analyze` | `/classify-approach` |
|---|---|---|
| Deterministic rule matched | `rule_only` (no LLM call) | n/a |
| LLM success + valid JSON | `completed` | `completed` |
| LLM unreachable / timeout | `error` (rule-engine fallback text) | `error`, `confidence: 0` |
| LLM returned unusable JSON | `llm_output_invalid` (fallback text) | `llm_output_invalid`, `confidence: 0` |

On the classify path, `confidence: 0` tells the backend to keep
structural-weights-only crediting (no approach gating). `/analyze` never leaks a
solution even in fallback text.

## Configuration

vLLM-only (`app/config/settings.py`): `VLLM_BASE_URL`, `VLLM_MODEL`,
`VLLM_API_KEY` (optional), `LLM_TIMEOUT_SECONDS`, plus `SERVICE_API_KEY`
(optional shared bearer for backend→AI), `RULE_ENGINE_ENABLED`,
`MAX_CONCEPT_GAPS`, `MAX_USED_ITEMS`, prompt/source char budgets. See
[.env.example](../.env.example). For choosing and running an LLM backend
(Ollama for dev, vLLM in WSL2, hosted vLLM for prod, VRAM sizing) see
[LLM_SETUP.md](LLM_SETUP.md).

## Module map

```
app/
  main.py            FastAPI app: middleware, exception handlers, routers, /health
  api/               routes_analyze · routes_classify · deps (optional bearer auth)
  orchestrator/      analyze · classify  (the deep module)
  llm/               client: LLMClient protocol · VLLMClient · FakeLLMClient
  rule_engine/       engine · rules  (deterministic short-circuits)
  prompt_builder/    builder (analyze) · classify_builder
  parser/            normalizer
  validator/         response_validator (pure)
  security/          sanitizer
  models/            request_schemas · response_schemas · domain
  middleware/        request logging · error handlers
  config/            settings
  logging/           structured JSON logger
tests/               unit + integration (fake LLM; run offline)
```

## Invariants

- The LLM is never trusted: all output passes the validator; bad output degrades.
- Feedback is conceptual — the system prompt forbids returning a full solution.
- No persistence: the service is stateless; the backend stores everything.

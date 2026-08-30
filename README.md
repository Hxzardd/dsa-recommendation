<div align="center">

![ACM VIT](https://raw.githubusercontent.com/ACM-VIT/.github/master/profile/acm_gif_banner.gif)

<h2>KNode AI Code Analysis</h2>

<p>The AI microservice behind the Knode code editor. It reviews a learner's
submission and returns mentor-style feedback + a hint, and it classifies which
topics/techniques a submission actually used so mastery is credited fairly.</p>

<p>
  <a href="https://acmvit.in/" target="_blank">
    <img alt="made-by-acm" src="https://img.shields.io/badge/MADE%20BY-ACM%20VIT-orange?style=flat-square&logo=acm&link=acmvit.in" />
  </a>
  <img alt="python" src="https://img.shields.io/badge/python-3.12%2B-blue?style=flat-square&logo=python" />
  <img alt="fastapi" src="https://img.shields.io/badge/FastAPI-vLLM-009688?style=flat-square&logo=fastapi" />
</p>

</div>

---

## What it does

Two endpoints, both consumed by the [`dsa-website`](../dsa-website) backend:

- **`POST /analyze`** — mentor-style feedback for a (usually failed) submission:
  `feedback_text`, `hint_text`, `error_category`, `reasoning_quality`,
  `concept_gaps`. Never returns a full solution.
- **`POST /classify-approach`** — decides which of a problem's candidate
  topics/patterns the code *actually used* (e.g. a brute-force Two Sum does not
  use a hash map), so the backend can gate mastery credit on the real approach.
- **`GET /health`** / **`GET /live`** — probes.

## Architecture

A small pipeline, glued by a **deep orchestrator** (the only seam routes/tests
cross):

```
request → sanitizer → parser → rule_engine → prompt_builder → LLM → validator → response
                                     │                          │
                        deterministic short-circuit      injected LLMClient
                        (accepted, clear errors)         (vLLM adapter / fake)
```

- **LLM = vLLM only** (OpenAI-compatible `/v1/chat/completions`). Code analysis
  is demanding, so we target a capable served code model, not a small local one.
- **Safe degradation:** if vLLM is unreachable, slow, or returns bad JSON, the
  service still returns a valid response — `/analyze` falls back to the rule
  engine (`processing_status` = `rule_only`/`error`/`llm_output_invalid`),
  `/classify-approach` returns `confidence: 0` (backend then stays
  weights-only). It never 5xxes on an LLM outage.

Modules: `app/orchestrator` (analyze, classify) · `app/llm` (vLLM client + fake)
· `app/rule_engine` · `app/prompt_builder` · `app/parser` · `app/validator` ·
`app/security` · `app/models` · `app/api` · `app/middleware` · `app/config`.

## Documentation

| Doc | What's in it |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Pipeline, deep orchestrator, LLM seam, degradation, module map |
| [docs/API.md](docs/API.md) | `/analyze` + `/classify-approach` schemas, auth, status codes |
| [.env.example](.env.example) | All configuration (vLLM, auth, budgets) |

## Quick start

Requires **Python 3.12+**. With [uv](https://docs.astral.sh/uv/):

```bash
uv venv && uv pip install -e ".[dev]"
cp .env.example .env                       # set VLLM_BASE_URL / VLLM_MODEL
uv run uvicorn app.main:app --port 8099    # Swagger at /docs
uv run pytest -q                           # tests run offline (no vLLM needed)
```

The full test suite uses an in-memory fake LLM, so it passes with no vLLM
running. Point `VLLM_BASE_URL` at a live server to get real analysis output.

## Configuration

See [.env.example](.env.example). Key vars: `VLLM_BASE_URL`, `VLLM_MODEL`,
`VLLM_API_KEY` (optional), `SERVICE_API_KEY` (optional shared bearer for
backend→AI, must match the backend's `AI_SERVICE_API_KEY`), `LLM_TIMEOUT_SECONDS`.

## Backend integration

The backend calls this service from `src/integrations/ai-feedback` (`/analyze`,
→ `AI_ANALYSIS_URL`) and `src/integrations/approach-detection`
(`/classify-approach`, → `AI_CLASSIFY_URL`). Request/response schemas here are
the source of truth for those integrations.

---

<div align="center">

🤍 Crafted with love by <a href="https://acmvit.in/" target="_blank">ACM‑VIT</a>

</div>

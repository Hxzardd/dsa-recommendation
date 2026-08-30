<div align="center">

![ACM VIT](https://raw.githubusercontent.com/ACM-VIT/.github/master/profile/acm_gif_banner.gif)

<h2>DSA Recommendation Engine</h2>

<p>The machine-learning service behind the Knode DSA platform — it scores each
submission, tracks topic mastery, schedules spaced-repetition reviews, and
recommends what a learner should solve next.</p>

<p>
  <a href="https://acmvit.in/" target="_blank">
    <img alt="made-by-acm" src="https://img.shields.io/badge/MADE%20BY-ACM%20VIT-orange?style=flat-square&logo=acm&link=acmvit.in" />
  </a>
  <img alt="python" src="https://img.shields.io/badge/python-3.12-blue?style=flat-square&logo=python" />
  <img alt="fastapi" src="https://img.shields.io/badge/FastAPI-stateless-009688?style=flat-square&logo=fastapi" />
</p>

</div>

---

## What it does

This is a **FastAPI** service that pairs with the [`dsa-website`](../dsa-website)
backend (they share one Postgres database). It has two jobs:

1. **Score a submission** (`POST /update`) — given a learner's submission and
   behavioural telemetry, it formulates the submission score and computes the
   updated **BKT** topic mastery and **HLR** review schedule, and returns them.
   It is a **stateless calculator**: the backend owns all persistence of user
   mastery.
2. **Recommend what's next** (`GET /recommend`, `/topic/recommend`, …) — a
   candidate-generation → filtering → LightGBM-ranking pipeline over problem
   embeddings (Qdrant) and a per-user concept graph (Neo4j/Redis).

```
submit → backend → AI /classify-approach (which topics were used)
                 → ML  /update (telemetry + current mastery + per-topic weights)
                 → score + BKT + HLR  ←  this service
                 → backend persists mastery / XP / reviews
```

### Part of a three-service system

| Service | Repo / branch | Role |
|---|---|---|
| Backend | `dsa-website` (`personal`) | Next.js app + API; owns all persistence |
| **ML engine** | `dsa-recommendation` (`personalML`) — *this* | score + BKT + HLR + recommendations |
| AI analysis | `dsa-recommendation` (`AI` branch) | vLLM `/analyze` + `/classify-approach` |

They share one Postgres and degrade independently. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full picture.

## Documentation

| Doc | What's in it |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, the two responsibilities, data stores, invariants |
| [docs/API.md](docs/API.md) | Every endpoint, request/response schemas, auth model |
| [docs/INTEGRATION.md](docs/INTEGRATION.md) | The backend ↔ ML contract, env vars, XP policy, retry path |
| [RUNNING.md](RUNNING.md) | Install, run, and test end-to-end |
| [scripts/README.md](scripts/README.md) | Operational, data-generation and training scripts |

## Quick start

Requires **Python 3.12** and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                  # install deps
cp .env.example .env                     # then fill in DATABASE_URL etc.
uvicorn main:app --reload --port 8000    # run — Swagger at /docs
uv run pytest tests/ -q                  # test (runs offline)
```

See [RUNNING.md](RUNNING.md) for Postgres setup, Qdrant/Neo4j config, and the
end-to-end integration test. The core BKT/HLR/scoring/telemetry logic runs
without any external service.

## Project layout

```
main.py                     FastAPI app + startup graph load
routes/                     thin HTTP routers  (submission, mastery, recommendation, seeding, health)
controllers/                request handlers   (orchestrate the pipeline)
models/schemas/             pydantic request/response models
middlewares/auth.py         session-token + ML_SERVICE_TOKEN auth
pipeline/
  recommender/              scoring.py · bkt.py · hlr.py · telemetry.py · ranking.py · pools/ · services/
  ingestion/ · graphs/ · embeddings/
database/                   postgres / qdrant / neo4j clients
training/                   LightGBM dataset generation + training
scripts/                    one-off ops / data / training scripts (run from repo root)
tests/                      offline unit + endpoint smoke tests
docs/                       architecture, API, integration
```

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md). Please keep PRs focused, add/adjust
tests under `tests/`, and run `uv run pytest tests/ -q` before opening one.
By participating you agree to our [Code of Conduct](CODE_OF_CONDUCT.md).

---

<div align="center">

🤍 Crafted with love by <a href="https://acmvit.in/" target="_blank">ACM‑VIT</a>

</div>

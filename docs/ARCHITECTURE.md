# Architecture

The DSA recommendation service is a **stateless FastAPI calculator + a
recommendation pipeline**. It computes learning signals (submission score,
topic mastery, review scheduling) and recommends what a learner should do next.
It shares a Postgres database with the backend ([`dsa-website`](../README.md)),
but the **backend owns all persistence of user mastery**; this service returns
numbers, it does not write the backend's mastery tables.

```
                         ┌──────────────────────────────────────────┐
   learner submits code  │                dsa-website               │
        (SUBMIT)  ─────►  │  POST /api/submissions → Judge0 webhook  │
                         │        → onSubmissionComplete             │
                         └───────────────┬──────────────────────────┘
                                         │  POST /update  (Bearer: ML_SERVICE_TOKEN)
                                         │  { verdict, telemetry, problemTopics[current mastery+HLR], … }
                                         ▼
                         ┌──────────────────────────────────────────┐
                         │           dsa-recommendation (this)       │
                         │                                           │
                         │  scoring.py   → submission score          │
                         │  bkt.py       → updated topic mastery P(L) │
                         │  hlr.py       → updated HLR / next review  │
                         │                                           │
                         │  returns { score, updatedTopics[mastery,  │
                         │            hlr], masteredTopics, results } │
                         │                                           │
                         │  (also projects results into its OWN      │
                         │   recommendation graph: Redis + Neo4j)    │
                         └───────────────┬──────────────────────────┘
                                         │  response
                                         ▼
                         backend persists user_topic_mastery,
                         user_hlr_state, submission.finalScore, XP …
```

## Two responsibilities, one service

### 1. The `/update` write path (stateless calculator)
On each accepted submission the backend calls `POST /update` with the full
behavioural telemetry and the user's *current* per-topic mastery/HLR state
(the backend reads that from Postgres and sends it in the body). This service:

- **`pipeline/recommender/scoring.py`** — formulates the 0..1 submission
  `score` from telemetry (first-pass, approach, fluency, edge-case, speed,
  hint penalty, runtime/memory optimality). Weight-parity with the backend's
  `src/server/scoring.ts`.
- **`pipeline/recommender/bkt.py`** — Bayesian Knowledge Tracing. Updates each
  topic's mastery `P(L)` with a bounded, diminishing-returns delta
  (`MAX_MASTERY_DELTA`, mastery-proximity dampening, trivial-gap dampening,
  repeat-solve dampening, per-topic weight). A single submission can never
  jump mastery arbitrarily.
- **`pipeline/recommender/hlr.py`** — Half-Life Regression. Updates each
  topic's forgetting half-life and next-review date (spaced repetition).
- **`pipeline/recommender/telemetry.py`** — the *shared* performance signal
  consumed by both BKT and HLR (so they never disagree about "how well did
  the learner do"), with a confidence penalty for thrashing.

It returns the new values and the backend persists them. It **does not** write
`user_topic_mastery` / `user_hlr_state` — that keeps a single writer for those
tables (the backend) and matches the documented contract.

### 2. The recommendation read paths (pipeline)
`GET /recommend`, `GET /topic/recommend`, `POST /topic/recommend/problems`,
`GET /mastery`, `GET /urgency` serve "what should I do next". The pipeline
lives under `pipeline/`:

- **`pipeline/recommender/pools/`** — candidate pools (review-due, weak-topic,
  novelty/exploration, progression).
- **`pipeline/recommender/services/candidate_filtering.py`** — prerequisite
  gating and de-duplication.
- **`pipeline/recommender/ranking.py`** + **`training/`** — a LightGBM ranker
  (falls back to a heuristic ranker when no model is present).
- **`pipeline/recommender/services/user_graph_service.py`** +
  **`state_update_service.py`** — the per-user concept graph, cached in Redis
  and persisted to Neo4j. `/update` projects its computed BKT/HLR results into
  this graph (ML-internal state, distinct from the backend's mastery tables).

## Data stores

| Store | Role | Required for |
|---|---|---|
| **Postgres** (shared with backend) | `session` (auth), reads current mastery via request body; seeding writes | everything |
| **Qdrant** | problem embeddings for candidate generation | `GET /recommend` |
| **Neo4j** | durable per-user concept graph + offline concept graph | full pipeline (degrades gracefully if down) |
| **Redis** | hot cache for the user graph | performance (optional) |

## Auth
See [API.md](API.md#auth). Two modes: a per-user Better-Auth **session token**
(validated against the shared `session` table) for end-user calls, and a shared
**`ML_SERVICE_TOKEN`** for server-to-server calls like the submission webhook.

## Design invariants
- `/update` never recalculates BKT/HLR twice, never persists the backend's
  mastery tables, and propagates errors so the caller can retry.
- Mastery threshold is a single canonical constant (`MASTERY_THRESHOLD = 0.75`).
- All telemetry is neutral-safe: a missing signal yields a neutral component,
  never a penalty.

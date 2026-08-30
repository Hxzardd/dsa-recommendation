# Architecture

The DSA recommendation service is a **stateless FastAPI calculator + a
recommendation pipeline**. It computes learning signals (submission score,
topic mastery, review scheduling) and recommends what a learner should do next.
It shares a Postgres database with the backend ([`dsa-website`](../README.md)),
but the **backend owns all persistence of user mastery**; this service returns
numbers, it does not write the backend's mastery tables.

## System context — three services

Knode is three cooperating services sharing one Postgres:

| Service | Repo / branch | Role |
|---|---|---|
| **Backend** | `dsa-website` (`personal`) | Next.js app + API. Owns all persistence (mastery, XP, submissions) and orchestrates the submission hook. |
| **ML engine** | `dsa-recommendation` (`personalML`) — *this* | Stateless score + BKT + HLR calculator, plus the recommendation pipeline. |
| **AI analysis** | `dsa-recommendation` (`AI` branch) — "KNode AI Code Analysis" | vLLM-backed `POST /analyze` (mentor feedback on failed submissions) and `POST /classify-approach` (which topics/techniques a submission actually used). |

On a submission the backend first asks the **AI service** to classify the
approach, then calls this ML service's `/update` with per-topic `weight`s derived
from that classification — so mastery credit reflects the technique the learner
actually used (a brute-force Two Sum does not credit `hash_map`). All three
degrade independently: if the AI service or vLLM is down, crediting falls back to
structural weights; if this ML service is down, the backend defers mastery for
retry. See [INTEGRATION.md](INTEGRATION.md).

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

### Problem-topic graph artifacts (weighted)
`scripts/regenerate_graph_artifacts.py` builds the offline graph from the
curated `data/source-target.txt` (72-canonical-tag problem→topic mapping) + the
manifest, emitting `data/*_normalized.json` and `question-graph/data/*.json`.
Each problem-topic edge carries a **`role`** (`domain` = the substrate any
solution touches, e.g. `array`; `optional` = a technique/auxiliary structure a
solution chooses, e.g. `hash_map`) and a structural **`weight`**. This is the
same taxonomy the backend uses to gate mastery credit (so a brute-force Two Sum
doesn't credit `hash_map`) — kept in sync with the backend's
`scripts/seed-problemtopic-weights.ts` `DOMAIN_TAGS`. `topic_topic_edges` carry
co-occurrence (`jaccard`). Regenerate with:
```
python scripts/regenerate_graph_artifacts.py
```
`question-graph/src/graph_builder.py` assembles these into the GraphML the RGCN
pipeline consumes; the edge `role`/`weight`/`is_primary_topic` flow through.

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

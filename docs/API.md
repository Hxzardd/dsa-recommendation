# API reference

Base URL (local): `http://localhost:8000` — interactive docs at `/docs`.

## Auth

Every route except the health/docs paths requires
`Authorization: Bearer <token>`. There are two accepted token types
(`middlewares/auth.py`):

| Token | How it's validated | Used by |
|---|---|---|
| **Session token** | looked up in the shared Postgres `session` table (non-expired) → resolves a `user_id` | end-user calls from the backend UI |
| **`ML_SERVICE_TOKEN`** | constant-time compared to the env var of the same name; marks the call `is_service_call` (no `user_id`) | server-to-server calls (the Judge0 submission webhook's `POST /update`) |

Ownership: routes with a `{user_id}` (or `userId` body) reject a session token
whose `user_id` differs (`403`). A **service-token** call bypasses the
ownership check — it acts on whatever `userId` the body names, on the backend's
authority.

Public (no token): `/`, `/health`, `/live`, `/docs`, `/openapi.json`, `/redoc`.

---

## `POST /update` — score + BKT + HLR (stateless)

The submission write path. Computes the score, updated mastery and updated HLR
and **returns** them; it does not persist the backend's mastery tables.

**Request body** (`models/schemas/submission.py`):

```jsonc
{
  "userId": "user_abc123",
  "problemId": "two-sum",          // LeetCode title slug
  "verdict": "OK",                  // exactly "OK" for a full-credit solve
  "hintsUsed": 0,
  "testCasesPassed": 10,
  "totalTestCases": 10,
  "submissionCount": 1,
  "normalisedScore": 0.95,          // 0..1 (NOT 0..100) — 422 otherwise
  "problemDifficulty": 0.35,        // optional 0..1, enables difficulty-aware BKT
  "timestamp": 1752345600.0,        // unix seconds
  "problemTopics": [
    { "topicId": "array", "currentMastery": 0.45,
      "currentHlr": { "half_life": 5.0, "last_review": "2026-07-01T00:00:00+00:00",
                      "p_recall": 0.8, "next_review_days": 3.5 },
      "weight": 0.7 },
    { "topicId": "hash_map", "currentMastery": null, "currentHlr": null, "weight": 0.3 }
  ],
  "telemetry": {                    // optional; ML formulates the score from it
    "majorRewriteCount": 0, "backspaceCount": 12, "totalKeystrokes": 400,
    "sessionDurationSeconds": 320, "hintsUsed": 0, "firstHintOpenedAtSeconds": null,
    "edgeCasesPassed": 10, "totalEdgeCases": 10,
    "runtimePercentile": 0.2, "memoryPercentile": 0.3, "recentSessionScores": []
  }
}
```

**Response**:

```jsonc
{
  "userId": "user_abc123",
  "problemId": "two-sum",
  "score": 0.61,                    // ML-owned 0..1 submission score
  "normalisedScore": 61,            // 0..100
  "updatedTopics": [
    { "topicId": "array", "updatedMastery": 0.4832,
      "updatedHlr": { "half_life": 6.1, "last_review": "…", "p_recall": 0.9,
                      "next_review_days": 4.2 } }
  ],
  "masteredTopics": [],             // topics that crossed MASTERY_THRESHOLD this call
  "results": { "bkt": [...], "hlr": [...], "score": { ...breakdown... } }
}
```

`topicId` is opaque to ML — whatever the backend sends (its own `topic.id`
CUID) round-trips untouched. `422` on out-of-range fields (e.g.
`normalisedScore` outside `0..1`) or missing required fields.

---

## `GET /mastery/{user_id}` — current mastery + decayed proficiency
Returns the user's per-topic mastery and a time-decayed proficiency view.

## `GET /urgency/{user_id}` — review urgency
Returns per-topic urgency scores (how overdue each topic's review is).

## `GET /recommend/{user_id}?limit=10` — problem recommendations
Runs the full pipeline (pools → filtering → ranking). `limit` clamped 1..50.
Needs Qdrant. May cold-start slowly on first call.

## `GET /topic/recommend/{user_id}` — next topic
Returns a single recommended topic + reason.

## `POST /topic/recommend/problems` — problems within a topic
Body: `{ userId, topicId, currentProblemId, masteryScore, candidates[], limit }`.
Ranks the supplied candidate problems for that topic.

## `POST /seed_bkt/{user_id}` · `POST /seed_hlr/{user_id}` — history import
One-time seeding of initial mastery/HLR from Codeforces/LeetCode history.
`ON CONFLICT DO NOTHING` for mastery — never clobbers real in-platform progress.

## `GET /health` · `GET /` — readiness
Probes Postgres/Qdrant (critical) and Neo4j (optional). `503` when a critical
dependency is down; `200` "degraded" if only Neo4j is down.

## `GET /live` — liveness
Dependency-free process-up ping; always `200`.

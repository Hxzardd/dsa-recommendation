# Backend ↔ ML integration

How [`dsa-website`](https://github.com/ACM-VIT) (the backend) and this service
talk on a submission. Read alongside the backend's `TESTING_ML_UPDATE.md`.

## The contract in one paragraph

A learner **submits** (not "runs") code. The backend persists the editor
telemetry onto the submission row, Judge0 judges it, and the webhook runs
`onSubmissionComplete`. For an accepted solve the backend calls
`POST /update` with the full telemetry **and** the user's current per-topic
mastery/HLR. This service formulates the **score** and computes the updated
**BKT mastery** and **HLR** and returns them. The backend persists everything:
`submission.finalScore` / `normalisedScore`, `user_topic_mastery`,
`user_hlr_state`, XP, streak, review schedule. **ML owns the formulation; the
backend owns persistence.** If ML is unreachable, the backend keeps its own
`calculateScore` result as a fallback and marks the submission
`topic_mastery_pending` for a later retry — mastery is never invented locally.

## Wiring

| Concern | Backend side | ML side |
|---|---|---|
| Endpoint | `src/integrations/ml/update-client.ts` → `POST {ML_SERVICE_URL}/update` | `routes/submission.py` → `controllers/submission_controller.py::handle_update` |
| Auth | forwards `ML_SERVICE_TOKEN` (Bearer); falls back to the user's session token | `middlewares/auth.py` validates either |
| Apply/persist | `src/integrations/ml/apply-topic-mastery.ts` writes `user_topic_mastery` / `user_hlr_state` | returns values only (stateless) |
| Score | `src/server/on-submission-complete.ts` uses ML's `score`; local `calculateScore` is the fallback | `pipeline/recommender/scoring.py` |
| Retry | `retry-pending-mastery.ts` + `POST /api/internal/ml/retry-mastery` | idempotent — same `/update` |

## Required configuration

Both repos must share the **same** `DATABASE_URL` (so the ML service can
validate the session token the backend forwards, and so seeding writes land in
the same tables).

Set the **same** secret in both `.env` files:

```
# dsa-website/.env  and  dsa-recommendation/.env
ML_SERVICE_TOKEN=<same value in both>
```

Backend also sets `ML_SERVICE_URL` (default `http://localhost:8000`) and, for
the retry route, `CRON_SECRET`.

> Why a shared service token? The Judge0 webhook runs server-to-server with no
> end-user session in context. Forwarding a user's session token couples
> webhook reliability to whether that user happens to have a live session. The
> shared token decouples them; the per-user session token remains the fallback.

## Request/response shape

See [API.md](API.md#post-update--score--bkt--hlr-stateless) for the full
`/update` body and response. Key rules:

- `normalisedScore` in the request is **0..1** (0..100 → `422`).
- `verdict` must be exactly `"OK"` for a full-credit solve; anything else is
  down-weighted by BKT, not rejected.
- `topicId` is opaque and round-trips (the backend sends its own `topic.id`).
- `telemetry` is optional and neutral-safe; the retry runner omits it.
- each `problemTopics[]` carries a **`weight`** (0..1) = structural relevance ×
  approach-usage. The backend computes it from `problem_topic.weight`/`role`
  and an LLM approach classification, so a topic the submission didn't actually
  use (e.g. `hash_map` on a brute-force Two Sum) arrives with weight 0 and BKT
  doesn't move it. `update_bkt` already scales its delta by this weight.

## XP policy (backend-owned)

- **First accepted solve** → full XP.
- **Due-review re-solve** (past the mastery gate) → full XP ×
  `0.5 ^ priorSolves`, floored at 5%. Streak and daily-goal count stay
  first-solve-only, so reviews are rewarded but can't be farmed.
- A re-solve **before** the review is due / within the cooldown credits no
  mastery and no XP (spam gate in `onSubmissionComplete`).

## End-to-end test

See [../RUNNING.md](../RUNNING.md) and the backend's `TESTING_ML_UPDATE.md`
for the full harness walkthrough (`scripts/test-ml-update.ts`).

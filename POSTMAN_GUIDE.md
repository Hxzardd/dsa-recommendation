# Postman Guide

Request-by-request walkthrough of `postman_collection.json`, with expected
results, for testing this service's endpoints (mastery updates in
particular) without writing any code. See [RUNNING.md](RUNNING.md) for how
to get the server and Postgres running first.

## 0. Setup

1. `psql "$DATABASE_URL" -f database/postgres/dev_schema.sql` -- creates
   `user`/`user_topic_mastery`/`user_hlr_state`/`session` if they don't
   already exist (all `IF NOT EXISTS`, safe against a DB that already has
   them).
2. `python seed_test_session.py postman_demo_user` -- creates the test user
   row AND a real session token (add `--cf`/`--lc` if you also want to
   exercise the seeding routes; `--days N` for a shorter/longer-lived
   token, default 30). **Copy the printed token** -- you need it next.
3. In Postman: File > Import > select both `postman_collection.json` and
   `postman_environment.json`.
4. Select the "DSA ML Service - Local" environment (top-right dropdown),
   then set its `auth_token` variable to the token from step 2.
5. Start the server: `uvicorn main:app --reload --port 8000`.

**If you're getting 403 Forbidden on POST /update:** this is always a
token/user mismatch, never a code bug -- `/update` 403s whenever the
Bearer token's resolved user doesn't match `submission.userId` in the
request body (by design, see auth note below). The single most common
cause: you re-ran `seed_test_session.py` (or someone else on the team
did) and it printed a **new** token, but the `auth_token` variable in your
Postman environment still has the **old** one. Fix:
```
python seed_test_session.py postman_demo_user
```
Copy the freshly printed token, paste it into the "DSA ML Service - Local"
environment's `auth_token` value (Postman: environment quick-look eye icon
> edit), and re-send. Also double check you selected that environment
(top-right dropdown) -- "No Environment" selected silently sends no auth
header at all, which 401s, not 403 (a 403 specifically means the token
DID resolve, just to the wrong user). If you're running against a
different Postgres than the one this token was seeded against (e.g. a
teammate's DB, or you changed `DATABASE_URL`), the token won't resolve at
all either -- reseed against whichever DB your server is actually using.

**Auth is now required on every route** except `/`, `/docs`,
`/openapi.json`, `/redoc` -- `middlewares/auth.py` checks for a valid
`Authorization: Bearer <token>` resolving to a non-expired row in
`session`. `POST /update` additionally 403s if the token's resolved
user doesn't match `submission.userId` in the request body -- a token
only works for the user it was minted for. The two seeding requests
separately ALSO send an `X-User-Id` header (already wired into the
collection) on top of the Bearer token -- that's a documented placeholder
equality check, not real auth on its own; see `routes/seeding.py`'s
`require_same_user` docstring.

## 1. Root (health check)

`GET /` -- should return `{"message": "Welcome to Recommendation Service"}`.
Confirms the server is up before testing anything else.

## 2. Telemetry Scenarios (`POST /update`)

This is the core of the mastery-update fix. All seven requests post to the
same stateless `/update` endpoint; each demonstrates one specific mechanism
in `pipeline/recommender/bkt.py`'s `update_bkt` and
`pipeline/recommender/telemetry.py`'s `compute_telemetry_signal`. Every
body already lives in `postman/telemetry_samples/*.json`, so you can also
inspect/edit them directly.

Send them in order (each stands alone -- `currentMastery` is hardcoded per
scenario rather than chained from the previous response, so you can also
run them individually or out of order).

**Topic resolution:** `process_submission`/`process_hlr` use the
`problemTopics` list from the request body when present (each entry's
`topicId`, seeded from `currentMastery` if given), falling back to the
static `data/problem_topic_edges_normalized.json` mapping (keyed by
`problemId`) only when `problemTopics` is omitted or empty. The table
below reports the topics/values actually touched (re-verified directly
against the current merged `bkt.py`, including its current `BKT_PARAMS`
tuning -- these numbers WILL drift again if `BKT_PARAMS` changes; see
"Reproducing these numbers" to regenerate them yourself).

| # | Scenario | Topic | Before | After | Delta | Why |
|---|----------|-------|--------|-------|-------|-----|
| 1 | Cold start, easy solve | `array`, `hash_map` | 0.15 (default) | **0.2127** | +0.0627 | First-ever submission. The difficulty-scaled `MAX_MASTERY_DELTA` cap (easy problem -> smaller cap) plus mastery-proximity dampening bound this to a small, controlled step instead of a large uncapped jump. |
| 2 | Struggling, many attempts | `two_pointers` (seeded 0.30), `hash_map`, `string` (default) | 0.30 / 0.15 | **0.254** / **0.1496** | −0.0460 / −0.0004 | 7 submissions + 4 hints pulls the confidence-adjusted signal (0.3889) below `LEARNING_TRANSITION_THRESHOLD` (0.5) -- no learning-transition credit applies, so this reads as a slight regression rather than a gain, even though the final verdict is OK. |
| 3 | Wrong answer | `two_pointers` (seeded 0.50) | 0.50 | **0.3872** | −0.1128 | `verdict != "OK"` caps the signal at 0.35 -- clear negative evidence, and decreases are NOT subject to the same diminishing-returns dampening as gains. |
| 4 | Moderate difficulty, clean solve | `hash_map` (seeded 0.40) | 0.40 | **0.4453** | +0.0453 | A clean one-shot solve at roughly neutral difficulty (0.5). Solid, bounded gain -- the "normal" case. |
| 5a | Warmup before trivial (near-mastery) | `hash_map` (seeded 0.75) | 0.75 | **0.762** | +0.0120 | `hash_map` is already well-mastered (0.75). Mastery-proximity dampening (`_PROXIMITY_DAMPEN_FLOOR`/`_PROXIMITY_DECAY_RATE` in `bkt.py`) shrinks the gain sharply compared to a cold-start topic on the same signal strength -- diminishing returns as mastery increases. |
| 5b | Trivial relative to mastery | `hash_map` (seeded 0.85) | 0.85 | **0.8292** | −0.0208 | `hash_map` mastery=0.85, problem difficulty only 0.05 -- far below the user's level (gap well past `_TRIVIAL_GAP_THRESHOLD`). Combined with the confidence-adjusted signal, this nets to essentially flat/slightly negative rather than a meaningful gain: solving something you've clearly outgrown teaches you almost nothing. |
| 6 | Hard stretch solve | `binary_search_answer` (seeded 0.40), `array` (default) | 0.40 / 0.15 | **0.4612** / **0.2508** | +0.0612 / +0.1008 | difficulty=0.95 (hard) with a perfect score -- the difficulty-scaled cap is at its widest here, and the cold-start `array` topic in particular shows the largest single gain in this set: appropriately challenging practice rewarded more than trivial practice. |

**Diminishing returns as mastery grows** -- run
`postman/demo_progression.py` or the snippet in "Reproducing these
numbers" below for a live, current table (values depend on `BKT_PARAMS`,
which is not pinned in this doc since it's tunable independent of the
dampening mechanism itself). The delta should still shrink monotonically
as starting mastery climbs, regardless of the exact `BKT_PARAMS` values in
use -- that shape is what mastery-proximity dampening guarantees, not the
absolute numbers.

**Response shape** for every `/update` call:

```json
{
  "userId": "postman_demo_user",
  "problemId": "two-sum",
  "updatedTopics": [
    {
      "topicId": "array",
      "updatedMastery": 0.2306,
      "updatedHlr": { "half_life": 1.19, "next_review_days": 0.7, ... }
    }
  ],
  "masteredTopics": [],
  "results": { "bkt": [...], "hlr": [...] }
}
```

## 3. Get mastery + proficiency (`GET /mastery/{{user_id}}`)

Reads whatever's currently in Postgres for `user_id` (populated by the
seeding routes below, or by your backend's own writes -- `/update` itself
never writes to Postgres, per the stateless design). Returns:

```json
{
  "userId": "postman_demo_user",
  "mastery": { "array": 0.42 },
  "mastered_topics": [],
  "proficiency": { "array": 0.39 }
}
```

`proficiency` is new: raw `mastery` decayed by the HLR forgetting curve's
recall probability -- "how good are they at this topic RIGHT NOW" rather
than "the highest mastery they've ever demonstrated." A topic mastered
long ago with no recent review shows a lower `proficiency` than `mastery`.

## 4. Get urgency (`GET /urgency/{{user_id}}`)

Per-topic urgency score (0-1) from the HLR forgetting curve -- higher means
more likely forgotten and due for review.

## 5. Get recommendations (`GET /recommend/{{user_id}}?limit=10`)

Runs the full ML pipeline (candidate pools -> ranking). Needs Qdrant and
Neo4j reachable (see RUNNING.md step 2); degrades to a cold-start pool if
Postgres mastery/HLR reads fail or the user has none yet.

## 6. Seeding (`POST /seed_hlr/{{user_id}}`, `POST /seed_bkt/{{user_id}}`)

Backfills mastery/HLR from a linked Codeforces/LeetCode handle. Requires
the `user` row to have `linked_codeforces`/`linked_leetcode` set first --
insert those directly via `psql` for local testing, since there's no route
for it in this service (account linking is a backend concern).

## Reproducing these numbers

No server or database needed -- exercises the exact same code path
`POST /update` does (`Submission` schema validation + `handle_update`):

```bash
python -c "
import json
from models.schemas.submission import Submission
from controllers.submission_controller import handle_update

body = json.load(open('postman/telemetry_samples/01_cold_start_easy_solve.json'))
result = handle_update(Submission(**body))
for r in result['results']['bkt']:
    print(r['topic'], r['previous_p_l'], '->', r['new_p_l'])
"
```

Swap the filename to reproduce any row in the table above.

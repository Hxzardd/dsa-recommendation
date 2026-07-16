# Postman Guide

Request-by-request walkthrough of `postman_collection.json`, with expected
results, for testing this service's endpoints (mastery updates in
particular) without writing any code. See [RUNNING.md](RUNNING.md) for how
to get the server and Postgres running first.

## 0. Setup

1. `psql "$DATABASE_URL" -f database/postgres/dev_schema.sql`
2. `python seed_test_session.py postman_demo_user` -- creates the test user
   row (add `--cf`/`--lc` if you also want to exercise the seeding routes).
3. In Postman: File > Import > select both `postman_collection.json` and
   `postman_environment.json`.
4. Select the "DSA ML Service - Local" environment (top-right dropdown).
5. Start the server: `uvicorn main:app --reload --port 8000`.

No auth setup needed -- at this commit (813b4e9), `POST /update`,
`GET /mastery`, `GET /urgency`, and `GET /recommend` are all unauthenticated.
Only the two seeding requests send an `X-User-Id` header (already wired
into the collection), and it's a documented placeholder, not real auth --
see `routes/seeding.py`'s `require_same_user` docstring.

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

**Important quirk at this commit:** `process_submission`/`process_hlr`
resolve which topics a problem touches from the static
`data/problem_topic_edges_normalized.json` mapping (keyed by `problemId`),
**not** from the `problemTopics` list in the request body. The
`problemTopics` list only supplies `currentMastery`/`currentHlr` *seed
values*, keyed by topic slug -- if the static mapping returns a topic slug
that isn't in your `problemTopics` list, it silently starts from the
default (`DEFAULT_P_L["branch"] = 0.15`) instead of whatever you intended
to seed. This is existing behavior, not something this update changes --
the table below reports the topics that are *actually* touched (verified
by running each sample through `handle_update` directly, see "Reproducing
these numbers").

| # | Scenario | Topic | Before | After | Delta | Why |
|---|----------|-------|--------|-------|-------|-----|
| 1 | Cold start, easy solve | `array`, `hash_map`, `simulation` | 0.15 (default) | **0.2306** | +0.0806 | First-ever submission. Uncapped Bayesian math would have jumped this to ~0.69 off one solve; the difficulty-scaled `MAX_MASTERY_DELTA` cap (0.12 x 0.67 for an easy problem) bounds it to a small, controlled step. |
| 2 | Struggling, many attempts | `string`, `hash_map`, `two_pointers`, `hash_map_counting` | 0.15 (default) | **0.2353** | +0.0853 | 7 submissions + 4 hints. The confidence penalty in `telemetry.py` pulls the observed signal down even though the final verdict is OK -- thrashing actively suppresses the gain, not just fails to reward it (compare to scenario 4's clean-solve delta at a similar starting point). |
| 3 | Wrong answer | `array` (default) / `two_pointers` (seeded 0.50) | 0.15 / 0.50 | **0.2157** / **0.3872** | +0.0657 / −0.1128 | `verdict != "OK"` caps the signal at 0.35. `two_pointers` had a seeded `currentMastery` matching the static mapping's slug, so its real decrease is visible: a wrong answer is genuine negative evidence, and decreases are NOT subject to the same diminishing-returns dampening as gains. |
| 4 | Moderate difficulty, clean solve | `array`, `simulation` (default) / `hash_map` (seeded 0.40) | 0.15 / 0.40 | **0.2459** / **0.4677** | +0.0959 / +0.0677 | A clean one-shot solve at roughly neutral difficulty (0.4). Solid, bounded gain either way -- this is the "normal" case. |
| 5a | Warmup before trivial (near-mastery) | `string`, `simulation` (default) / `hash_map` (seeded 0.75) | 0.15 / 0.75 | **0.2398** / **0.7645** | +0.0898 / +0.0145 | `hash_map` is already well-mastered (0.75). Mastery-proximity dampening (`_PROXIMITY_DAMPEN_FLOOR` in `bkt.py`) shrinks its gain to a fraction of the default-mastery topics' gain -- diminishing returns as mastery increases, exactly as requested. |
| 5b | Trivial relative to mastery | `array`, `simulation` (default) / `hash_map` (seeded 0.85) | 0.15 / 0.85 | **0.2245** / **0.8506** | +0.0745 / +0.0006 | `hash_map` mastery=0.85, but the problem's difficulty is only 0.05 -- far below the user's level. Both mastery-proximity AND trivial-gap dampening stack here, so ITS gain is almost zero, in sharp contrast to the default-mastery topics on the same submission: solving something you've clearly outgrown teaches you almost nothing new. |
| 6 | Hard stretch solve | `array`, `binary_search_answer` | 0.15 (default) | **0.2795** | +0.1295 | Moderate-difficulty-agnostic starting point solving a HARD problem (difficulty=0.95) with a hint and a retry. The largest gain in this set -- appropriately challenging practice is rewarded more than trivial practice, even after the (mild) confidence penalty from the extra attempt/hint. |

**Diminishing returns as mastery grows, illustrated in one line** (same
perfect solve, difficulty=0.5 neutral, replayed at increasing starting
mastery):

```
mastery=0.15 -> 0.2520  (delta +0.1020)
mastery=0.30 -> 0.3840  (delta +0.0840)
mastery=0.50 -> 0.5600  (delta +0.0600)
mastery=0.65 -> 0.6857  (delta +0.0357)
mastery=0.80 -> 0.8210  (delta +0.0210)
mastery=0.90 -> 0.9135  (delta +0.0135)
mastery=0.97 -> 0.9736  (delta +0.0036)
```

The delta shrinks monotonically as mastery climbs -- the higher a user's
mastery already is, the less any single solve moves it, exactly the
behavior requested. (Reproduce this table yourself: `python -c` snippet in
the "Reproducing these numbers" section below.)

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

# Running This Project

Quick reference for installing, running, and testing this repo end to end.
For the Postman walkthrough (scenarios, expected mastery values), see
[POSTMAN_GUIDE.md](POSTMAN_GUIDE.md).

## 1. Install dependencies

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

On Windows inside a OneDrive-synced folder, `uv sync`'s default hardlink
mode can fail against OneDrive's Files-On-Demand placeholders. If you hit
`Access is denied` / "incompatible hardlinks" errors:

```bash
uv sync --link-mode=copy
```

## 2. Environment variables

Create a `.env` file at the repo root:

```
DATABASE_URL=postgresql://user:password@localhost:5432/dsa_dev

# Only needed for GET /recommend (the full ML pipeline) -- POST /update,
# GET /mastery, GET /urgency, and the seeding routes only need DATABASE_URL.
QDRANT_URL=...
QDRANT_API_KEY=...
NEO4J_URI=...
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
NEO4J_DATABASE=neo4j
```

`DATABASE_URL` is read directly by `database/postgres/db.py`. Qdrant/Neo4j
credentials are read by `db_env.py` -- see its module docstring for where
each value comes from (Qdrant Cloud dashboard / Neo4j Aura credentials file).

## 3. Set up Postgres for local testing

This ML service never migrates the `user`/`user_topic_mastery`/
`user_hlr_state` tables -- in production those belong to the backend repo.
For standalone local testing, a minimal schema is provided:

```bash
psql "$DATABASE_URL" -f database/postgres/dev_schema.sql
```

Then create a test user row (needed for `GET /mastery`, `GET /urgency`, and
the seeding routes to have anything to read):

```bash
python seed_test_session.py postman_demo_user
# optionally link handles to exercise the seeding routes:
python seed_test_session.py postman_demo_user --cf my_cf_handle --lc my_lc_handle
```

**Auth note for this commit (813b4e9):** `POST /update`, `GET /mastery`,
`GET /urgency`, and `GET /recommend` have no auth at all -- call them
directly. Only `POST /seed_hlr/{user_id}` and `POST /seed_bkt/{user_id}`
check anything, and it's a documented placeholder (`routes/seeding.py`'s
`require_same_user`): an `X-User-Id` header that must equal the path's
`user_id`, with no real signature/session verification behind it. See that
file's module docstring before relying on this for anything beyond local
testing.

## 4. Run the API server

```bash
uvicorn main:app --reload --port 8000
```

- Swagger UI: `http://localhost:8000/docs`
- OpenAPI schema: `http://localhost:8000/openapi.json`

## 5. Run the test suite

```bash
uv run pytest tests/ -q
```

Runs entirely offline against pure-Python modules (BKT/HLR/telemetry) --
no real Postgres/Qdrant/Neo4j needed for those. Run a single file with e.g.:

```bash
uv run pytest tests/test_bkt.py tests/test_telemetry.py -v
```

## 6. Postman

Import both `postman_collection.json` and `postman_environment.json` into
Postman (File > Import), select the "DSA ML Service - Local" environment,
and see [POSTMAN_GUIDE.md](POSTMAN_GUIDE.md) for the full request-by-request
walkthrough with expected results. No token/auth setup needed except for
the two seeding requests (`X-User-Id` header, already wired into the
collection).

## Architecture notes relevant to this update

- **Stateless `/update`**: the backend sends the user's current
  mastery/HLR state per topic IN the request body (`problemTopics`); ML
  computes new values and returns them. ML never reads/writes mastery to
  Postgres on this path -- the backend owns persistence. See
  `models/schemas/submission.py`'s module docstring.
- **Shared telemetry signal** (`pipeline/recommender/telemetry.py`): BKT
  and HLR now derive "how well did the learner do on this submission" from
  one shared calculation instead of two independently-drifting formulas.
  Includes a confidence penalty for thrashing (excess attempts/hints) and
  an optional difficulty credit.
- **Bounded, diminishing-returns mastery updates** (`pipeline/recommender/bkt.py`):
  a single submission can move a topic's mastery by at most
  `MAX_MASTERY_DELTA` (0.12), scaled by problem difficulty (0.084-0.156),
  further shrunk as the topic's current mastery approaches 1.0 (mastery-
  proximity dampening) and if the problem is trivial relative to current
  mastery (trivial-gap dampening). This replaces the previous uncapped
  Bayesian update, which could swing a cold-start topic (P_L=0.15) to
  0.6+ off a single strong submission.

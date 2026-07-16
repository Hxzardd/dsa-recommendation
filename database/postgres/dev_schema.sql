-- Minimal local-dev schema for standalone testing of this ML service.
--
-- Production tables (user, user_topic_mastery, user_hlr_state, session) are
-- owned by the backend repo, not this one -- this ML service only ever
-- reads/writes them, never migrates them. This file exists purely so this
-- repo can be smoke-tested (pytest is fully mocked and needs none of this,
-- but GET /mastery, GET /urgency, POST /seed_hlr, POST /seed_bkt, and now
-- every route's auth middleware all read the columns database/postgres/
-- db.py and middlewares/auth.py query) against a throwaway local Postgres
-- without needing the full backend stack running.
--
-- NOTE on auth: as of the middlewares/auth.py merge, EVERY route except
-- "/", "/docs", "/openapi.json", "/redoc" requires a real Bearer token
-- that resolves to a non-expired row in `session`. There is no more
-- unauthenticated path for POST /update, GET /mastery, GET /urgency, or
-- GET /recommend. POST /seed_hlr/{user_id} and POST /seed_bkt/{user_id}
-- separately ALSO check X-User-Id (routes/seeding.py's require_same_user,
-- a documented placeholder equality check, not real verification) --
-- that's on top of, not instead of, the Bearer token auth_middleware now
-- requires on every route.
--
-- Usage:
--   psql "$DATABASE_URL" -f database/postgres/dev_schema.sql
--
-- Then seed a test user AND a real session token with seed_test_session.py.

CREATE TABLE IF NOT EXISTS "user" (
    id                TEXT PRIMARY KEY,
    linked_codeforces  TEXT,
    linked_leetcode    TEXT
);

CREATE TABLE IF NOT EXISTS user_topic_mastery (
    user_id       TEXT NOT NULL REFERENCES "user"(id),
    topic_id      TEXT NOT NULL,
    mastery_score DOUBLE PRECISION NOT NULL,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, topic_id)
);

CREATE TABLE IF NOT EXISTS user_hlr_state (
    user_id           TEXT NOT NULL REFERENCES "user"(id),
    topic_id          TEXT NOT NULL,
    half_life         DOUBLE PRECISION NOT NULL,
    last_review       TIMESTAMPTZ,
    p_recall          DOUBLE PRECISION,
    next_review_days  DOUBLE PRECISION,
    PRIMARY KEY (user_id, topic_id)
);

-- Matches the real deployed schema exactly (verified against a live
-- instance: id, expires_at, token, created_at, updated_at, ip_address,
-- user_id, user_agent) -- middlewares/auth.py::verify_session_token reads
-- user_id/expires_at by token.
CREATE TABLE IF NOT EXISTS session (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES "user"(id),
    token       TEXT NOT NULL UNIQUE,
    expires_at  TIMESTAMPTZ NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ip_address  TEXT,
    user_agent  TEXT
);

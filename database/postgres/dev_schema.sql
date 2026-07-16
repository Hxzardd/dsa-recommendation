-- Minimal local-dev schema for standalone testing of this ML service at
-- commit 813b4e9 ("p1 and p2").
--
-- Production tables (user, user_topic_mastery, user_hlr_state) are owned by
-- the backend repo, not this one -- this ML service only ever reads/writes
-- them, never migrates them. This file exists purely so this repo can be
-- smoke-tested (pytest is fully mocked and needs none of this, but
-- GET /mastery, GET /urgency, POST /seed_hlr, and POST /seed_bkt all read
-- the columns database/postgres/db.py queries) against a throwaway local
-- Postgres without needing the full backend stack running.
--
-- NOTE on auth at this commit: POST /update, GET /mastery, GET /urgency,
-- and GET /recommend have NO auth at all -- call them directly. Only
-- POST /seed_hlr/{user_id} and POST /seed_bkt/{user_id} check anything,
-- and it's a documented PLACEHOLDER (routes/seeding.py's
-- require_same_user): an X-User-Id header with no signature/session
-- verification, just user_id-in-path == X-User-Id-header. No `session`
-- table exists or is needed at this commit.
--
-- Usage:
--   psql "$DATABASE_URL" -f database/postgres/dev_schema.sql
--
-- Then seed a test user (and optionally linked CF/LC handles) with
-- seed_test_session.py.

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

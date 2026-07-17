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

-- email is NOT NULL with no default in the real deployed schema --
-- seed_test_session.py synthesizes "{user_id}@test.local" for it.
--
-- name/onboarding_completed are read by UserGraphService._fetch_user()
-- (SELECT u.id, u.name, u.onboarding_completed, x.total_xp, x.current_level
-- FROM "user" u LEFT JOIN user_xp x ...) -- without them that query raises,
-- _fetch_user's try/except swallows it and returns None, _build() then
-- raises ValueError("User not found"), and _get_graph's caller falls back
-- to a brand-new cold-start graph -- silently discarding whatever real
-- mastery/HLR state get_user_mastery()/get_user_hlr() already loaded, for
-- EVERY user, on EVERY /recommend and /topic/recommend call locally.
CREATE TABLE IF NOT EXISTS "user" (
    id                    TEXT PRIMARY KEY,
    email                 TEXT NOT NULL,
    name                  TEXT,
    onboarding_completed  BOOLEAN NOT NULL DEFAULT FALSE,
    linked_codeforces     TEXT,
    linked_leetcode       TEXT
);

-- LEFT JOINed by _fetch_user() for total_xp/current_level -- absence alone
-- wasn't fatal (LEFT JOIN + `x.total_xp` reads as NULL -> defaults to 0/1
-- in UserNode), but the table itself must exist or the whole SELECT
-- throws before the LEFT JOIN's NULL-tolerance ever applies.
CREATE TABLE IF NOT EXISTS user_xp (
    user_id       TEXT PRIMARY KEY REFERENCES "user"(id),
    total_xp      INTEGER NOT NULL DEFAULT 0,
    current_level INTEGER NOT NULL DEFAULT 1
);

-- Backend-owned topic catalog. user_topic_mastery.topic_id and
-- user_hlr_state.topic_id are FKs into this table's opaque `id` (NOT an
-- ML-pipeline slug like "array") -- database/postgres/topic_taxonomy.py
-- translates between the two via this table's `slug` column, and
-- database/postgres/db.py's _load_topic_cache() queries `SELECT id, slug
-- FROM topic` directly, so this table must exist with real slug values
-- for any mastery/HLR read or write to work locally. `id = slug` here for
-- simplicity (the real deployed table uses opaque generated ids, but
-- topic_taxonomy.py only ever looks up by `slug`, never assumes a
-- particular `id` format, so this is a safe substitution for local dev).
CREATE TABLE IF NOT EXISTS topic (
    id    TEXT PRIMARY KEY,
    slug  TEXT NOT NULL UNIQUE
);

INSERT INTO topic (id, slug) VALUES
    ('arrays', 'arrays'),
    ('backtracking', 'backtracking'),
    ('bfs', 'bfs'),
    ('binary-indexed-tree', 'binary-indexed-tree'),
    ('binary-lifting', 'binary-lifting'),
    ('binary-search', 'binary-search'),
    ('binary-search-tree', 'binary-search-tree'),
    ('bitmask-dp', 'bitmask-dp'),
    ('bitwise', 'bitwise'),
    ('design', 'design'),
    ('dfs', 'dfs'),
    ('digit-dp', 'digit-dp'),
    ('dijkstra', 'dijkstra'),
    ('dynamic-programming', 'dynamic-programming'),
    ('fast-slow-pointers', 'fast-slow-pointers'),
    ('floyd-warshall', 'floyd-warshall'),
    ('geometry', 'geometry'),
    ('graphs', 'graphs'),
    ('greedy', 'greedy'),
    ('hash-map', 'hash-map'),
    ('heap', 'heap'),
    ('kmp', 'kmp'),
    ('linked-lists', 'linked-lists'),
    ('math', 'math'),
    ('merge-intervals', 'merge-intervals'),
    ('monotonic-stack', 'monotonic-stack'),
    ('number-theory', 'number-theory'),
    ('prefix-sum', 'prefix-sum'),
    ('priority-queue', 'priority-queue'),
    ('queue', 'queue'),
    ('recursion', 'recursion'),
    ('segment-tree', 'segment-tree'),
    ('shortest-path', 'shortest-path'),
    ('simulation', 'simulation'),
    ('sliding-window', 'sliding-window'),
    ('stacks', 'stacks'),
    ('string-matching', 'string-matching'),
    ('strings', 'strings'),
    ('topological-sort', 'topological-sort'),
    ('tree-dp', 'tree-dp'),
    ('trees', 'trees'),
    ('trie', 'trie'),
    ('two-pointers', 'two-pointers'),
    ('union-find', 'union-find')
ON CONFLICT (id) DO NOTHING;

-- confidence..next_review_date are read by
-- UserGraphService._load_topic_mastery() (SELECT topic_id, mastery_score,
-- confidence, attempt_count, problems_solved, last_attempted, sm2_ef,
-- sm2_interval, next_review_date FROM user_topic_mastery ...) -- without
-- them that SELECT raises, the try/except swallows it and returns early,
-- so a locally-bootstrapped graph gets ZERO concept edges even when
-- get_user_mastery() (a separate, narrower query) already succeeded --
-- same silent-fallback failure mode as the "user" table fix above.
CREATE TABLE IF NOT EXISTS user_topic_mastery (
    user_id          TEXT NOT NULL REFERENCES "user"(id),
    topic_id         TEXT NOT NULL REFERENCES topic(id),
    mastery_score    DOUBLE PRECISION NOT NULL,
    confidence       TEXT,              -- 'low' | 'medium' | 'high', defaults to 'medium' if null
    attempt_count    INTEGER,
    problems_solved  INTEGER,
    last_attempted   TIMESTAMPTZ,
    sm2_ef           DOUBLE PRECISION,  -- defaults to 2.5 if null
    sm2_interval     INTEGER,           -- defaults to 1 if null
    next_review_date DATE,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, topic_id)
);

CREATE TABLE IF NOT EXISTS user_hlr_state (
    user_id           TEXT NOT NULL REFERENCES "user"(id),
    topic_id          TEXT NOT NULL REFERENCES topic(id),
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

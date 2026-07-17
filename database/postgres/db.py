import os
import uuid
from datetime import datetime, timezone

import psycopg2
from pathlib import Path
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from psycopg2.pool import ThreadedConnectionPool, PoolError


from database.postgres.topic_taxonomy import topic_slug_for_ml_slug, ml_slug_for_topic_slug

# Walk up from this file to find the repo root .env -- works regardless
# of what directory uvicorn is launched from.
_here = Path(__file__).resolve()
for _p in [_here.parent, *_here.parents]:
    if (_p / ".env").exists():
        load_dotenv(_p / ".env")
        break
else:
    load_dotenv()  # fallback
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set")

db_pool = None

# Cache of the `topic` table's id<->slug columns. Small (44 rows) and
# effectively static (backend-owned, this service never writes to it), so
# one query per process is enough rather than a join on every mastery
# read/write. Populated lazily on first use.
_topic_id_by_slug = None
_topic_slug_by_id = None


def _load_topic_cache(conn):
    global _topic_id_by_slug, _topic_slug_by_id
    with conn.cursor() as cur:
        cur.execute("SELECT id, slug FROM topic")
        rows = cur.fetchall()
    _topic_id_by_slug = {slug: tid for tid, slug in rows}
    _topic_slug_by_id = {tid: slug for tid, slug in rows}


def _ml_slug_to_topic_id(conn, ml_slug: str) -> str | None:
    """ML pipeline slug (e.g. "array") -> topic.id, via topic_taxonomy's
    verified slug mapping + a live lookup against the real topic table.
    None if the ML pipeline emits a slug with no backend counterpart --
    callers must skip the write for that topic, not guess an id."""
    global _topic_id_by_slug
    if _topic_id_by_slug is None:
        _load_topic_cache(conn)
    topic_slug = topic_slug_for_ml_slug(ml_slug)
    if topic_slug is None:
        return None
    return _topic_id_by_slug.get(topic_slug)


def _topic_id_to_ml_slug(conn, topic_id: str) -> str | None:
    """Reverse of _ml_slug_to_topic_id -- topic.id -> ML pipeline slug, so
    values read back out of user_topic_mastery/user_hlr_state are usable
    by UserGraph/the pools (which key everything by ML slug)."""
    global _topic_slug_by_id
    if _topic_slug_by_id is None:
        _load_topic_cache(conn)
    topic_slug = _topic_slug_by_id.get(topic_id)
    if topic_slug is None:
        return None
    return ml_slug_for_topic_slug(topic_slug)


def get_connection():
    global db_pool

    if db_pool is None:
        db_pool = ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=DATABASE_URL,
        )

    try:
        return db_pool.getconn()
    except PoolError as e:
        raise RuntimeError(
            "Database connection pool exhausted. Please try again."
        ) from e

def release_connection(conn):
    if db_pool is not None:
        db_pool.putconn(conn)

def get_user_mastery(user_id: str) -> dict:
    """
    Returns {ml_slug: mastery_score}. user_topic_mastery.topic_id stores the
    backend's opaque topic.id (FK), so each row is translated back to the
    ML pipeline's slug (e.g. "array") via topic_taxonomy -- UserGraph/the
    pools key everything by ML slug, not the backend's topic.id. Rows for
    a topic.id with no ML-slug counterpart (topic_taxonomy.UNMAPPED_TOPIC_SLUGS)
    are skipped rather than surfaced under a wrong/guessed key.
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT topic_id, mastery_score FROM user_topic_mastery WHERE user_id = %s ORDER BY updated_at DESC",
                (user_id,)
            )
            rows = cur.fetchall()
            out = {}
            for row in rows:
                ml_slug = _topic_id_to_ml_slug(conn, row["topic_id"])
                if ml_slug is None or ml_slug in out:
                    continue  # unmapped topic, or a staler row for a slug we already have
                out[ml_slug] = float(row["mastery_score"]) if row["mastery_score"] is not None else 0.0
            return out
    finally:
     release_connection(conn)


def get_user_hlr(user_id: str) -> dict:
    """Returns {ml_slug: hlr_state}. See get_user_mastery's docstring --
    same topic.id -> ML slug translation applies here."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT topic_id, half_life, last_review, p_recall, next_review_days FROM user_hlr_state WHERE user_id = %s",
                (user_id,)
            )
            rows = cur.fetchall()
            out = {}
            for row in rows:
                ml_slug = _topic_id_to_ml_slug(conn, row["topic_id"])
                if ml_slug is None or ml_slug in out:
                    continue
                out[ml_slug] = {
                    "half_life": float(row["half_life"]) if row["half_life"] is not None else 1.0,
                    "last_review": str(row["last_review"]) if row["last_review"] is not None else None,
                    "p_recall": float(row["p_recall"]) if row["p_recall"] is not None else 0.5,
                    "next_review_days": float(row["next_review_days"]) if row["next_review_days"] is not None else 1.0
                }
            return out
    finally:
        release_connection(conn)


def save_user_hlr(user_id: str, hlr_state: dict):
    """hlr_state is keyed by ML pipeline slug (e.g. "array"); translated to
    the backend's topic.id via topic_taxonomy before writing (see
    _ml_slug_to_topic_id). Topics with no backend counterpart are skipped --
    there is no row to write to."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for ml_slug, state in hlr_state.items():
                topic_id = _ml_slug_to_topic_id(conn, ml_slug)
                if topic_id is None:
                    continue
                cur.execute("""
                    INSERT INTO user_hlr_state (user_id, topic_id, half_life, last_review, p_recall, next_review_days)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, topic_id)
                    DO UPDATE SET
                        half_life = EXCLUDED.half_life,
                        last_review = EXCLUDED.last_review,
                        p_recall = EXCLUDED.p_recall,
                        next_review_days = EXCLUDED.next_review_days
                """, (
                    user_id, topic_id,
                    state["half_life"], state.get("last_review"),
                    state.get("p_recall", 0.5), state.get("next_review_days", 1.0)
                ))
        conn.commit()
    finally:
        release_connection(conn)


def save_user_mastery(user_id: str, mastery: dict):
    """
    Write BKT mastery scores to user_topic_mastery table.
    Used by seeding_controller.py when seeding initial mastery from
    LeetCode/Codeforces history. Uses ON CONFLICT DO NOTHING so it
    never overwrites mastery that was already computed from real
    in-platform submissions.

    mastery is keyed by ML pipeline slug (e.g. "array"); translated to the
    backend's topic.id via topic_taxonomy before writing -- topic_id is a
    real FK to the topic table, and the raw ML slug is never a valid value
    for it (mismatched naming convention -- see topic_taxonomy.py). Topics
    with no backend counterpart are skipped, not guessed.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for ml_slug, mastery_score in mastery.items():
                topic_id = _ml_slug_to_topic_id(conn, ml_slug)
                if topic_id is None:
                    continue
                cur.execute("""
                    INSERT INTO user_topic_mastery (user_id, topic_id, mastery_score, updated_at)
                    VALUES (%s, %s, %s, NOW())
                    ON CONFLICT (user_id, topic_id) DO NOTHING
                """, (user_id, topic_id, mastery_score))
        conn.commit()
    finally:
        release_connection(conn)


def save_user_mastery_live(user_id: str, mastery: dict):
    """
    Write freshly-recomputed BKT mastery scores from a LIVE submission
    (submission_controller.py::handle_update) -- unlike save_user_mastery
    (CF/LC one-time import, ON CONFLICT DO NOTHING so it never clobbers
    real in-platform progress), this MUST overwrite on every call: it's
    the new P(L) after this exact submission, and mastery is meant to be
    the single source of truth for that going forward. Same topic-id
    translation as save_user_mastery -- see topic_taxonomy.py.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for ml_slug, mastery_score in mastery.items():
                topic_id = _ml_slug_to_topic_id(conn, ml_slug)
                if topic_id is None:
                    continue
                cur.execute("""
                    INSERT INTO user_topic_mastery (user_id, topic_id, mastery_score, updated_at)
                    VALUES (%s, %s, %s, NOW())
                    ON CONFLICT (user_id, topic_id) DO UPDATE SET
                        mastery_score = EXCLUDED.mastery_score,
                        updated_at = NOW()
                """, (user_id, topic_id, mastery_score))
        conn.commit()
    finally:
        release_connection(conn)


def resolve_problem_ids_by_title_slugs(title_slugs: list) -> dict:
    """
    ML's own `problem_id` (this repo's internal ingestion hash, e.g.
    "f07r6rjgpeoc8h18g24o9ye9") is NOT the backend's `problem.problem_id`
    (a CUID) -- recommendation_log.problem_id FKs the latter. The join key
    both sides agree on is `title_slug` (LeetCode-style, e.g. "two-sum"),
    which is already on every recommendation (_resolve_titles in
    recommend.py reads it straight from Qdrant payload). Bulk resolver:
    {title_slug: backend problem.problem_id}, skipping slugs with no match
    (a problem present in our Qdrant catalog but not yet in the backend's
    `problem` table) rather than guessing.
    """
    if not title_slugs:
        return {}
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT title_slug, problem_id FROM problem WHERE title_slug = ANY(%s)",
                (list(title_slugs),),
            )
            return {row[0]: row[1] for row in cur.fetchall()}
    finally:
        release_connection(conn)


def save_recommendation_log(user_id: str, recommendations: list, session_mode: str = None):
    """
    One row per recommended problem, was_attempted/was_skipped starting
    False -- this is the write half of the attempt/skip feedback loop (see
    mark_recommendation_attempted for the other half, called from
    submission_controller.py::handle_update). Resolves each recommendation's
    title_slug to the backend's real problem.problem_id via
    resolve_problem_ids_by_title_slugs; recommendations with no title_slug
    or no match in the backend's `problem` table are skipped rather than
    writing a row with a guessed/invalid FK.

    The real deployed schema has a partial unique index,
    recommendation_log_pending_unique_idx, on (user_id, problem_id, source)
    WHERE was_attempted = false AND source IS NOT NULL -- at most one
    PENDING (not-yet-attempted) recommendation per user+problem+pool at a
    time. ON CONFLICT below matches that exact partial index (Postgres
    requires the predicate to match for a partial unique index) and does
    nothing on a repeat -- the same problem being recommended again by the
    same pool before the user has acted on it isn't a new event worth a
    second log row.
    """
    title_slugs = [r.get("title_slug") for r in recommendations if r.get("title_slug")]
    problem_ids_by_slug = resolve_problem_ids_by_title_slugs(title_slugs)

    rows = []
    now = datetime.now(timezone.utc)
    for r in recommendations:
        title_slug = r.get("title_slug")
        problem_id = problem_ids_by_slug.get(title_slug) if title_slug else None
        if problem_id is None:
            continue
        rows.append((
            str(uuid.uuid4()), user_id, problem_id, now,
            r.get("source"), False, False, 0, session_mode,
        ))

    if not rows:
        return

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.executemany("""
                INSERT INTO recommendation_log
                    (log_id, user_id, problem_id, recommended_at, source,
                     was_attempted, was_skipped, skip_count, session_mode)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, problem_id, source)
                WHERE was_attempted = false AND source IS NOT NULL
                DO NOTHING
            """, rows)
        conn.commit()
    finally:
        release_connection(conn)


def mark_recommendation_attempted(user_id: str, title_slug: str):
    """
    Read half of the attempt/skip feedback loop's write side --
    submission_controller.py::handle_update calls this so a submission on
    a problem the user was actually recommended marks that
    recommendation_log row was_attempted=TRUE. Only meaningful if the
    problem resolves to a real backend problem.problem_id (same
    translation as save_recommendation_log); otherwise a no-op, not an
    error -- most submissions won't correspond to a prior /recommend call
    at all (a user can solve any problem, recommended or not).
    """
    if not title_slug:
        return
    resolved = resolve_problem_ids_by_title_slugs([title_slug])
    problem_id = resolved.get(title_slug)
    if problem_id is None:
        return

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE recommendation_log
                SET was_attempted = TRUE
                WHERE user_id = %s AND problem_id = %s AND was_attempted = FALSE
            """, (user_id, problem_id))
        conn.commit()
    finally:
        release_connection(conn)
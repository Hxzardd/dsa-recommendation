import os
import psycopg2
from pathlib import Path
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from psycopg2.pool import ThreadedConnectionPool

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

def get_connection():
    global db_pool

    if db_pool is None:
        db_pool = ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=DATABASE_URL,
        )

    return db_pool.getconn()

def release_connection(conn):
    if db_pool is not None:
        db_pool.putconn(conn)

def get_user_mastery(user_id: str) -> dict:
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT topic_id, mastery_score FROM user_topic_mastery WHERE user_id = %s ORDER BY updated_at DESC",
                (user_id,)
            )
            rows = cur.fetchall()
            return {
                row["topic_id"]: float(row["mastery_score"]) if row["mastery_score"] is not None else 0.0
                for row in rows
            }
    finally:
     release_connection(conn)


def get_user_hlr(user_id: str) -> dict:
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT topic_id, half_life, last_review, p_recall, next_review_days FROM user_hlr_state WHERE user_id = %s",
                (user_id,)
            )
            rows = cur.fetchall()
            return {
                row["topic_id"]: {
                    "half_life": float(row["half_life"]) if row["half_life"] is not None else 1.0,
                    "last_review": str(row["last_review"]) if row["last_review"] is not None else None,
                    "p_recall": float(row["p_recall"]) if row["p_recall"] is not None else 0.5,
                    "next_review_days": float(row["next_review_days"]) if row["next_review_days"] is not None else 1.0
                }
                for row in rows
            }
    finally:
        release_connection(conn)


def save_user_hlr(user_id: str, hlr_state: dict):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for topic_id, state in hlr_state.items():
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
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for topic_id, mastery_score in mastery.items():
                cur.execute("""
                    INSERT INTO user_topic_mastery (user_id, topic_id, mastery_score, updated_at)
                    VALUES (%s, %s, %s, NOW())
                    ON CONFLICT (user_id, topic_id) DO NOTHING
                """, (user_id, topic_id, mastery_score))
        conn.commit()
    finally:
        release_connection(conn)
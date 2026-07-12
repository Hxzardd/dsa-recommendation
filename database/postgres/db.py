import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")


def get_connection():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    return psycopg2.connect(DATABASE_URL)


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
        conn.close()


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
        conn.close()


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
                    state["half_life"], state["last_review"],
                    state["p_recall"], state["next_review_days"]
                ))
        conn.commit()
    finally:
        conn.close()
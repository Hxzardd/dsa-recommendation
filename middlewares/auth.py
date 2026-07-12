from datetime import datetime, timezone
from fastapi import Header, HTTPException
from psycopg2.extras import RealDictCursor

from database.postgres.db import get_connection


def verify_session(authorization: str = Header(None)):
    if authorization is None:
        raise HTTPException(401, "Missing Authorization header")

    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid Authorization header")

    token = authorization.replace("Bearer ", "")

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT user_id, expires_at
                FROM session
                WHERE token = %s
                """,
                (token,),
            )

            session = cur.fetchone()

            if session is None:
                raise HTTPException(401, "Invalid session")

            if session["expires_at"] <= datetime.now(timezone.utc):
                raise HTTPException(401, "Session expired")

            return session["user_id"]

    finally:
        conn.close()
from datetime import datetime, timezone

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from psycopg2.extras import RealDictCursor

from database.postgres.db import get_connection, release_connection


PUBLIC_PATHS = {
    "/",
    "/docs",
    "/openapi.json",
    "/redoc",
}


def verify_session_token(token: str):
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Missing token",
        )

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
                raise HTTPException(
                    status_code=401,
                    detail="Invalid session",
                )

            expires_at = session["expires_at"]

            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)

            if expires_at <= datetime.now(timezone.utc):
                raise HTTPException(
                    status_code=401,
                    detail="Session expired",
                )

            return session["user_id"]

    finally:
        release_connection(conn)


async def auth_middleware(request: Request, call_next):
    if request.url.path in PUBLIC_PATHS:
        return await call_next(request)

    authorization = request.headers.get("Authorization")

    if authorization is None:
        return JSONResponse(
            status_code=401,
            content={"detail": "Missing Authorization header"},
        )

    if not authorization.startswith("Bearer "):
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid Authorization header"},
        )

    token = authorization.split(" ", 1)[1]

    try:
        request.state.user_id = verify_session_token(token)
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"detail": e.detail},
        )

    return await call_next(request)
import hmac
import os
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

# Server-to-server auth for callers with no end-user session -- e.g. the
# backend's judge0 submission webhook calling POST /update directly after a
# judged submission, with no user in the request loop at all. A per-user
# session token doesn't apply there (there's no browser session to hold
# one), and having the webhook forward a user's session token instead would
# couple webhook reliability to whether that specific user happens to have
# a live session, which is fragile for an infra-level call. Set
# ML_SERVICE_TOKEN to a shared secret known only to trusted backend
# services; unset (the default) disables this path entirely -- every
# caller then needs a real per-user session token, same as before.
_ML_SERVICE_TOKEN = os.environ.get("ML_SERVICE_TOKEN")


def _is_service_token(token: str) -> bool:
    if not _ML_SERVICE_TOKEN or not token:
        return False
    # Constant-time compare -- a shared secret is exactly the kind of value
    # a naive `==` timing side-channel could leak over enough requests.
    return hmac.compare_digest(token, _ML_SERVICE_TOKEN)


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

    if _is_service_token(token):
        # Trusted backend service, not an end user -- there's no user_id to
        # resolve from a session (there is no session), so downstream route
        # handlers must check request.state.is_service_call before doing
        # anything that assumes request.state.user_id identifies a real,
        # authenticated end user (e.g. routes/submission.py's
        # auth_user_id == submission.userId ownership check, which doesn't
        # apply here -- the service call acts on whatever userId the
        # request body names, on the backend's own authority).
        request.state.user_id = None
        request.state.is_service_call = True
        return await call_next(request)

    request.state.is_service_call = False
    try:
        request.state.user_id = verify_session_token(token)
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"detail": e.detail},
        )

    return await call_next(request)
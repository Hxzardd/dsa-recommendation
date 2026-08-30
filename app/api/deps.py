"""FastAPI dependencies for the AI analysis service."""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, status

from app.config.settings import get_settings


async def require_service_auth(authorization: str | None = Header(default=None)) -> None:
    """Optional shared-secret bearer auth for backend -> AI calls.

    Enforced only when ``service_api_key`` is configured; otherwise open (dev).
    Uses a constant-time compare — the token is a shared secret.
    """

    expected = get_settings().service_api_key
    if not expected:
        return

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
        )

    token = authorization.split(" ", 1)[1]
    if not hmac.compare_digest(token, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid service token",
        )

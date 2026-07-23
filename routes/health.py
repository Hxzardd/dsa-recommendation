"""
Health endpoints.

/health -- READINESS: actually probes Postgres/Qdrant/Neo4j and returns 503
           when a critical dependency is down, so a platform health check
           (Render, k8s readiness) stops reporting the instance as healthy
           while real requests would fail. Point Render's "Health Check Path"
           here.

/live   -- LIVENESS: process-up ping with no dependency I/O, always 200.
           Point a restart probe here so a transient dependency outage never
           triggers a restart loop.

Both are unauthenticated (added to middlewares/auth.py PUBLIC_PATHS) -- a
platform probe carries no user session token.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from controllers.health_controller import handle_health

router = APIRouter()


@router.get("/health")
def health():
    body, ready = handle_health()
    return JSONResponse(status_code=200 if ready else 503, content=body)


@router.get("/live")
def live():
    return {"status": "alive", "service": "recommendation"}

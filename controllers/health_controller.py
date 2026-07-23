"""
Health / readiness controller.

The root endpoint used to return a static 200 regardless of whether the
service could actually do anything -- so Render's health check stayed green
while Postgres/Qdrant were unreachable and every real request 500'd. This
controller performs an actual readiness check against each dependency so the
health signal reflects whether the service can serve traffic.

Two distinct notions, deliberately kept apart:
  * LIVENESS  -- is the process up and responding at all? (see /live in
                 routes/health.py: always 200, no dependency I/O). Point a
                 platform's *restart* probe here so a transient dependency
                 outage never triggers a pointless restart loop.
  * READINESS -- can the process actually serve requests right now? (this
                 module). Point the platform's *traffic/health* probe here so
                 it stops routing to / reporting green for an instance whose
                 critical dependencies are down.

Critical vs optional:
  * postgres -- CRITICAL. The auth middleware queries it on EVERY
                authenticated request; recommendation reads mastery from it.
                Down => the service cannot serve => readiness fails (503).
  * qdrant   -- CRITICAL. It is the problem catalog; with it down,
                /recommend returns nothing useful. Down => readiness fails.
  * neo4j    -- OPTIONAL. The durable graph tier degrades gracefully to the
                Redis cache + Postgres rebuild (Neo4jGraphStore(driver=None)
                is a documented no-op). Down => reported as "degraded", but
                readiness still passes (200) -- killing a still-serviceable
                instance over an optional cache tier would be wrong.
"""

import logging

import db_env

log = logging.getLogger(__name__)

# Dependencies whose failure means the service genuinely cannot serve traffic
# and readiness must report unhealthy (HTTP 503).
CRITICAL_DEPENDENCIES = ("postgres", "qdrant")


def _check_postgres() -> str:
    """'ok' if a pooled connection can round-trip a trivial query, else 'down'."""
    from database.postgres.db import get_connection, release_connection
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        return "ok"
    except Exception as exc:
        log.warning("Health: Postgres check failed (%s: %s)",
                    exc.__class__.__name__, exc)
        return "down"
    finally:
        if conn is not None:
            try:
                release_connection(conn)
            except Exception:
                pass


def _check_qdrant() -> str:
    """'ok' if the shared Qdrant client can list collections, else 'down'.
    Reuses the same lazily-built client the recommendation path uses, so this
    probes exactly what real requests would hit."""
    from controllers.recommendation_controller import _get_qdrant
    try:
        _get_qdrant().get_collections()
        return "ok"
    except Exception as exc:
        log.warning("Health: Qdrant check failed (%s: %s)",
                    exc.__class__.__name__, exc)
        return "down"


def _check_neo4j() -> str:
    """
    'disabled' when Neo4j isn't configured at all (NEO4J_PASSWORD unset --
    the service is meant to run without it), 'down' when it's configured but
    unreachable, 'ok' when reachable. Reuses the app's own store singleton so
    the report reflects what the service is actually using, not a throwaway
    probe connection.
    """
    if not db_env.NEO4J_PASSWORD:
        return "disabled"
    from controllers.recommendation_controller import _get_neo4j_store
    store = _get_neo4j_store()
    if not store.enabled:
        # Configured, but the driver failed to initialise (e.g. Aura was
        # unreachable at startup) -- the app is running without Neo4j.
        return "down"
    return "ok" if store.ping() else "down"


def handle_health() -> tuple[dict, bool]:
    """
    Run every dependency check and return (body, ready).

    `ready` is False only when a CRITICAL dependency is down -- the route
    layer maps that to HTTP 503 so the platform stops reporting the instance
    as healthy. An optional dependency being down yields status "degraded"
    but ready=True (still 200): the service can serve, just without that
    tier.
    """
    checks = {
        "postgres": _check_postgres(),
        "qdrant": _check_qdrant(),
        "neo4j": _check_neo4j(),
    }

    critical_down = [dep for dep in CRITICAL_DEPENDENCIES if checks[dep] != "ok"]
    ready = not critical_down

    if not ready:
        status = "unhealthy"
    elif any(v == "down" for v in checks.values()):
        status = "degraded"
    else:
        status = "ok"

    body = {
        "status": status,
        "service": "recommendation",
        "dependencies": checks,
    }
    return body, ready

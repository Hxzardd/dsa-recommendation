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
                 critical dependencies are down. render.yaml's
                 healthCheckPath: / lands here.

Critical vs optional:
  * postgres -- CRITICAL. The auth middleware queries it on EVERY
                authenticated request; recommendation reads mastery from it.
  * qdrant   -- CRITICAL. It is the problem catalog; without it /recommend
                returns nothing useful.
  * neo4j    -- OPTIONAL. The durable graph tier degrades gracefully to the
                Redis cache + Postgres rebuild (Neo4jGraphStore(driver=None)
                is a documented no-op). Down => reported as "degraded", but
                readiness still passes (200) -- killing a still-serviceable
                instance over an optional cache tier would be wrong.

CONNECTIVITY IS NOT ENOUGH. A bare `SELECT 1` succeeds against a Postgres
that is missing the `session` table -- every authenticated request would then
401/500 while the probe reported healthy. Likewise `get_collections()`
succeeds against a Qdrant with no `problems_full` collection, or an empty
one, and every /recommend would return an empty slate. So each probe verifies
the specific RESOURCES this service actually reads, not just that a socket
opened.
"""

import logging

import db_env

log = logging.getLogger(__name__)

# Dependencies whose failure means the service genuinely cannot serve traffic
# and readiness must report unhealthy (HTTP 503).
CRITICAL_DEPENDENCIES = ("postgres", "qdrant")

# Tables this service actually reads/writes. Missing any of these means some
# endpoint is broken even though the connection itself is fine:
#   session             -- middlewares/auth.py, EVERY authenticated request
#   user                -- UserGraphService._fetch_user
#   topic               -- topic.id <-> ML-slug translation (all mastery I/O)
#   user_topic_mastery  -- /mastery, /update writes, graph build
#   user_hlr_state      -- /urgency, /update writes
#   problem             -- title_slug -> problem_id resolution
#   recommendation_log  -- /recommend writes, attempt/skip feedback loop
#   submission          -- graph build
#   concept_gap_profile -- graph build (weak concepts)
REQUIRED_TABLES = (
    "session",
    "user",
    "topic",
    "user_topic_mastery",
    "user_hlr_state",
    "problem",
    "recommendation_log",
    "submission",
    "concept_gap_profile",
)


def _check_postgres() -> tuple[str, str]:
    """
    Verifies the connection AND that every table in REQUIRED_TABLES exists.

    to_regclass() returns NULL for a missing relation instead of raising, so
    all tables are checked in a single round trip. Names are passed as
    quoted-qualified identifiers ('public."user"') because `user` is a
    reserved word.
    """
    from database.postgres.db import get_connection, release_connection
    conn = None
    try:
        conn = get_connection()
        qualified = [f'public."{t}"' for t in REQUIRED_TABLES]
        with conn.cursor() as cur:
            cur.execute(
                "SELECT n, to_regclass(n) IS NOT NULL "
                "FROM unnest(%s::text[]) AS n",
                (qualified,),
            )
            rows = cur.fetchall()

        present = {str(name) for name, ok in rows if ok}
        missing = [t for t, q in zip(REQUIRED_TABLES, qualified)
                   if q not in present]
        if missing:
            log.warning("Health: Postgres reachable but missing table(s): %s",
                        ", ".join(missing))
            return "down", f"missing table(s): {', '.join(missing)}"
        return "ok", f"{len(REQUIRED_TABLES)} required tables present"
    except Exception as exc:
        log.warning("Health: Postgres check failed (%s: %s)",
                    exc.__class__.__name__, exc)
        return "down", f"{exc.__class__.__name__}: {exc}"
    finally:
        if conn is not None:
            try:
                release_connection(conn)
            except Exception:
                pass


def _check_qdrant() -> tuple[str, str]:
    """
    Verifies the connection AND that the CONFIGURED collection exists and is
    non-empty. A reachable Qdrant with no `problems_full` collection (or an
    empty one) serves every recommendation request as an empty slate, which
    is a failure the probe must not report as healthy.

    Reuses the same lazily-built client and collection name the recommendation
    path uses, so this probes exactly what real requests would hit.
    """
    from controllers.recommendation_controller import _get_qdrant, COLLECTION
    try:
        client = _get_qdrant()

        try:
            exists = client.collection_exists(COLLECTION)
        except AttributeError:
            # Older qdrant-client without collection_exists().
            exists = COLLECTION in {
                c.name for c in client.get_collections().collections
            }
        if not exists:
            log.warning("Health: Qdrant reachable but collection %r is missing",
                        COLLECTION)
            return "down", f"collection {COLLECTION!r} not found"

        # exact=False -> cheap approximate count, enough to tell "populated"
        # from "empty" without scanning a large collection.
        count = client.count(COLLECTION, exact=False).count
        if not count:
            log.warning("Health: Qdrant collection %r is empty", COLLECTION)
            return "down", f"collection {COLLECTION!r} is empty"

        return "ok", f"collection {COLLECTION!r}, ~{count} points"
    except Exception as exc:
        log.warning("Health: Qdrant check failed (%s: %s)",
                    exc.__class__.__name__, exc)
        return "down", f"{exc.__class__.__name__}: {exc}"


def _check_neo4j() -> tuple[str, str]:
    """
    'disabled' when Neo4j isn't configured at all (NEO4J_PASSWORD unset --
    the service is meant to run without it), 'down' when it's configured but
    unreachable, 'ok' when reachable. Reuses the app's own store singleton so
    the report reflects what the service is actually using, not a throwaway
    probe connection.
    """
    if not db_env.NEO4J_PASSWORD:
        return "disabled", "NEO4J_PASSWORD not set -- durable tier off by config"
    from controllers.recommendation_controller import _get_neo4j_store
    store = _get_neo4j_store()
    if not store.enabled:
        # Configured, but the driver failed to initialise (e.g. Aura was
        # unreachable/paused at startup) -- the app is running without Neo4j.
        return "down", "driver unavailable (unreachable at startup?)"
    if not store.ping():
        return "down", "connectivity check failed"
    return "ok", "reachable"


def handle_health() -> tuple[dict, bool]:
    """
    Run every dependency check and return (body, ready).

    `ready` is False only when a CRITICAL dependency is down -- the route
    layer maps that to HTTP 503 so the platform stops reporting the instance
    as healthy. An optional dependency being down yields status "degraded"
    but ready=True (still 200): the service can serve, just without that
    tier.

    Each dependency reports {"status", "detail"}; the detail names the exact
    missing table/collection so a red health check is directly actionable
    from the Render dashboard instead of just "down".
    """
    results = {
        "postgres": _check_postgres(),
        "qdrant": _check_qdrant(),
        "neo4j": _check_neo4j(),
    }

    critical_down = [dep for dep in CRITICAL_DEPENDENCIES
                     if results[dep][0] != "ok"]
    ready = not critical_down

    if not ready:
        status = "unhealthy"
    elif any(s == "down" for s, _ in results.values()):
        status = "degraded"
    else:
        status = "ok"

    body = {
        "status": status,
        "service": "recommendation",
        "dependencies": {
            name: {"status": s, "detail": d} for name, (s, d) in results.items()
        },
    }
    return body, ready

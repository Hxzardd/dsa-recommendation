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
import threading
import time

import db_env

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Probe throttling
# ---------------------------------------------------------------------------
# / and /health are UNAUTHENTICATED by necessity (a platform probe carries no
# session token), so anyone -- or a misconfigured monitor, or several Render
# probes overlapping -- can drive them at arbitrary rate. Running the full
# Postgres+Qdrant+Neo4j sequence per request would let readiness traffic
# consume the very capacity it is meant to be reporting on: the Postgres pool
# is a SHARED ThreadedConnectionPool with maxconn=10 (database/postgres/db.py),
# so ~10 simultaneous probes could exhaust it and make real requests fail
# while every dependency is actually healthy.
#
# Two mechanisms prevent that:
#   1. TTL cache     -- a result is reused for _CACHE_TTL seconds, so a probe
#      every second costs one real check every _CACHE_TTL seconds.
#   2. Single-flight, NEVER BLOCKING -- only ONE thread ever runs the checks
#      at a time, via a non-blocking lock acquire. Every other caller gets an
#      answer IMMEDIATELY: the last known result if one exists (bounded
#      staleness beats waiting), or an honest "starting" 503 if nothing has
#      ever been cached yet. No caller ever blocks waiting for the lock.
#
#      This matters because these are sync FastAPI handlers: they execute on
#      the SAME shared thread pool authenticated requests and the auth
#      middleware's session lookup also run on. A thread parked waiting on
#      `lock.acquire(blocking=True)` is a worker authenticated traffic can't
#      use -- under a burst of concurrent unauthenticated probes (exactly the
#      case a monitoring tool or several overlapping Render checks would
#      produce), that pile-up degrades real requests even though every
#      dependency is healthy. A blocking acquire here would reproduce the
#      exact class of bug this module exists to prevent, just moved from the
#      Postgres pool to the ASGI thread pool.
#
# Together these cap health-check load at a single Postgres connection and a
# single Qdrant/Neo4j round trip at any instant, no matter the probe rate,
# WITHOUT ever parking a thread-pool worker on a wait.
_CACHE_TTL = 5.0          # seconds; well under any sane probe interval
_cache_lock = threading.Lock()
_cached: tuple | None = None      # (expires_at_monotonic, body, ready)

# Returned (uncached, ready=False -> 503) when a caller arrives while the
# very first check is still in flight and nothing has EVER been cached --
# i.e. we genuinely don't know yet. Honest and fast beats blocking to find
# out: Render already retries a failed probe, and this state clears itself
# within one _CACHE_TTL-scale window regardless.
_STARTING_STATUS = "starting"

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


def _check_postgres() -> tuple[str, str, str]:
    """
    Verifies the connection AND that every table in REQUIRED_TABLES exists.

    to_regclass() returns NULL for a missing relation instead of raising, so
    all tables are checked in a single round trip. Names are passed as
    quoted-qualified identifiers ('public."user"') because `user` is a
    reserved word.

    Returns (status, public_reason, private_detail) -- see _run_checks for
    why the verbose detail never leaves the logs.
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
            return ("down", "missing_tables",
                    f"missing table(s): {', '.join(missing)}")
        return ("ok", "ready", f"{len(REQUIRED_TABLES)} required tables present")
    except Exception as exc:
        return ("down", "unreachable", f"{exc.__class__.__name__}: {exc}")
    finally:
        if conn is not None:
            try:
                release_connection(conn)
            except Exception:
                pass


def _check_qdrant() -> tuple[str, str, str]:
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
            return ("down", "missing_collection",
                    f"collection {COLLECTION!r} not found")

        # exact=False -> cheap approximate count, enough to tell "populated"
        # from "empty" without scanning a large collection.
        count = client.count(COLLECTION, exact=False).count
        if not count:
            return ("down", "empty_collection",
                    f"collection {COLLECTION!r} is empty")

        return ("ok", "ready", f"collection {COLLECTION!r}, ~{count} points")
    except Exception as exc:
        return ("down", "unreachable", f"{exc.__class__.__name__}: {exc}")


def _check_neo4j() -> tuple[str, str, str]:
    """
    'disabled' when Neo4j isn't configured at all (NEO4J_PASSWORD unset --
    the service is meant to run without it), 'down' when it's configured but
    unreachable, 'ok' when reachable. Reuses the app's own store singleton so
    the report reflects what the service is actually using, not a throwaway
    probe connection.
    """
    if not db_env.NEO4J_PASSWORD:
        return ("disabled", "not_configured",
                "NEO4J_PASSWORD not set -- durable tier off by config")
    from controllers.recommendation_controller import _get_neo4j_store
    store = _get_neo4j_store()
    if not store.enabled:
        # Configured, but the driver failed to initialise (e.g. Aura was
        # unreachable/paused at startup) -- the app is running without Neo4j.
        return ("down", "unreachable",
                "driver unavailable (unreachable at startup?)")
    if not store.ping():
        return ("down", "unreachable", "connectivity check failed")
    return ("ok", "ready", "reachable")


def _run_checks() -> tuple[dict, bool]:
    """
    Execute every dependency probe once, uncached. Call handle_health()
    instead unless you specifically want to bypass the throttle -- this one
    opens a real Postgres connection on every call.

    SECURITY: / and /health are unauthenticated (a platform probe carries no
    token), so the RESPONSE carries only a coarse, non-identifying reason code
    per dependency ("missing_tables", "unreachable", ...). The verbose detail
    -- raw driver exception text, which routinely embeds hostnames and
    connection parameters, plus internal table and collection names -- is
    written to the LOGS only, where it is just as actionable for an operator
    without handing an anonymous caller a map of the infrastructure.
    """
    results = {
        "postgres": _check_postgres(),
        "qdrant": _check_qdrant(),
        "neo4j": _check_neo4j(),
    }

    for name, (status_, reason, detail) in results.items():
        if status_ == "down":
            log.warning("Health: %s %s -- %s", name, reason, detail)
        else:
            log.debug("Health: %s %s -- %s", name, reason, detail)

    critical_down = [dep for dep in CRITICAL_DEPENDENCIES
                     if results[dep][0] != "ok"]
    ready = not critical_down

    if not ready:
        status = "unhealthy"
    elif any(s == "down" for s, _, _ in results.values()):
        status = "degraded"
    else:
        status = "ok"

    body = {
        "status": status,
        "service": "recommendation",
        "dependencies": {
            name: {"status": s, "reason": r} for name, (s, r, _) in results.items()
        },
    }
    return body, ready


def handle_health() -> tuple[dict, bool]:
    """
    Return (body, ready) for the readiness endpoints, throttled.

    `ready` is False only when a CRITICAL dependency is down -- the route
    layer maps that to HTTP 503 so the platform stops reporting the instance
    as healthy. An optional dependency being down yields status "degraded"
    but ready=True (still 200): the service can serve, just without that
    tier.

    Each dependency reports {"status", "reason"}; a "starting" status/503
    means a check is already in flight and nothing has been cached yet (only
    possible in the first ~_CACHE_TTL seconds after process start) -- ask
    again shortly rather than reading it as a real dependency failure.

    Results are cached for _CACHE_TTL seconds and refreshed single-flight,
    via a NON-BLOCKING lock acquire only -- see the module header for why a
    caller here must never wait: this runs on the same shared thread pool as
    authenticated request handlers.
    """
    global _cached

    now = time.monotonic()
    cached = _cached
    if cached is not None and now < cached[0]:
        return cached[1], cached[2]

    # Stale or absent. Try to become the single refresher -- but NEVER block
    # waiting for the lock; a parked thread here is a thread-pool worker
    # authenticated traffic can't use.
    if not _cache_lock.acquire(blocking=False):
        # Someone else is already refreshing. Serve the last known result --
        # stale-but-known beats occupying a shared worker thread to wait for
        # a fresher one.
        if cached is not None:
            return cached[1], cached[2]
        # Nothing has EVER been cached (process just started, and another
        # thread already claimed the very first check) -- say so honestly
        # and return immediately rather than guessing or waiting.
        return {
            "status": _STARTING_STATUS,
            "service": "recommendation",
            "dependencies": {},
        }, False

    try:
        # Re-check: another thread may have refreshed between our first
        # cache read above and acquiring the lock.
        cached = _cached
        now = time.monotonic()
        if cached is not None and now < cached[0]:
            return cached[1], cached[2]

        body, ready = _run_checks()
        _cached = (now + _CACHE_TTL, body, ready)
        return body, ready
    finally:
        _cache_lock.release()


def reset_health_cache() -> None:
    """Drop the cached readiness result. For tests, and for any caller that
    needs the next probe to reflect reality immediately."""
    global _cached
    _cached = None

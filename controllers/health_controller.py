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
# Three mechanisms prevent that:
#
#   1. TTL cache -- a fresh result (younger than _CACHE_TTL) is reused as-is,
#      so a probe every second costs one real check every _CACHE_TTL seconds.
#
#   2. Refresh runs on a DEDICATED background thread, never a caller's
#      thread. Every `handle_health()` call -- whether it's the one that
#      kicks off a refresh or one that arrives while a refresh is already
#      running -- returns IMMEDIATELY without waiting for that thread.
#      This is stronger than "don't block on a lock": even the single
#      caller that triggers the refresh must not be the one EXECUTING it,
#      because these are sync FastAPI handlers dispatched to the SAME
#      shared thread pool authenticated requests and the auth middleware's
#      session lookup also run on. If the refresher ran inline on a
#      caller's thread, a stuck Postgres connection attempt would pin that
#      one request-handling worker for however long the stall lasts --
#      smaller than the naive "every caller blocks" bug, but still real
#      shared-pool consumption. Spawning a throwaway daemon thread for the
#      actual I/O removes request-handling capacity from this path
#      entirely: a stuck dependency call pins only that one disposable
#      thread, never anything shared with real traffic.
#
#   3. Bounded staleness -- a cached result is only trusted for
#      _MAX_STALE_AGE seconds past when it was computed, refresh or no
#      refresh. Without this, an indefinitely stuck refresh (e.g. a
#      Postgres connection attempt with no client-side timeout, hanging on
#      a dead TCP peer) would let the LAST GOOD result be served forever --
#      Render would keep routing traffic to an instance whose Postgres has
#      actually been unreachable for the entire stall, because the readiness
#      endpoint kept honestly reporting the last time it checked, not the
#      current truth. Past _MAX_STALE_AGE, readiness stops asserting the old
#      verdict and reports "stale" (not ready) instead -- an instance that
#      hasn't had a completed check in that long is not safely assumed
#      healthy just because nothing has proven otherwise yet.
_CACHE_TTL = 5.0          # seconds; how long a fresh result is reused as-is
_MAX_STALE_AGE = 20.0     # seconds; hard ceiling on trusting a stale result,
                          # even mid-refresh -- see point 3 above

_state_lock = threading.Lock()   # Guards ONLY the small bookkeeping below
                                  # (a few attribute assignments) -- NEVER
                                  # held across dependency I/O, so contention
                                  # here is always sub-microsecond. This is a
                                  # different kind of lock use than the
                                  # earlier (fixed) bug: that one held a lock
                                  # across a potentially 10s+ Postgres call;
                                  # this one never does.
_cached: tuple | None = None         # (computed_at_monotonic, body, ready)
_refreshing = False                  # True while a background probe is running
_refresh_started_at: float | None = None
_generation = 0                      # bumped by reset_health_cache() so a
                                      # refresh already in flight at reset
                                      # time can't clobber post-reset state
                                      # when it eventually completes -- keeps
                                      # tests (which call reset_health_cache()
                                      # between cases while a slow mocked
                                      # check from a prior case may still be
                                      # running on its own daemon thread)
                                      # from leaking results across cases.

_STARTING_STATUS = "starting"   # no result has EVER been cached yet; a
                                 # refresh is in flight -- ask again shortly
_STALLED_STATUS = "stalled"     # like "starting", but the in-flight refresh
                                 # has itself been running past _MAX_STALE_AGE
_STALE_STATUS = "stale"         # a result exists but is older than
                                 # _MAX_STALE_AGE is willing to trust

# Dependencies whose failure means the service genuinely cannot serve traffic
# and readiness must report unhealthy (HTTP 503).
CRITICAL_DEPENDENCIES = ("postgres", "qdrant")

# Tables this service actually reads/writes, derived directly from every raw
# SQL statement in the online serving path (database/postgres/db.py,
# pipeline/recommender/services/user_graph_service.py, middlewares/auth.py,
# controllers/seeding_controller.py) -- not from memory of the schema.
# Missing any of these means some endpoint is broken even though the
# connection itself is fine:
#   session             -- middlewares/auth.py, EVERY authenticated request
#   user                -- UserGraphService._fetch_user, seeding_controller
#   user_xp              -- UserGraphService._fetch_user (LEFT JOIN on every
#                           graph build -- a missing table here doesn't fail
#                           softly like a missing ROW would; the query itself
#                           errors, _fetch_user returns None, and every
#                           subsequent /recommend silently falls back to an
#                           empty cold-start graph, discarding the user's
#                           real submission/mastery history without ever
#                           surfacing an error)
#   topic               -- topic.id <-> ML-slug translation (all mastery I/O)
#   user_topic_mastery  -- /mastery, /update writes, graph build
#   user_hlr_state      -- /urgency, /update writes, seeding_controller
#   problem             -- title_slug -> problem_id resolution
#   recommendation_log  -- /recommend writes, attempt/skip feedback loop
#   submission          -- graph build
#   concept_gap_profile -- graph build (weak concepts)
REQUIRED_TABLES = (
    "session",
    "user",
    "user_xp",
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
    opens a real Postgres connection on every call. This is also the
    function that runs on the dedicated background refresh thread (see
    _refresh_worker) -- it must never be called on a request-handling thread
    once a caller other than the refresh trigger is involved.

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


def _refresh_worker(gen: int) -> None:
    """
    Runs the real dependency sequence on a DEDICATED, throwaway daemon
    thread -- never a FastAPI request-handling thread (see the module header,
    point 2). Started fire-and-forget by handle_health(); no caller ever
    joins it.

    `gen` pins this run to the generation that was current when it started.
    If reset_health_cache() bumps the generation before this finishes (e.g. a
    test resetting state while a slow mocked check from a previous case is
    still in flight), the result is discarded instead of overwriting
    newer/reset state -- this worker's own view of `_refreshing` is stale by
    then anyway.
    """
    global _cached, _refreshing, _refresh_started_at
    try:
        body, ready = _run_checks()
    except Exception as exc:
        # Defense in depth: _run_checks() already catches each dependency's
        # own errors individually, so this should be unreachable in
        # practice. But if it somehow isn't (a bug, a bad patch in a test),
        # an uncaught exception on this background thread must NOT leave
        # `_refreshing` stuck True forever -- that would permanently wedge
        # readiness at "stalled"/"starting" even after the real problem
        # clears, since no future caller would ever see refreshing=False and
        # try again.
        log.error("Health probe crashed unexpectedly: %s", exc, exc_info=True)
        body, ready = {
            "status": "unhealthy", "service": "recommendation",
            "dependencies": {},
        }, False
    with _state_lock:
        if gen == _generation:
            _cached = (time.monotonic(), body, ready)
            _refreshing = False
            _refresh_started_at = None
        # else: superseded by a reset -- discard. Whatever refresh is
        # current now (if any) owns _refreshing/_refresh_started_at; this
        # stale run must not touch them.


def _starting_body(refresh_started_at: float | None, now: float) -> dict:
    stalled = (refresh_started_at is not None
              and (now - refresh_started_at) > _MAX_STALE_AGE)
    return {
        "status": _STALLED_STATUS if stalled else _STARTING_STATUS,
        "service": "recommendation",
        "dependencies": {},
    }


def _stale_body(age: float) -> dict:
    return {
        "status": _STALE_STATUS,
        "service": "recommendation",
        "dependencies": {},
        "stale_for_seconds": round(age, 1),
    }


def handle_health() -> tuple[dict, bool]:
    """
    Return (body, ready) for the readiness endpoints, throttled.

    `ready` is False only when a CRITICAL dependency is down (or the result
    is "starting"/"stalled"/"stale" -- see below) -- the route layer maps
    that to HTTP 503 so the platform stops reporting the instance as healthy.
    An optional dependency being down yields status "degraded" but
    ready=True (still 200): the service can serve, just without that tier.

    Non-dependency statuses (all ready=False):
      "starting" -- nothing has ever been cached; a refresh just started or
                    is already in flight. Only possible in the first
                    ~_CACHE_TTL seconds after process start.
      "stalled"  -- same as "starting", but that in-flight refresh has
                    itself been running longer than _MAX_STALE_AGE (e.g. a
                    hung Postgres connection attempt with no timeout).
      "stale"    -- a real result exists but is older than _MAX_STALE_AGE:
                    the refresh meant to replace it has been stuck for too
                    long to keep trusting the old verdict. This is the fix
                    for the specific failure mode where a stalled refresh
                    would otherwise let a last-known-"ok" result be served
                    forever, keeping Render's traffic routed to an instance
                    whose Postgres has actually been unreachable the whole
                    time.

    Never blocks: the actual dependency sequence always runs on a disposable
    background thread (see _refresh_worker), and every caller -- including
    the one that triggers a new refresh -- returns immediately with either a
    cached/stale/starting verdict, never waiting on that thread.
    """
    global _refreshing, _refresh_started_at

    with _state_lock:
        cached = _cached
        refreshing = _refreshing
        refresh_started_at = _refresh_started_at
        now = time.monotonic()
        fresh = cached is not None and (now - cached[0]) < _CACHE_TTL

        spawn_gen = None
        if not fresh and not refreshing:
            spawn_gen = _generation
            _refreshing = True
            _refresh_started_at = now
            refresh_started_at = now

    if spawn_gen is not None:
        threading.Thread(
            target=_refresh_worker, args=(spawn_gen,),
            daemon=True, name="health-probe-refresh",
        ).start()

    if fresh:
        return cached[1], cached[2]

    if cached is not None:
        age = now - cached[0]
        if age <= _MAX_STALE_AGE:
            # Stale-but-known, within the trust window -- still the best
            # answer we have while a refresh is (or just became) in flight.
            return cached[1], cached[2]
        return _stale_body(age), False

    # No cached result at all -- genuinely unknown yet.
    return _starting_body(refresh_started_at, now), False


def reset_health_cache() -> None:
    """
    Drop the cached readiness result and invalidate any refresh currently in
    flight. For tests, and for any caller that needs the next probe to
    reflect reality immediately.
    """
    global _cached, _refreshing, _refresh_started_at, _generation
    with _state_lock:
        _cached = None
        _refreshing = False
        _refresh_started_at = None
        _generation += 1

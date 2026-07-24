"""
tests/test_health.py

Covers the readiness controller (controllers/health_controller.py): the root
/ endpoint (render.yaml's healthCheckPath) and /health must report unhealthy
-> HTTP 503 when a CRITICAL dependency (Postgres, Qdrant) is unusable,
instead of the old static 200 that let the platform stay green while real
requests failed. An OPTIONAL dependency (Neo4j) being down must NOT fail
readiness -- the service still serves via the Redis/Postgres fallback.

Critically, "usable" means more than "a socket opened":
  * Postgres reachable but missing the `session` table  -> every
    authenticated request fails -> must report down.
  * Qdrant reachable but missing (or with an empty) configured collection
    -> every /recommend returns an empty slate -> must report down.

No real Postgres/Qdrant/Neo4j: every probe is driven by fakes.

Run:
    python -m pytest tests/test_health.py -v
"""

from __future__ import annotations

import contextlib
import threading
import time
import unittest
from unittest.mock import patch

from controllers import health_controller as hc


def _wait_until(predicate, timeout=2.0, interval=0.01):
    """Poll `predicate` until it's true or `timeout` elapses. Needed because
    handle_health() never blocks the calling thread for the real check (see
    controllers/health_controller.py) -- it always runs on a disposable
    background thread, so a test that needs to observe the REAL result has
    to wait for that thread the way any other caller effectively would (by
    polling), not by calling handle_health() once and expecting it inline."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError(f"condition not met within {timeout}s")


def _wait_for_cache(timeout=2.0):
    _wait_until(lambda: hc._cached is not None, timeout)


def _trigger_and_wait():
    """First call kicks off the background refresh (and returns an immediate
    'starting' verdict that this helper discards); once it lands, a second
    call returns the real, now-cached result."""
    hc.handle_health()
    _wait_for_cache()
    return hc.handle_health()


def _drain_health_refresh(timeout=15.0):
    """
    Wait for any in-flight background refresh to finish, then reset.

    MUST be called before a test's `with patch.object(hc, "_check_*", ...)`
    block closes if that test triggered a refresh without already waiting
    for it (via _wait_for_cache/_trigger_and_wait) -- otherwise a daemon
    thread that hasn't yet reached its `_check_postgres()`/`_check_qdrant()`
    call by the time the patch unwinds will go on to call the REAL,
    unpatched function once it does, hitting real infrastructure from a
    stray thread while a LATER, unrelated test is running. Also called from
    setUp/tearDown as defense in depth so no test starts (or leaves) with a
    dangling refresh outstanding.

    Waiting for `_refreshing` to clear works regardless of whether the
    thread ends up running mocked or real checks -- _refresh_worker
    guarantees it eventually goes False either way (including on a crash,
    see its own try/except), so this reliably drains the thread rather than
    just hoping a fixed sleep was long enough.
    """
    deadline = time.monotonic() + timeout
    while hc._refreshing and time.monotonic() < deadline:
        time.sleep(0.02)
    hc.reset_health_cache()


class TestReadinessAggregation(unittest.TestCase):
    """handle_health() maps per-dependency states to an overall verdict."""

    def setUp(self):
        _drain_health_refresh()

    tearDown = setUp

    def _run(self, postgres, qdrant, neo4j):
        with patch.object(hc, "_check_postgres", return_value=(postgres, "r", "d")), \
             patch.object(hc, "_check_qdrant", return_value=(qdrant, "r", "d")), \
             patch.object(hc, "_check_neo4j", return_value=(neo4j, "r", "d")):
            return _trigger_and_wait()

    def _statuses(self, body):
        return {k: v["status"] for k, v in body["dependencies"].items()}

    def test_all_up_is_ready_and_ok(self):
        body, ready = self._run("ok", "ok", "ok")
        self.assertTrue(ready)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(self._statuses(body),
                         {"postgres": "ok", "qdrant": "ok", "neo4j": "ok"})

    def test_postgres_down_is_unhealthy(self):
        body, ready = self._run("down", "ok", "ok")
        self.assertFalse(ready, "Postgres is critical -- must fail readiness")
        self.assertEqual(body["status"], "unhealthy")

    def test_qdrant_down_is_unhealthy(self):
        body, ready = self._run("ok", "down", "ok")
        self.assertFalse(ready, "Qdrant is critical -- must fail readiness")
        self.assertEqual(body["status"], "unhealthy")

    def test_both_critical_down_is_unhealthy(self):
        body, ready = self._run("down", "down", "ok")
        self.assertFalse(ready)
        self.assertEqual(body["status"], "unhealthy")

    def test_neo4j_down_is_degraded_but_still_ready(self):
        body, ready = self._run("ok", "ok", "down")
        self.assertTrue(ready, "Neo4j is optional -- readiness must still pass")
        self.assertEqual(body["status"], "degraded")

    def test_neo4j_disabled_is_ok_and_ready(self):
        body, ready = self._run("ok", "ok", "disabled")
        self.assertTrue(ready)
        self.assertEqual(body["status"], "ok",
                         "Neo4j not configured is a normal, healthy state")

    def test_body_reports_a_coarse_reason_per_dependency(self):
        with patch.object(hc, "_check_postgres",
                          return_value=("down", "missing_tables", "missing table(s): session")), \
             patch.object(hc, "_check_qdrant", return_value=("ok", "ready", "fine")), \
             patch.object(hc, "_check_neo4j", return_value=("ok", "ready", "fine")):
            body, ready = _trigger_and_wait()
        self.assertFalse(ready)
        self.assertEqual(body["dependencies"]["postgres"]["reason"],
                         "missing_tables")

    def test_response_never_leaks_internal_detail_to_anonymous_callers(self):
        """/ and /health are unauthenticated. Raw driver exception text embeds
        hostnames and connection parameters, and table/collection names map
        out the infrastructure -- all of that belongs in the logs, not in a
        body any anonymous caller can read."""
        import json
        secret = "host=prod-db.internal password=hunter2"
        with patch.object(hc, "_check_postgres",
                          return_value=("down", "unreachable", secret)), \
             patch.object(hc, "_check_qdrant",
                          return_value=("down", "missing_collection", secret)), \
             patch.object(hc, "_check_neo4j",
                          return_value=("down", "unreachable", secret)):
            body, _ = _trigger_and_wait()

        serialised = json.dumps(body)
        for leak in ("prod-db.internal", "hunter2", "password"):
            self.assertNotIn(leak, serialised,
                             f"health response leaked {leak!r} to an "
                             f"unauthenticated caller")
        for dep in body["dependencies"].values():
            self.assertNotIn("detail", dep)


class TestProbeThrottling(unittest.TestCase):
    """
    / and /health are unauthenticated, so probe volume is not under our
    control. The dependency sequence must NOT run per-request, and must
    NEVER execute on a caller's own thread -- not even the caller that
    triggers the refresh. It always runs on a disposable background thread
    (see _refresh_worker), so every `handle_health()` call returns
    immediately regardless of whether a check is fresh, in flight, or about
    to be kicked off. The Postgres pool is shared with real traffic and only
    has maxconn=10, so unthrottled/blocking probes could exhaust it and fail
    live requests while every dependency is actually healthy.
    """

    def setUp(self):
        _drain_health_refresh()

    tearDown = setUp

    @contextlib.contextmanager
    def _counting_probes(self, delay=0.0):
        """Counts how many times the real dependency sequence executes."""
        calls = {"n": 0}

        def _pg():
            calls["n"] += 1
            if delay:
                time.sleep(delay)
            return ("ok", "ready", "d")

        with patch.object(hc, "_check_postgres", side_effect=_pg), \
             patch.object(hc, "_check_qdrant", return_value=("ok", "ready", "d")), \
             patch.object(hc, "_check_neo4j", return_value=("ok", "ready", "d")):
            yield calls

    def _wait_until(self, predicate, timeout=2.0, interval=0.01):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(interval)
        self.fail("condition not met within timeout")

    def _wait_for_cache(self, timeout=2.0):
        self._wait_until(lambda: hc._cached is not None, timeout)

    def test_first_call_never_blocks_for_the_real_check(self):
        """
        The exact bug being fixed: not even the caller who TRIGGERS the
        refresh may execute it -- these are sync handlers dispatched to
        FastAPI's shared thread pool, the same pool authenticated requests
        and the auth middleware's session lookup use. The real check always
        runs on a disposable background thread instead, so this call must
        return almost instantly even though the check itself takes 0.3s.
        """
        with self._counting_probes(delay=0.3):
            t0 = time.perf_counter()
            body, ready = hc.handle_health()
            dt = time.perf_counter() - t0
        self.assertLess(dt, 0.1, "first call waited for the background check")
        self.assertFalse(ready)
        self.assertEqual(body["status"], hc._STARTING_STATUS)

    def test_result_becomes_available_once_the_background_check_finishes(self):
        with self._counting_probes() as calls:
            hc.handle_health()          # triggers the background refresh
            self._wait_for_cache()
            body, ready = hc.handle_health()
        self.assertTrue(ready)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(calls["n"], 1)

    def test_repeated_probes_within_ttl_run_the_checks_once(self):
        with self._counting_probes() as calls:
            hc.handle_health()
            self._wait_for_cache()
            for _ in range(25):
                body, ready = hc.handle_health()
                self.assertTrue(ready)
        self.assertEqual(calls["n"], 1,
                         "25 probes within TTL must collapse to ONE real check")

    def test_concurrent_probes_trigger_at_most_one_in_flight_check(self):
        """The pool-exhaustion case: many probes landing at once while
        nothing is cached must still result in a single in-flight sequence,
        and NONE of them may block waiting for it -- every one returns an
        immediate 'starting' verdict, including whichever one happened to be
        the trigger."""
        with self._counting_probes(delay=0.25) as calls:
            results = []
            lock = threading.Lock()

            def hit():
                t0 = time.perf_counter()
                r = hc.handle_health()
                dt = time.perf_counter() - t0
                with lock:
                    results.append((r, dt))

            threads = [threading.Thread(target=hit) for _ in range(15)]
            [t.start() for t in threads]
            [t.join() for t in threads]
            self._wait_for_cache()

        self.assertEqual(len(results), 15)
        self.assertEqual(
            calls["n"], 1,
            f"15 concurrent probes triggered {calls['n']} dependency "
            f"sequences -- each one borrows from the shared 10-connection "
            f"Postgres pool, so this must be 1")
        slow = [dt for _, dt in results if dt >= 0.1]
        self.assertEqual(
            slow, [],
            f"{len(slow)} of 15 callers took >=0.1s -- something is "
            f"blocking on the background check instead of returning "
            f"immediately")
        for (body, ready), _ in results:
            self.assertFalse(ready)
            self.assertEqual(body["status"], hc._STARTING_STATUS)

    def test_slow_check_still_caches_for_a_full_ttl_after_it_finishes(self):
        """
        Regression: expiry must be measured from when the background check
        FINISHES, not when it started. A dependency check can legitimately
        take longer than _CACHE_TTL during a real outage (e.g. Qdrant's own
        client timeout is 10s, well past a 5s default TTL) -- if expiry were
        computed from the pre-check timestamp, the cache entry would be born
        already expired, and the very next caller would immediately trigger
        another refresh instead of being shielded by the cache. That defeats
        the whole point of throttling exactly when a slow/failing dependency
        makes it matter most.
        """
        check_duration = 0.2
        ttl = 0.15   # shorter than the check itself -- the exact failure mode
        with patch.object(hc, "_CACHE_TTL", ttl):
            with self._counting_probes(delay=check_duration) as calls:
                hc.handle_health()
                self._wait_for_cache()
                self.assertEqual(calls["n"], 1)

                # Immediately after the check finishes, the cache must still
                # be honoured (an entry timed from the pre-check clock would
                # already be expired here, since check_duration > ttl).
                _, ready = hc.handle_health()
                self.assertTrue(ready)
                self.assertEqual(
                    calls["n"], 1,
                    "cache entry from a slow check was already expired -- "
                    "expiry is being measured from before the check ran")

    def test_cache_expires_so_a_real_outage_is_noticed(self):
        with self._counting_probes() as calls:
            hc.handle_health()
            self._wait_for_cache()
            self.assertEqual(calls["n"], 1)
            with patch.object(hc, "_CACHE_TTL", 0.0):
                hc.handle_health()   # cache now stale -> triggers a 2nd refresh
                self._wait_until(lambda: calls["n"] >= 2)
        self.assertEqual(calls["n"], 2, "an expired cache must re-probe")

    def test_status_flip_is_picked_up_after_the_cache_expires(self):
        with patch.object(hc, "_check_qdrant", return_value=("ok", "ready", "d")), \
             patch.object(hc, "_check_neo4j", return_value=("ok", "ready", "d")):
            with patch.object(hc, "_check_postgres", return_value=("ok", "ready", "d")):
                hc.handle_health()
                self._wait_for_cache()
                _, ready = hc.handle_health()
                self.assertTrue(ready)
            hc.reset_health_cache()
            with patch.object(hc, "_check_postgres", return_value=("down", "unreachable", "x")):
                hc.handle_health()
                self._wait_for_cache()
                _, ready = hc.handle_health()
                self.assertFalse(ready, "outage must surface once cache clears")


class TestStaleness(unittest.TestCase):
    """
    A stuck refresh (e.g. a Postgres connection attempt with no timeout,
    hanging indefinitely) must NOT let readiness keep serving the last
    known-good verdict forever -- see _MAX_STALE_AGE. Without this, Render
    would keep routing traffic to an instance whose Postgres has actually
    been unreachable for the entire stall, because the endpoint kept
    honestly reporting the last time it successfully checked rather than the
    current truth.
    """

    def setUp(self):
        _drain_health_refresh()

    tearDown = setUp

    def test_result_older_than_max_stale_age_is_reported_stale_not_ok(self):
        """Seeds a cached 'ok' result whose computed_at is already older
        than _MAX_STALE_AGE (simulating a refresh that's been stuck this
        whole time) and confirms handle_health() refuses to keep trusting
        it, even though the cached verdict itself says 'ok'."""
        # _CACHE_TTL must also be shorter than the seeded age, or the
        # freshness check (which runs BEFORE the staleness ceiling) would
        # short-circuit and return the cached value without ever reaching
        # the code path this test exercises.
        with patch.object(hc, "_CACHE_TTL", 0.01), \
             patch.object(hc, "_MAX_STALE_AGE", 0.05):
            hc._cached = (
                time.monotonic() - 1.0,
                {"status": "ok", "service": "recommendation", "dependencies": {}},
                True,
            )
            with patch.object(hc, "_check_postgres", return_value=("ok", "ready", "d")), \
                 patch.object(hc, "_check_qdrant", return_value=("ok", "ready", "d")), \
                 patch.object(hc, "_check_neo4j", return_value=("ok", "ready", "d")):
                body, ready = hc.handle_health()
        self.assertFalse(
            ready,
            "a result older than _MAX_STALE_AGE must not be trusted, even "
            "though its own cached verdict was 'ok'")
        self.assertEqual(body["status"], hc._STALE_STATUS)
        self.assertIn("stale_for_seconds", body)

    def test_result_within_max_stale_age_is_still_trusted(self):
        with patch.object(hc, "_MAX_STALE_AGE", 5.0), \
             patch.object(hc, "_CACHE_TTL", 0.0):
            hc._cached = (
                time.monotonic() - 1.0,
                {"status": "ok", "service": "recommendation", "dependencies": {}},
                True,
            )
            with patch.object(hc, "_check_postgres", return_value=("ok", "ready", "d")), \
                 patch.object(hc, "_check_qdrant", return_value=("ok", "ready", "d")), \
                 patch.object(hc, "_check_neo4j", return_value=("ok", "ready", "d")):
                body, ready = hc.handle_health()
        self.assertTrue(ready)
        self.assertEqual(body["status"], "ok")

    def test_stalled_status_when_first_ever_refresh_has_been_stuck_a_long_time(self):
        """Nothing has ever been cached, and the in-flight refresh has
        itself been running past _MAX_STALE_AGE -- distinct from the normal
        'starting' state so an operator can tell 'just started' apart from
        'something is actually hung'."""
        try:
            with patch.object(hc, "_MAX_STALE_AGE", 0.05):
                hc._refreshing = True
                hc._refresh_started_at = time.monotonic() - 1.0
                body, ready = hc.handle_health()
            self.assertFalse(ready)
            self.assertEqual(body["status"], hc._STALLED_STATUS)
        finally:
            # `_refreshing` was set manually with no real thread behind it
            # (that's the point of this test) -- _drain_health_refresh()'s
            # wait-for-clear in setUp/tearDown would otherwise spin for its
            # full timeout on every subsequent test, since nothing will ever
            # flip this back to False on its own. Clear it directly instead
            # of relying on the generic drain, which is only correct for
            # flags a real background thread is actually going to clear.
            hc.reset_health_cache()

    def test_reset_discards_a_stale_generations_late_arriving_result(self):
        """A slow refresh started BEFORE a reset must not clobber state
        written AFTER it -- the generation guard is what keeps this
        deterministic instead of racing a background thread that's still
        finishing up on its own schedule."""
        release = threading.Event()

        def slow_pg():
            release.wait(timeout=5)
            return ("ok", "ready", "d")

        with patch.object(hc, "_check_postgres", side_effect=slow_pg), \
             patch.object(hc, "_check_qdrant", return_value=("ok", "ready", "d")), \
             patch.object(hc, "_check_neo4j", return_value=("ok", "ready", "d")):
            hc.handle_health()           # spawns the slow background refresh
            time.sleep(0.05)             # let it actually start
            hc.reset_health_cache()      # bumps the generation mid-flight
            release.set()                # let the now-superseded worker finish
            time.sleep(0.15)             # give its (discarded) result a chance to land

        self.assertIsNone(
            hc._cached,
            "a refresh from a superseded generation wrote to the cache "
            "after a reset")

    def test_a_crashing_check_does_not_permanently_wedge_refreshing(self):
        """Defense in depth: if _run_checks() itself somehow raises (each
        individual _check_* already catches its own errors, so this should
        be unreachable in practice), the background thread must still clear
        `_refreshing` -- otherwise every future call would be stuck reporting
        'stalled'/'starting' forever, even after the real problem clears."""
        with patch.object(hc, "_run_checks", side_effect=RuntimeError("boom")):
            hc.handle_health()
            self._wait_until_not_refreshing()

    def _wait_until_not_refreshing(self, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not hc._refreshing:
                return
            time.sleep(0.01)
        self.fail("_refreshing was never cleared after the check crashed")


class TestPostgresProbe(unittest.TestCase):
    """
    _check_postgres verifies REQUIRED_TABLES exist, not just connectivity,
    and always returns the pooled connection.
    """

    def test_user_xp_table_is_required(self):
        """
        UserGraphService._fetch_user LEFT JOINs user_xp on EVERY graph
        build. A missing table (unlike a missing matching ROW, which LEFT
        JOIN tolerates fine) makes the query itself error -- _fetch_user
        catches it and returns None, _build raises ValueError, and
        get_recommendations silently falls back to a brand-new cold-start
        graph, discarding the user's real submission/mastery history without
        ever surfacing an error. Readiness must catch this before it reaches
        a real user's request.
        """
        self.assertIn("user_xp", hc.REQUIRED_TABLES)

    def _run_with(self, conn, released):
        with patch("database.postgres.db.get_connection", return_value=conn), \
             patch("database.postgres.db.release_connection",
                   side_effect=lambda c: released.setdefault("conn", c)):
            return hc._check_postgres()

    def test_ok_when_all_required_tables_present(self):
        released = {}
        conn = _FakeConn(present=set(hc.REQUIRED_TABLES))
        status, reason, detail = self._run_with(conn, released)
        self.assertEqual(status, "ok")
        self.assertIs(released.get("conn"), conn,
                      "connection must be returned to the pool")

    def test_down_when_session_table_is_missing(self):
        """The exact gap a bare SELECT 1 misses: auth would break on every
        authenticated request while the probe reported healthy."""
        released = {}
        present = set(hc.REQUIRED_TABLES) - {"session"}
        status, reason, detail = self._run_with(_FakeConn(present=present), released)
        self.assertEqual(status, "down")
        self.assertIn("session", detail)

    def test_down_lists_every_missing_table(self):
        released = {}
        present = set(hc.REQUIRED_TABLES) - {"topic", "user_topic_mastery"}
        status, reason, detail = self._run_with(_FakeConn(present=present), released)
        self.assertEqual(status, "down")
        self.assertIn("topic", detail)
        self.assertIn("user_topic_mastery", detail)

    def test_down_when_query_raises_and_connection_still_released(self):
        released = {}
        conn = _FakeConn(raise_on_execute=True)
        status, reason, detail = self._run_with(conn, released)
        self.assertEqual(status, "down")
        self.assertIs(released.get("conn"), conn,
                      "a failed probe must not leak the pooled connection")

    def test_down_when_get_connection_raises(self):
        with patch("database.postgres.db.get_connection",
                   side_effect=RuntimeError("pool exhausted")):
            status, reason, detail = hc._check_postgres()
        self.assertEqual(status, "down")
        self.assertIn("pool exhausted", detail)


class TestQdrantProbe(unittest.TestCase):
    """_check_qdrant verifies the CONFIGURED collection exists and is populated."""

    def _run(self, client):
        with patch("controllers.recommendation_controller._get_qdrant",
                   return_value=client), \
             patch("controllers.recommendation_controller.COLLECTION",
                   "problems_full"):
            return hc._check_qdrant()

    def test_ok_when_collection_exists_and_is_populated(self):
        status, reason, detail = self._run(_FakeQdrant(exists=True, count=1158))
        self.assertEqual(status, "ok")
        self.assertIn("problems_full", detail)

    def test_down_when_configured_collection_is_missing(self):
        """Reachable Qdrant, wrong/absent collection -> every /recommend
        returns an empty slate, so readiness must fail."""
        status, reason, detail = self._run(_FakeQdrant(exists=False))
        self.assertEqual(status, "down")
        self.assertIn("problems_full", detail)
        self.assertIn("not found", detail)

    def test_down_when_collection_is_empty(self):
        status, reason, detail = self._run(_FakeQdrant(exists=True, count=0))
        self.assertEqual(status, "down")
        self.assertIn("empty", detail)

    def test_down_when_client_raises(self):
        status, reason, detail = self._run(_FakeQdrant(raises=True))
        self.assertEqual(status, "down")
        self.assertIn("ConnectionError", detail)

    def test_falls_back_to_get_collections_on_older_client(self):
        """Clients without collection_exists() still get a correct verdict."""
        status, reason, detail = self._run(_LegacyQdrant(names=["problems_full"], count=42))
        self.assertEqual(status, "ok")
        status, reason, detail = self._run(_LegacyQdrant(names=["something_else"]))
        self.assertEqual(status, "down")
        self.assertIn("not found", detail)


class TestNeo4jProbe(unittest.TestCase):
    def test_disabled_when_no_password_configured(self):
        with patch.object(hc.db_env, "NEO4J_PASSWORD", None):
            status, reason, detail = hc._check_neo4j()
        self.assertEqual(status, "disabled")

    def test_down_when_configured_but_store_not_enabled(self):
        with patch.object(hc.db_env, "NEO4J_PASSWORD", "secret"), \
             patch("controllers.recommendation_controller._get_neo4j_store",
                   return_value=_FakeStore(enabled=False)):
            status, reason, detail = hc._check_neo4j()
        self.assertEqual(status, "down")

    def test_ok_when_store_pings(self):
        with patch.object(hc.db_env, "NEO4J_PASSWORD", "secret"), \
             patch("controllers.recommendation_controller._get_neo4j_store",
                   return_value=_FakeStore(enabled=True, ping=True)):
            status, reason, detail = hc._check_neo4j()
        self.assertEqual(status, "ok")

    def test_down_when_store_ping_fails(self):
        with patch.object(hc.db_env, "NEO4J_PASSWORD", "secret"), \
             patch("controllers.recommendation_controller._get_neo4j_store",
                   return_value=_FakeStore(enabled=True, ping=False)):
            status, reason, detail = hc._check_neo4j()
        self.assertEqual(status, "down")


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------

class _FakeCursor:
    def __init__(self, present, raise_on_execute):
        self._present = present
        self._raise = raise_on_execute
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        if self._raise:
            raise RuntimeError("connection reset")
        # Mirrors `SELECT n, to_regclass(n) IS NOT NULL FROM unnest(...)`
        qualified = params[0]
        self._rows = [
            (q, q.split('"')[1] in self._present) for q in qualified
        ]

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, present=None, raise_on_execute=False):
        self._present = present or set()
        self._raise = raise_on_execute

    def cursor(self, *a, **k):
        return _FakeCursor(self._present, self._raise)


class _Count:
    def __init__(self, count):
        self.count = count


class _FakeQdrant:
    def __init__(self, exists=True, count=1, raises=False):
        self._exists = exists
        self._count = count
        self._raises = raises

    def collection_exists(self, name):
        if self._raises:
            raise ConnectionError("qdrant unreachable")
        return self._exists

    def count(self, name, exact=False):
        return _Count(self._count)


class _LegacyQdrant:
    """No collection_exists() -- exercises the get_collections() fallback."""

    def __init__(self, names, count=1):
        self._names = names
        self._count = count

    def get_collections(self):
        class _C:
            collections = [type("N", (), {"name": n})() for n in self._names]
        return _C()

    def count(self, name, exact=False):
        return _Count(self._count)


class _FakeStore:
    def __init__(self, enabled=True, ping=True):
        self.enabled = enabled
        self._ping = ping

    def ping(self):
        return self._ping


if __name__ == "__main__":
    unittest.main(verbosity=2)

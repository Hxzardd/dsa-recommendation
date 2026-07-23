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


class TestReadinessAggregation(unittest.TestCase):
    """handle_health() maps per-dependency states to an overall verdict."""

    def setUp(self):
        hc.reset_health_cache()

    tearDown = setUp

    def _run(self, postgres, qdrant, neo4j):
        with patch.object(hc, "_check_postgres", return_value=(postgres, "r", "d")), \
             patch.object(hc, "_check_qdrant", return_value=(qdrant, "r", "d")), \
             patch.object(hc, "_check_neo4j", return_value=(neo4j, "r", "d")):
            return hc.handle_health()

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
            body, ready = hc.handle_health()
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
            body, _ = hc.handle_health()

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
    control. The dependency sequence must NOT run per-request: the Postgres
    pool is shared with real traffic and only has maxconn=10, so unthrottled
    probes could exhaust it and fail live requests while every dependency is
    actually healthy.
    """

    def setUp(self):
        hc.reset_health_cache()

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

    def test_repeated_probes_within_ttl_run_the_checks_once(self):
        with self._counting_probes() as calls:
            for _ in range(25):
                body, ready = hc.handle_health()
                self.assertTrue(ready)
        self.assertEqual(calls["n"], 1,
                         "25 sequential probes must collapse to ONE real check")

    def test_concurrent_probes_do_not_stack_up_on_the_pool(self):
        """The pool-exhaustion case: many probes landing at once must still
        result in a single in-flight dependency sequence."""
        with self._counting_probes(delay=0.25) as calls:
            results = []
            threads = [threading.Thread(
                target=lambda: results.append(hc.handle_health()[1]))
                for _ in range(15)]
            [t.start() for t in threads]
            [t.join() for t in threads]

        self.assertEqual(len(results), 15)
        self.assertTrue(all(results), "every probe should get a verdict")
        self.assertEqual(
            calls["n"], 1,
            f"15 concurrent probes triggered {calls['n']} dependency "
            f"sequences -- each one borrows from the shared 10-connection "
            f"Postgres pool, so this must be 1")

    def test_cache_expires_so_a_real_outage_is_noticed(self):
        with self._counting_probes() as calls:
            hc.handle_health()
            self.assertEqual(calls["n"], 1)
            with patch.object(hc, "_CACHE_TTL", 0.0):
                hc.reset_health_cache()
                hc.handle_health()
        self.assertEqual(calls["n"], 2, "an expired cache must re-probe")

    def test_status_flip_is_picked_up_after_the_cache_expires(self):
        with patch.object(hc, "_check_qdrant", return_value=("ok", "ready", "d")), \
             patch.object(hc, "_check_neo4j", return_value=("ok", "ready", "d")):
            with patch.object(hc, "_check_postgres", return_value=("ok", "ready", "d")):
                _, ready = hc.handle_health()
                self.assertTrue(ready)
            hc.reset_health_cache()
            with patch.object(hc, "_check_postgres", return_value=("down", "unreachable", "x")):
                _, ready = hc.handle_health()
                self.assertFalse(ready, "outage must surface once cache clears")


class TestPostgresProbe(unittest.TestCase):
    """
    _check_postgres verifies REQUIRED_TABLES exist, not just connectivity,
    and always returns the pooled connection.
    """

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

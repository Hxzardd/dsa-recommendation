"""
tests/test_health.py

Covers the readiness controller (controllers/health_controller.py): the root
/ endpoint (and /health) must report unhealthy -> HTTP 503 when a CRITICAL
dependency (Postgres, Qdrant) is down, instead of the old static 200 that let
a platform health check stay green while real requests failed. An OPTIONAL
dependency (Neo4j) being down must NOT fail readiness -- the service still
serves via the Redis/Postgres fallback.

No real Postgres/Qdrant/Neo4j: the three per-dependency probes are monkey-
patched to simulate each up/down/disabled combination.

Run:
    python -m pytest tests/test_health.py -v
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from controllers import health_controller


class TestReadinessAggregation(unittest.TestCase):
    """handle_health() maps per-dependency states to an overall verdict."""

    def _run(self, postgres, qdrant, neo4j):
        with patch.object(health_controller, "_check_postgres", return_value=postgres), \
             patch.object(health_controller, "_check_qdrant", return_value=qdrant), \
             patch.object(health_controller, "_check_neo4j", return_value=neo4j):
            return health_controller.handle_health()

    def test_all_up_is_ready_and_ok(self):
        body, ready = self._run("ok", "ok", "ok")
        self.assertTrue(ready)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["dependencies"],
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


class TestPostgresProbe(unittest.TestCase):
    """_check_postgres round-trips a trivial query and always releases the conn."""

    def test_ok_when_query_succeeds_and_connection_released(self):
        released = {}
        fake_conn = _FakeConn()
        with patch("database.postgres.db.get_connection", return_value=fake_conn), \
             patch("database.postgres.db.release_connection",
                   side_effect=lambda c: released.setdefault("conn", c)):
            self.assertEqual(health_controller._check_postgres(), "ok")
        self.assertIs(released.get("conn"), fake_conn,
                      "connection must be returned to the pool")

    def test_down_when_query_raises_and_connection_still_released(self):
        released = {}
        fake_conn = _FakeConn(raise_on_execute=True)
        with patch("database.postgres.db.get_connection", return_value=fake_conn), \
             patch("database.postgres.db.release_connection",
                   side_effect=lambda c: released.setdefault("conn", c)):
            self.assertEqual(health_controller._check_postgres(), "down")
        self.assertIs(released.get("conn"), fake_conn,
                      "a failed probe must not leak the pooled connection")

    def test_down_when_get_connection_raises(self):
        with patch("database.postgres.db.get_connection",
                   side_effect=RuntimeError("pool exhausted")):
            self.assertEqual(health_controller._check_postgres(), "down")


class TestQdrantProbe(unittest.TestCase):
    def test_ok_when_get_collections_succeeds(self):
        class _Client:
            def get_collections(self):
                return object()
        with patch("controllers.recommendation_controller._get_qdrant",
                   return_value=_Client()):
            self.assertEqual(health_controller._check_qdrant(), "ok")

    def test_down_when_get_collections_raises(self):
        class _Client:
            def get_collections(self):
                raise ConnectionError("qdrant unreachable")
        with patch("controllers.recommendation_controller._get_qdrant",
                   return_value=_Client()):
            self.assertEqual(health_controller._check_qdrant(), "down")


class TestNeo4jProbe(unittest.TestCase):
    def test_disabled_when_no_password_configured(self):
        with patch.object(health_controller.db_env, "NEO4J_PASSWORD", None):
            self.assertEqual(health_controller._check_neo4j(), "disabled")

    def test_down_when_configured_but_store_not_enabled(self):
        store = _FakeStore(enabled=False)
        with patch.object(health_controller.db_env, "NEO4J_PASSWORD", "secret"), \
             patch("controllers.recommendation_controller._get_neo4j_store",
                   return_value=store):
            self.assertEqual(health_controller._check_neo4j(), "down")

    def test_ok_when_store_pings(self):
        store = _FakeStore(enabled=True, ping=True)
        with patch.object(health_controller.db_env, "NEO4J_PASSWORD", "secret"), \
             patch("controllers.recommendation_controller._get_neo4j_store",
                   return_value=store):
            self.assertEqual(health_controller._check_neo4j(), "ok")

    def test_down_when_store_ping_fails(self):
        store = _FakeStore(enabled=True, ping=False)
        with patch.object(health_controller.db_env, "NEO4J_PASSWORD", "secret"), \
             patch("controllers.recommendation_controller._get_neo4j_store",
                   return_value=store):
            self.assertEqual(health_controller._check_neo4j(), "down")


# --------------------------------------------------------------------------
# Minimal fakes
# --------------------------------------------------------------------------

class _FakeCursor:
    def __init__(self, raise_on_execute=False):
        self._raise = raise_on_execute
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def execute(self, *a, **k):
        if self._raise:
            raise RuntimeError("connection reset")
    def fetchone(self):
        return (1,)


class _FakeConn:
    def __init__(self, raise_on_execute=False):
        self._raise = raise_on_execute
    def cursor(self, *a, **k):
        return _FakeCursor(self._raise)


class _FakeStore:
    def __init__(self, enabled=True, ping=True):
        self.enabled = enabled
        self._ping = ping
    def ping(self):
        return self._ping


if __name__ == "__main__":
    unittest.main(verbosity=2)

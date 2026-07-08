"""
tests/test_neo4j_graph_store.py

Tests Neo4jGraphStore's save/load/delete round trip, and UserGraphService's
three-tier read path (Redis -> Neo4j -> Postgres rebuild).

Uses a fake Neo4j driver matching the real `neo4j` package's
driver.session()/session.execute_write()/execute_read()/tx.run() interface
-- no real Neo4j instance needed to verify the logic.

Run:
    python -m pytest tests/test_neo4j_graph_store.py -v
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from pipeline.recommender.models.user_graph import (
    UserGraph, UserNode, ProblemEdge, ConceptEdge, ConceptConceptEdge, EdgeType,
)
from pipeline.recommender.services.neo4j_graph_store import Neo4jGraphStore
from pipeline.recommender.services.user_graph_service import UserGraphService


class _FakeRecord(dict):
    pass


class _FakeNode(dict):
    pass


class FakeGraphDB:
    def __init__(self):
        self.users = {}
        self.problem_edges = {}
        self.concept_edges = {}
        self.cc_edges = {}

    def delete_user(self, user_id):
        self.users.pop(user_id, None)
        for key in [k for k in self.problem_edges if k[0] == user_id]:
            del self.problem_edges[key]
        for key in [k for k in self.concept_edges if k[0] == user_id]:
            del self.concept_edges[key]


class FakeTx:
    def __init__(self, db: FakeGraphDB):
        self.db = db

    def run(self, query, **params):
        q = " ".join(query.split())

        if "MERGE (u:User {user_id: $user_id})" in q and "SET u.username" in q:
            self.db.users[params["user_id"]] = dict(params)
            return _FakeResult([])

        if "MERGE (u)-[e:PROBLEM_EDGE]->(p)" in q:
            key = (params["user_id"], params["problem_id"])
            self.db.problem_edges[key] = dict(params)
            return _FakeResult([])

        if "MERGE (u)-[e:CONCEPT_EDGE]->(c)" in q:
            key = (params["user_id"], params["slug"])
            self.db.concept_edges[key] = dict(params)
            return _FakeResult([])

        if "MERGE (a)-[e:CC_EDGE" in q:
            key = (params["src"], params["tgt"], params["edge_type"])
            self.db.cc_edges[key] = dict(params)
            return _FakeResult([])

        if "MATCH (u:User {user_id: $user_id}) RETURN u LIMIT 1" in q:
            u = self.db.users.get(params["user_id"])
            if u is None:
                return _FakeResult([])
            return _FakeResult([_FakeRecord(u=_FakeNode(u))])

        if "MATCH (u:User {user_id: $user_id})-[e:PROBLEM_EDGE]->(p:Problem)" in q:
            recs = []
            for (uid, pid), e in self.db.problem_edges.items():
                if uid == params["user_id"]:
                    recs.append(_FakeRecord(pid=pid, e=_FakeNode(e)))
            return _FakeResult(recs)

        if "MATCH (u:User {user_id: $user_id})-[e:CONCEPT_EDGE]->(c:Concept)" in q:
            recs = []
            for (uid, slug), e in self.db.concept_edges.items():
                if uid == params["user_id"]:
                    recs.append(_FakeRecord(slug=slug, e=_FakeNode(e)))
            return _FakeResult(recs)

        if "MATCH (a:Concept)-[e:CC_EDGE]->(b:Concept)" in q:
            recs = []
            for (src, tgt, etype), e in self.db.cc_edges.items():
                recs.append(_FakeRecord(src=src, tgt=tgt, e=_FakeNode(e)))
            return _FakeResult(recs)

        if "DETACH DELETE u" in q:
            self.db.delete_user(params["user_id"])
            return _FakeResult([])

        raise AssertionError(f"FakeTx doesn't know how to handle query: {q[:80]}")


class _FakeResult:
    def __init__(self, records):
        self._records = records

    def single(self):
        return self._records[0] if self._records else None

    def __iter__(self):
        return iter(self._records)


class FakeSession:
    def __init__(self, db: FakeGraphDB):
        self.db = db

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute_write(self, fn, *args):
        return fn(FakeTx(self.db), *args)

    def execute_read(self, fn, *args):
        return fn(FakeTx(self.db), *args)


class FakeDriver:
    def __init__(self):
        self.db = FakeGraphDB()

    def session(self, database=None):
        return FakeSession(self.db)


def _sample_graph(user_id="u1"):
    g = UserGraph(user=UserNode(
        user_id=user_id, username="tester", elo_rating=1250,
        total_xp=500, current_level=3,
    ))
    g.add_problem_edge(ProblemEdge(
        problem_id="p1", edge_type=EdgeType.SOLVED,
        normalised_score=0.9, timestamp=1234.0,
    ))
    g.add_concept_edge(ConceptEdge(
        concept_slug="arrays", edge_type=EdgeType.MASTERED,
        mastery_score=0.8, confidence=0.9, urgency=0.1, half_life=10.0,
    ))
    g.add_cc_edge(ConceptConceptEdge(
        source_slug="arrays", target_slug="dp",
        edge_type=EdgeType.PREREQ, weight=1.0,
    ))
    g.solved_ids.add("p1")
    return g


class TestNeo4jGraphStoreDisabled(unittest.TestCase):

    def test_enabled_false_when_no_driver(self):
        store = Neo4jGraphStore(driver=None)
        self.assertFalse(store.enabled)

    def test_save_noop_when_disabled(self):
        store = Neo4jGraphStore(driver=None)
        store.save(_sample_graph())

    def test_load_returns_none_when_disabled(self):
        store = Neo4jGraphStore(driver=None)
        self.assertIsNone(store.load("u1"))

    def test_delete_noop_when_disabled(self):
        store = Neo4jGraphStore(driver=None)
        store.delete("u1")


class TestNeo4jGraphStoreRoundTrip(unittest.TestCase):

    def setUp(self):
        self.driver = FakeDriver()
        self.store = Neo4jGraphStore(driver=self.driver)

    def test_enabled_true_with_driver(self):
        self.assertTrue(self.store.enabled)

    def test_load_returns_none_before_any_save(self):
        self.assertIsNone(self.store.load("never_saved"))

    def test_save_then_load_preserves_user_node(self):
        g = _sample_graph()
        self.store.save(g)
        loaded = self.store.load("u1")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.user.user_id, "u1")
        self.assertEqual(loaded.user.username, "tester")
        self.assertEqual(loaded.user.elo_rating, 1250)
        self.assertEqual(loaded.user.total_xp, 500)

    def test_save_then_load_preserves_problem_edges(self):
        g = _sample_graph()
        self.store.save(g)
        loaded = self.store.load("u1")
        self.assertIn("p1", loaded.problem_edges)
        self.assertEqual(loaded.problem_edges["p1"].edge_type, EdgeType.SOLVED)
        self.assertAlmostEqual(loaded.problem_edges["p1"].normalised_score, 0.9)

    def test_save_then_load_preserves_concept_edges(self):
        g = _sample_graph()
        self.store.save(g)
        loaded = self.store.load("u1")
        self.assertIn("arrays", loaded.concept_edges)
        self.assertAlmostEqual(loaded.concept_edges["arrays"].mastery_score, 0.8)
        self.assertAlmostEqual(loaded.concept_edges["arrays"].confidence, 0.9)

    def test_save_then_load_preserves_cc_edges(self):
        g = _sample_graph()
        self.store.save(g)
        loaded = self.store.load("u1")
        self.assertIn("arrays", loaded.cc_edges)
        self.assertEqual(loaded.cc_edges["arrays"][0].target_slug, "dp")

    def test_save_then_load_preserves_solved_ids(self):
        g = _sample_graph()
        self.store.save(g)
        loaded = self.store.load("u1")
        self.assertIn("p1", loaded.solved_ids)

    def test_lock_check_works_after_load(self):
        g = UserGraph(user=UserNode(user_id="u2"))
        g.add_concept_edge(ConceptEdge(
            "arrays", EdgeType.LEARNING, mastery_score=0.3,
        ))
        g.add_cc_edge(ConceptConceptEdge("arrays", "dp", EdgeType.PREREQ, 1.0))
        self.store.save(g)
        loaded = self.store.load("u2")
        self.assertTrue(loaded.is_locked(["dp"]))

    def test_resave_updates_rather_than_duplicates(self):
        g = _sample_graph()
        self.store.save(g)
        g.concept_edges["arrays"].mastery_score = 0.95
        self.store.save(g)
        loaded = self.store.load("u1")
        self.assertAlmostEqual(loaded.concept_edges["arrays"].mastery_score, 0.95)

    def test_delete_removes_user(self):
        g = _sample_graph()
        self.store.save(g)
        self.assertIsNotNone(self.store.load("u1"))
        self.store.delete("u1")
        self.assertIsNone(self.store.load("u1"))

    def test_different_users_independent(self):
        g1 = _sample_graph("u1")
        g2 = _sample_graph("u2")
        g2.concept_edges["arrays"].mastery_score = 0.2
        self.store.save(g1)
        self.store.save(g2)
        loaded1 = self.store.load("u1")
        loaded2 = self.store.load("u2")
        self.assertAlmostEqual(loaded1.concept_edges["arrays"].mastery_score, 0.8)
        self.assertAlmostEqual(loaded2.concept_edges["arrays"].mastery_score, 0.2)


class TestNeo4jSaveFailureIsGraceful(unittest.TestCase):

    def test_save_exception_does_not_raise(self):
        broken_driver = MagicMock()
        broken_driver.session.side_effect = Exception("connection refused")
        store = Neo4jGraphStore(driver=broken_driver)
        store.save(_sample_graph())

    def test_load_exception_returns_none(self):
        broken_driver = MagicMock()
        broken_driver.session.side_effect = Exception("connection refused")
        store = Neo4jGraphStore(driver=broken_driver)
        self.assertIsNone(store.load("u1"))


class FakeRedis:
    def __init__(self):
        self._store = {}
    def get(self, key):
        return self._store.get(key)
    def setex(self, key, ttl, value):
        self._store[key] = value
    def delete(self, key):
        self._store.pop(key, None)


class TestThreeTierReadPath(unittest.TestCase):

    def setUp(self):
        self.redis = FakeRedis()
        self.neo4j = Neo4jGraphStore(driver=FakeDriver())

    def test_new_user_graph_writes_to_neo4j(self):
        svc = UserGraphService(db=None, redis=self.redis, neo4j=self.neo4j)
        svc.new_user_graph("u1")
        self.assertIsNotNone(self.neo4j.load("u1"))

    def test_get_hits_neo4j_when_redis_empty(self):
        svc = UserGraphService(db=None, redis=self.redis, neo4j=self.neo4j)
        svc.new_user_graph("u1")
        self.redis.delete("user_graph:u1")
        graph = svc.get("u1")
        self.assertEqual(graph.user.user_id, "u1")

    def test_get_refreshes_redis_after_neo4j_hit(self):
        svc = UserGraphService(db=None, redis=self.redis, neo4j=self.neo4j)
        svc.new_user_graph("u1")
        self.redis.delete("user_graph:u1")
        svc.get("u1")
        self.assertIsNotNone(self.redis.get("user_graph:u1"))

    def test_delete_user_removes_from_both_tiers(self):
        svc = UserGraphService(db=None, redis=self.redis, neo4j=self.neo4j)
        svc.new_user_graph("u1")
        svc.delete_user("u1")
        self.assertIsNone(self.redis.get("user_graph:u1"))
        self.assertIsNone(self.neo4j.load("u1"))

    def test_invalidate_does_not_delete_from_neo4j(self):
        svc = UserGraphService(db=None, redis=self.redis, neo4j=self.neo4j)
        svc.new_user_graph("u1")
        svc.invalidate("u1")
        self.assertIsNone(self.redis.get("user_graph:u1"))
        self.assertIsNotNone(self.neo4j.load("u1"))

    def test_none_neo4j_does_not_break_anything(self):
        svc = UserGraphService(db=None, redis=self.redis, neo4j=None)
        graph = svc.new_user_graph("u1")
        self.assertEqual(graph.user.user_id, "u1")
        fetched = svc.get("u1")
        self.assertEqual(fetched.user.user_id, "u1")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestStateUpdatePersistsToNeo4j(unittest.TestCase):
    """
    Regression test for a real Greptile-flagged bug: StateUpdateService.
    process_submission() used to write the mutated graph to Redis only
    (via _to_cache), never to Neo4j. Once the 5-minute Redis entry
    expired, the next get() fell through to Neo4j and returned the STALE
    pre-submission graph -- silently dropping the just-solved problem and
    mastery updates. Fixed via UserGraphService.persist(), which writes
    to both tiers.
    """

    def test_submission_survives_redis_expiry_via_neo4j(self):
        import time
        from pipeline.recommender.services.state_update_service import StateUpdateService
        import pipeline.recommender.bkt as bkt_module
        import pipeline.recommender.hlr as hlr_module

        redis = FakeRedis()
        neo4j = Neo4jGraphStore(driver=FakeDriver())
        graph_service = UserGraphService(db=None, redis=redis, neo4j=neo4j)
        state_service = StateUpdateService(graph_service, qdrant=None, bkt_store={}, hlr_store={})

        original_bkt_map = dict(bkt_module.problem_to_topics)
        original_hlr_map = dict(hlr_module.problem_to_topics)
        bkt_module.problem_to_topics = {"arrays_0": ["arrays"]}
        hlr_module.problem_to_topics = {"arrays_0": ["arrays"]}
        try:
            submission = {
                "problemId": "arrays_0", "verdict": "OK", "hintsUsed": 0,
                "submissionCount": 1, "normalisedScore": 0.9,
                "testCasesPassed": 10, "totalTestCases": 10,
                "timestamp": time.time(),
            }
            state_service.process_submission("u1", submission, rebuild_vector=False)

            # simulate the Redis TTL expiring
            redis.delete("user_graph:u1")

            # the next read must fall through to Neo4j and find the FRESH
            # post-submission data, not a stale pre-submission graph
            fresh = graph_service.get("u1")
            self.assertIn("arrays_0", fresh.problem_edges)
        finally:
            bkt_module.problem_to_topics = original_bkt_map
            hlr_module.problem_to_topics = original_hlr_map

    def test_persist_writes_to_both_tiers(self):
        redis = FakeRedis()
        neo4j = Neo4jGraphStore(driver=FakeDriver())
        svc = UserGraphService(db=None, redis=redis, neo4j=neo4j)

        g = _sample_graph("u2")
        svc.persist("u2", g)

        self.assertIsNotNone(redis.get("user_graph:u2"))
        self.assertIsNotNone(neo4j.load("u2"))
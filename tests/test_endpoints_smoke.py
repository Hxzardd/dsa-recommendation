"""
tests/test_endpoints_smoke.py

End-to-end verification that EVERY route the backend calls actually works,
driven through the real FastAPI app (routes -> auth middleware -> controller
-> pipeline). Only external systems (Postgres, Qdrant, Neo4j) are faked.

Endpoints covered:
    GET  /                          readiness (render.yaml healthCheckPath)
    GET  /health                    readiness
    GET  /live                      liveness
    GET  /recommend/{user_id}       full 7-pool recommendation
    GET  /topic/recommend/{user_id} single best-next-topic
    POST /topic/recommend/problems  problems for a caller-chosen topic
    GET  /mastery/{user_id}         BKT mastery + decayed proficiency
    GET  /urgency/{user_id}         HLR urgency
    POST /update                    BKT/HLR recompute + persist + graph sync
    POST /seed_bkt/{user_id}        CF/LC history import
    POST /seed_hlr/{user_id}        CF/LC history import

/update gets the deepest treatment (see TestUpdateEndpoint): it must actually
MOVE mastery, persist it, mark the recommendation attempted, and project the
result into the recommendation graph -- not just return 200.

Run:
    python -m pytest tests/test_endpoints_smoke.py -v
"""

from __future__ import annotations

import contextlib
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import main as app_main
    import controllers.recommendation_controller as rec_ctl
    import controllers.submission_controller as sub_ctl
    import controllers.mastery_controller as mastery_ctl
    import controllers.seeding_controller as seed_ctl
    import middlewares.auth as auth_mod
    from fastapi.testclient import TestClient
    from pipeline.recommender.services.neo4j_graph_store import Neo4jGraphStore
    _AVAILABLE, _REASON = True, ""
except Exception as _exc:            # pragma: no cover - environment dependent
    _AVAILABLE = False
    _REASON = f"FastAPI app unavailable ({_exc.__class__.__name__}: {_exc})"

from tests.test_cold_start_personalization import (
    CATALOG, FakeQdrant, FakeDB, PERSONAS, _mastery_rows, _patched_pipeline,
)

USER = "a"
TOKEN = "a"


@unittest.skipUnless(_AVAILABLE, _REASON)
class _EndpointCase(unittest.TestCase):
    """Shared harness: a TestClient with every external system faked."""

    @contextlib.contextmanager
    def _client(self, mastery=None, hlr=None):
        fake_db = FakeDB(USER, _mastery_rows(PERSONAS[USER]))
        with contextlib.ExitStack() as st:
            ec = st.enter_context
            ec(patch.object(auth_mod, "verify_session_token",
                            side_effect=lambda token: token))
            # recommendation controller
            ec(patch.object(rec_ctl, "get_user_mastery", return_value=mastery or {}))
            ec(patch.object(rec_ctl, "get_user_hlr", return_value=hlr or {}))
            ec(patch.object(rec_ctl, "_get_db_wrapper", return_value=fake_db))
            ec(patch.object(rec_ctl, "_get_qdrant", return_value=FakeQdrant(CATALOG)))
            ec(patch.object(rec_ctl, "_get_neo4j_store",
                            return_value=Neo4jGraphStore(driver=None)))
            ec(patch.object(rec_ctl, "save_recommendation_log"))
            ec(patch.object(rec_ctl, "release_connection"))
            # mastery controller
            ec(patch.object(mastery_ctl, "get_user_mastery", return_value=mastery or {}))
            ec(patch.object(mastery_ctl, "get_user_hlr", return_value=hlr or {}))
            ec(_patched_pipeline())
            yield TestClient(app_main.app, raise_server_exceptions=False)

    def _auth(self):
        return {"Authorization": f"Bearer {TOKEN}"}


@unittest.skipUnless(_AVAILABLE, _REASON)
class TestHandlersDoNotBlockTheEventLoop(unittest.TestCase):
    """
    Every handler below performs BLOCKING I/O (pooled psycopg2 queries,
    Qdrant HTTP, Neo4j bolt). FastAPI runs `async def` handlers directly on
    the event loop and dispatches plain `def` handlers to a thread pool -- so
    any of these declared `async` would stall all concurrent traffic for the
    duration of its slowest dependency, and make the health probe time out on
    itself under load.

    This is a structural guard, not a convention: re-adding `async` to any of
    them fails here immediately.
    """

    BLOCKING_HANDLERS = (
        ("main", "root"),
        ("routes.recommendation", "recommend"),
        ("routes.recommendation", "topic_recommend"),
        ("routes.recommendation", "topic_problem_recommend"),
        ("routes.health", "health"),
        ("routes.mastery", "get_mastery"),
        ("routes.mastery", "get_urgency"),
        ("routes.submission", "update_endpoint"),
        ("routes.seeding", "seed_bkt"),
        ("routes.seeding", "seed_hlr"),
    )

    def test_blocking_handlers_are_sync_so_fastapi_threadpools_them(self):
        import importlib
        import inspect
        for module_name, func_name in self.BLOCKING_HANDLERS:
            mod = importlib.import_module(module_name)
            fn = getattr(mod, func_name)
            self.assertFalse(
                inspect.iscoroutinefunction(fn),
                f"{module_name}.{func_name} is `async def` but does blocking "
                f"I/O -- it would run on the event loop and stall every other "
                f"request. Declare it `def` so FastAPI dispatches it to the "
                f"thread pool.")

    def test_auth_middleware_offloads_its_blocking_db_call(self):
        """The middleware must stay async (ASGI contract), so its blocking
        session lookup has to be pushed to a thread pool explicitly."""
        import ast
        import inspect
        source = inspect.getsource(auth_mod.auth_middleware)
        tree = ast.parse(source.lstrip())
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn(
            "run_in_threadpool", calls,
            "auth_middleware calls verify_session_token (blocking psycopg2) "
            "directly on the event loop -- every authenticated request would "
            "serialise behind it. Wrap it in run_in_threadpool.")


class TestHealthEndpoints(_EndpointCase):

    def test_live_is_always_200_without_touching_dependencies(self):
        with self._client() as c:
            r = c.get("/live")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "alive")

    def test_root_and_health_report_ready_when_dependencies_are_up(self):
        import controllers.health_controller as hc
        with self._client() as c, \
             patch.object(hc, "_check_postgres", return_value=("ok", "")), \
             patch.object(hc, "_check_qdrant", return_value=("ok", "")), \
             patch.object(hc, "_check_neo4j", return_value=("ok", "")):
            for path in ("/", "/health"):
                r = c.get(path)
                self.assertEqual(r.status_code, 200, path)
                self.assertEqual(r.json()["status"], "ok", path)

    def test_root_and_health_return_503_when_a_critical_dependency_is_down(self):
        import controllers.health_controller as hc
        with self._client() as c, \
             patch.object(hc, "_check_postgres", return_value=("down", "boom")), \
             patch.object(hc, "_check_qdrant", return_value=("ok", "")), \
             patch.object(hc, "_check_neo4j", return_value=("ok", "")):
            for path in ("/", "/health"):
                r = c.get(path)
                self.assertEqual(r.status_code, 503, path)
                self.assertEqual(r.json()["status"], "unhealthy", path)

    def test_health_endpoints_need_no_auth(self):
        """Platform probes carry no session token."""
        with self._client() as c:
            for path in ("/", "/health", "/live"):
                self.assertNotEqual(c.get(path).status_code, 401, path)


class TestRecommendationEndpoints(_EndpointCase):

    def test_recommend_returns_slate(self):
        with self._client() as c:
            r = c.get(f"/recommend/{USER}?limit=8", headers=self._auth())
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["user_id"], USER)
        self.assertTrue(body["recommendations"])
        self.assertLessEqual(len(body["recommendations"]), 8)

    def test_recommend_clamps_limit(self):
        with self._client() as c:
            r = c.get(f"/recommend/{USER}?limit=9999", headers=self._auth())
        self.assertEqual(r.status_code, 200)
        self.assertLessEqual(len(r.json()["recommendations"]), 50)

    def test_topic_recommend_returns_a_topic(self):
        with self._client() as c:
            r = c.get(f"/topic/recommend/{USER}", headers=self._auth())
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["userId"], USER)
        self.assertTrue(body["topicId"])
        self.assertIn(body["reason"],
                      {"spaced_review", "in_progress", "unlocked",
                       "novelty", "cold_start"})

    def test_topic_problem_recommend_returns_problems_for_that_topic(self):
        with self._client() as c:
            r = c.post("/topic/recommend/problems", headers=self._auth(),
                       json={"userId": USER, "topicId": "graphs", "limit": 5})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["topicId"], "graphs")
        self.assertLessEqual(len(body["recommendations"]), 5)
        for rec in body["recommendations"]:
            self.assertIn("graphs", rec["topic_tags"])

    def test_topic_problem_recommend_validates_limit(self):
        with self._client() as c:
            r = c.post("/topic/recommend/problems", headers=self._auth(),
                       json={"userId": USER, "topicId": "graphs", "limit": 999})
        self.assertEqual(r.status_code, 422)

    def test_all_recommendation_routes_reject_other_users(self):
        with self._client() as c:
            self.assertEqual(
                c.get("/recommend/someone", headers=self._auth()).status_code, 403)
            self.assertEqual(
                c.get("/topic/recommend/someone", headers=self._auth()).status_code, 403)
            self.assertEqual(
                c.post("/topic/recommend/problems", headers=self._auth(),
                       json={"userId": "someone", "topicId": "graphs"}).status_code, 403)

    def test_routes_require_auth(self):
        with self._client() as c:
            self.assertEqual(c.get(f"/recommend/{USER}").status_code, 401)
            self.assertEqual(
                c.get(f"/recommend/{USER}",
                      headers={"Authorization": "Basic xyz"}).status_code, 401)


class TestMasteryEndpoints(_EndpointCase):

    def test_mastery_returns_scores_and_decayed_proficiency(self):
        mastery = {"graphs": 0.82, "dp": 0.30}
        hlr = {"graphs": {"half_life": 5.0,
                          "last_review": "2026-07-01T00:00:00+00:00",
                          "p_recall": 0.8, "next_review_days": 3.0}}
        with self._client(mastery=mastery, hlr=hlr) as c:
            r = c.get(f"/mastery/{USER}", headers=self._auth())
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["mastery"], mastery)
        self.assertIn("graphs", body["mastered_topics"])
        # graphs has stale HLR -> proficiency decayed below raw mastery
        self.assertLess(body["proficiency"]["graphs"], mastery["graphs"])
        # dp has no HLR state -> proficiency == raw mastery
        self.assertEqual(body["proficiency"]["dp"], mastery["dp"])

    def test_urgency_returns_scores(self):
        hlr = {"graphs": {"half_life": 2.0,
                          "last_review": "2026-07-01T00:00:00+00:00",
                          "p_recall": 0.5, "next_review_days": 1.0}}
        with self._client(hlr=hlr) as c:
            r = c.get(f"/urgency/{USER}", headers=self._auth())
        self.assertEqual(r.status_code, 200)
        scores = r.json()["urgency_scores"]
        self.assertIn("graphs", scores)
        self.assertGreaterEqual(scores["graphs"], 0.0)
        self.assertLessEqual(scores["graphs"], 1.0)

    def test_mastery_routes_reject_other_users(self):
        with self._client() as c:
            self.assertEqual(
                c.get("/mastery/someone", headers=self._auth()).status_code, 403)
            self.assertEqual(
                c.get("/urgency/someone", headers=self._auth()).status_code, 403)


class TestUpdateEndpoint(_EndpointCase):
    """
    /update is the write path: recompute BKT+HLR, persist both, mark the
    recommendation attempted, and project the result into the graph. It must
    actually MOVE mastery, not just return 200.
    """

    SUBMISSION = {
        "userId": USER,
        "problemId": "two-sum",
        "verdict": "OK",
        "hintsUsed": 0,
        "testCasesPassed": 10,
        "totalTestCases": 10,
        "submissionCount": 1,
        "normalisedScore": 0.95,
        "problemDifficulty": 0.35,
        "problemTopics": [
            {"topicId": "array", "currentMastery": 0.40, "weight": 0.7},
            {"topicId": "hash_map", "currentMastery": 0.20, "weight": 0.3},
        ],
    }

    @contextlib.contextmanager
    def _update_client(self):
        """Captures what /update persists so we can assert on it."""
        self.saved_mastery, self.saved_hlr, self.attempted = {}, {}, []
        state_svc = MagicMock()
        with contextlib.ExitStack() as st:
            ec = st.enter_context
            ec(patch.object(auth_mod, "verify_session_token",
                            side_effect=lambda token: token))
            ec(patch.object(sub_ctl, "save_user_mastery_live",
                            side_effect=lambda uid, m: self.saved_mastery.update(m)))
            ec(patch.object(sub_ctl, "save_user_hlr",
                            side_effect=lambda uid, h: self.saved_hlr.update(h)))
            ec(patch.object(sub_ctl, "mark_recommendation_attempted",
                            side_effect=lambda uid, slug: self.attempted.append(slug)))
            ec(patch.object(sub_ctl, "_get_state_update_service",
                            return_value=state_svc))
            self.state_svc = state_svc
            yield TestClient(app_main.app, raise_server_exceptions=False)

    def _post(self, client, body=None):
        return client.post("/update", headers=self._auth(),
                           json=body or self.SUBMISSION)

    def test_update_returns_200_with_expected_shape(self):
        with self._update_client() as c:
            r = self._post(c)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["userId"], USER)
        self.assertEqual(body["problemId"], "two-sum")
        for key in ("updatedTopics", "masteredTopics", "results"):
            self.assertIn(key, body)
        self.assertIn("bkt", body["results"])
        self.assertIn("hlr", body["results"])

    def test_update_actually_raises_mastery_on_a_clean_solve(self):
        with self._update_client() as c:
            body = self._post(c).json()
        by_topic = {t["topicId"]: t for t in body["updatedTopics"]}
        self.assertGreater(by_topic["array"]["updatedMastery"], 0.40,
                           "a clean accepted solve must increase mastery")
        self.assertGreater(by_topic["hash_map"]["updatedMastery"], 0.20)

    def test_update_respects_per_topic_weight(self):
        """array (weight .7) must gain more than hash_map (weight .3) would
        from the same submission at the same starting mastery."""
        body = dict(self.SUBMISSION)
        body["problemTopics"] = [
            {"topicId": "array", "currentMastery": 0.40, "weight": 0.9},
            {"topicId": "hash_map", "currentMastery": 0.40, "weight": 0.1},
        ]
        with self._update_client() as c:
            out = self._post(c, body).json()
        gains = {t["topicId"]: t["updatedMastery"] - 0.40
                 for t in out["updatedTopics"]}
        self.assertGreater(gains["array"], gains["hash_map"],
                           "topic weight is not influencing the BKT update")

    def test_update_does_not_raise_mastery_on_a_failed_submission(self):
        body = dict(self.SUBMISSION)
        body.update(verdict="WRONG_ANSWER", testCasesPassed=2,
                    normalisedScore=0.1)
        with self._update_client() as c:
            out = self._post(c, body).json()
        for t in out["updatedTopics"]:
            self.assertLessEqual(
                t["updatedMastery"], 0.40 if t["topicId"] == "array" else 0.20,
                "a failed attempt must not be credited as learning")

    def test_update_persists_mastery_and_hlr(self):
        with self._update_client() as c:
            self._post(c)
        self.assertTrue(self.saved_mastery, "mastery was never persisted")
        self.assertTrue(self.saved_hlr, "HLR was never persisted")
        self.assertIn("array", self.saved_mastery)
        self.assertIn("array", self.saved_hlr)

    def test_update_closes_the_recommendation_feedback_loop(self):
        with self._update_client() as c:
            self._post(c)
        self.assertEqual(self.attempted, ["two-sum"],
                         "submission must mark its recommendation attempted")

    def test_update_projects_into_the_recommendation_graph(self):
        with self._update_client() as c:
            self._post(c)
        self.state_svc.apply_update.assert_called_once()
        kwargs = self.state_svc.apply_update.call_args.kwargs
        self.assertTrue(kwargs["bkt_results"], "graph got no BKT results")
        self.assertEqual(kwargs["submission"]["problemId"], "two-sum")

    def test_update_rejects_another_users_submission(self):
        body = dict(self.SUBMISSION, userId="someone_else")
        with self._update_client() as c:
            self.assertEqual(self._post(c, body).status_code, 403)

    def test_update_rejects_malformed_payloads(self):
        with self._update_client() as c:
            # normalisedScore out of range -> 422 at the API boundary
            self.assertEqual(
                self._post(c, dict(self.SUBMISSION, normalisedScore=-50)).status_code,
                422)
            # missing required field
            self.assertEqual(
                self._post(c, {"userId": USER, "problemId": "x"}).status_code, 422)

    def test_update_accepts_a_service_token_without_a_user_session(self):
        """The backend's judge0 webhook has no end-user session."""
        with contextlib.ExitStack() as st:
            ec = st.enter_context
            ec(patch.object(auth_mod, "_ML_SERVICE_TOKEN", "svc-secret"))
            ec(patch.object(sub_ctl, "save_user_mastery_live"))
            ec(patch.object(sub_ctl, "save_user_hlr"))
            ec(patch.object(sub_ctl, "mark_recommendation_attempted"))
            ec(patch.object(sub_ctl, "_get_state_update_service",
                            return_value=MagicMock()))
            client = TestClient(app_main.app, raise_server_exceptions=False)
            r = client.post("/update", json=self.SUBMISSION,
                            headers={"Authorization": "Bearer svc-secret"})
        self.assertEqual(r.status_code, 200)


class TestSeedingEndpoints(_EndpointCase):

    def test_seed_endpoints_dispatch_and_guard_ownership(self):
        with contextlib.ExitStack() as st:
            ec = st.enter_context
            ec(patch.object(auth_mod, "verify_session_token",
                            side_effect=lambda token: token))
            ec(patch.object(seed_ctl, "handle_seed_bkt",
                            return_value={"ok": "bkt"}))
            ec(patch.object(seed_ctl, "handle_seed_hlr",
                            return_value={"ok": "hlr"}))
            c = TestClient(app_main.app, raise_server_exceptions=False)
            self.assertEqual(c.post(f"/seed_bkt/{USER}",
                                    headers=self._auth()).status_code, 200)
            self.assertEqual(c.post(f"/seed_hlr/{USER}",
                                    headers=self._auth()).status_code, 200)
            self.assertEqual(c.post("/seed_bkt/someone",
                                    headers=self._auth()).status_code, 403)
            self.assertEqual(c.post("/seed_hlr/someone",
                                    headers=self._auth()).status_code, 403)


if __name__ == "__main__":
    unittest.main(verbosity=2)

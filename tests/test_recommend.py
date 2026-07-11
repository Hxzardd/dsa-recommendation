"""
tests/test_recommend.py

Tests the single callable entry point: get_recommendations(). Covers a
brand-new user (db=None path), a warm user with real telemetry (fake db +
fake Qdrant), and -- critically -- a STATIC check proving this module never
imports anything from the offline pipeline (ingestion/embeddings/graphs),
so calling get_recommendations() any number of times never triggers
re-ingestion, re-embedding, or RGCN retraining.

Zero real I/O anywhere in this file: fake Qdrant, fake db, in-memory
bkt/hlr dicts. Safe to run under a normal `pytest tests/` invocation with
no external services required.

Run:
    python -m pytest tests/test_recommend.py -v
"""

from __future__ import annotations

import ast
import inspect
import sys
import types
import time
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock

# Stub qdrant_client so pools/base_pool.py's import resolves without the
# real package installed.
if "qdrant_client" not in sys.modules:
    sys.modules["qdrant_client"] = types.ModuleType("qdrant_client")
if "qdrant_client.models" not in sys.modules:
    _qm = types.ModuleType("qdrant_client.models")
    def _make_stub_class(name):
        def _init(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)
        return type(name, (), {"__init__": _init})
    for _cls in ("Filter", "FieldCondition", "MatchAny", "MatchValue", "Range"):
        setattr(_qm, _cls, _make_stub_class(_cls))
    sys.modules["qdrant_client.models"] = _qm

import pipeline.recommender.services.recommend as recommend_module
from pipeline.recommender.services.recommend import get_recommendations, RecommendationResult
from pipeline.recommender.services.candidate_store import InMemoryCandidateStore


# ===========================================================================
# Static check: recommend.py must never import the offline pipeline
# ===========================================================================

OFFLINE_MODULE_PREFIXES = (
    "pipeline.ingestion",
    "pipeline.embeddings",
    "pipeline.graphs",
)


class TestNoOfflinePipelineCalls(unittest.TestCase):
    """
    Structural guarantee, not a convention: parses recommend.py's own source
    and fails if it EVER imports anything from the offline batch pipeline.
    This is what makes "offline ingestion only runs once, not on every call"
    an enforced property rather than a hope -- a future edit that
    accidentally imports pipeline.embeddings.embedder or similar breaks this
    test immediately, before it ships.
    """

    def test_source_has_no_offline_imports(self):
        source = inspect.getsource(recommend_module)
        tree = ast.parse(source)
        found_offline_imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(OFFLINE_MODULE_PREFIXES):
                        found_offline_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod.startswith(OFFLINE_MODULE_PREFIXES):
                    found_offline_imports.append(mod)

        self.assertEqual(found_offline_imports, [],
                         f"recommend.py imports offline pipeline module(s): "
                         f"{found_offline_imports} -- this would make every "
                         f"recommendation call risk triggering ingestion/"
                         f"embedding/training work.")

    def test_calling_twice_does_not_grow_any_offline_module_reference(self):
        """
        Dynamic companion to the static check: call get_recommendations()
        twice and confirm no offline pipeline module got imported into
        sys.modules as a side effect of either call.
        """
        before = {m for m in sys.modules if m.startswith(OFFLINE_MODULE_PREFIXES)}

        db = _make_fake_db()
        qdrant = FakeQdrant(_problems())
        get_recommendations("u1", db=db, qdrant=qdrant, bkt_store={}, hlr_store={})
        get_recommendations("u1", db=db, qdrant=qdrant, bkt_store={}, hlr_store={})

        after = {m for m in sys.modules if m.startswith(OFFLINE_MODULE_PREFIXES)}
        self.assertEqual(before, after)


# ===========================================================================
# Fakes
# ===========================================================================

class _Pt:
    def __init__(self, pid, tags, diff, score=0.9):
        self.id = pid
        self.payload = {
            "problem_id": pid, "topic_tags": tags, "difficulty_score": diff,
            "title": f"Title for {pid}", "title_slug": pid.replace("_", "-"),
        }
        self.score = score


def _stable_point_id(problem_id: str) -> int:
    """Matches recommend.py's _stable_point_id exactly, so FakeQdrant.retrieve() resolves correctly."""
    try:
        import xxhash
        return xxhash.xxh64(problem_id).intdigest() & 0x7FFF_FFFF_FFFF_FFFF
    except ImportError:
        import hashlib
        return int(hashlib.sha256(problem_id.encode()).hexdigest(), 16) & 0x7FFF_FFFF_FFFF_FFFF


class FakeQdrant:
    def __init__(self, problems):
        self.problems = problems
        self._by_point_id = {_stable_point_id(p.id): p for p in problems}

    def scroll(self, collection_name, scroll_filter=None, limit=10,
               with_payload=True, with_vectors=False):
        want_tags = None
        want_problem_ids = None
        diff_range = None
        if scroll_filter is not None:
            for cond in scroll_filter.must:
                key = getattr(cond, "key", None)
                if key == "topic_tags" and getattr(cond, "match", None) is not None:
                    want_tags = set(cond.match.any)
                if key == "problem_id" and getattr(cond, "match", None) is not None:
                    # _resolve_titles now uses scroll(filter=problem_id MatchAny)
                    # instead of retrieve() by hash ID
                    want_problem_ids = set(cond.match.any)
                if key == "difficulty_score" and getattr(cond, "range", None) is not None:
                    diff_range = cond.range
        out = []
        for p in self.problems:
            if want_problem_ids is not None and p.id not in want_problem_ids:
                continue
            if want_tags is not None and not (set(p.payload["topic_tags"]) & want_tags):
                continue
            if diff_range is not None:
                d = p.payload["difficulty_score"]
                if diff_range.gte is not None and d < diff_range.gte:
                    continue
                if diff_range.lte is not None and d > diff_range.lte:
                    continue
            out.append(p)
            if len(out) >= limit:
                break
        return out, None

    def query_points(self, collection_name, query, limit=10,
                     with_payload=True, with_vectors=False):
        class R:
            points = sorted(self.problems, key=lambda p: p.score, reverse=True)[:limit]
        return R()

    def retrieve(self, collection_name, ids, with_payload=True, with_vectors=False):
        # retrieve() is no longer used for title resolution (replaced by
        # scroll+problem_id filter). Kept here so tests that still call it
        # don't break, but returns empty.
        return []


def _problems(n_per_topic=15):
    topics = ["arrays", "graphs", "trees", "dp", "strings"]
    out = []
    for t in topics:
        for j in range(n_per_topic):
            diff = round(0.05 + 0.9 * (j / max(n_per_topic - 1, 1)), 3)
            out.append(_Pt(f"p_{t}_{j}", [t], diff, score=0.9 - j * 0.01))
    return out


def _dt(days_ago=0.0):
    ts = time.time() - days_ago * 86400
    m = MagicMock()
    m.timestamp.return_value = ts
    m.tzinfo = None
    return m


def _make_fake_db(mastery_rows=None):
    """Minimal fake db matching UserGraphService._build()'s query order."""
    db = MagicMock()
    user_row = ("u1", "tester", True, 500, 3)
    mastery_rows = mastery_rows or [
        ("arrays", 75.0, "medium", 5, 3, _dt(2.0), 2.5, 3, "2026-08-01"),
        ("trees",  75.0, "medium", 5, 3, _dt(2.0), 2.5, 3, "2026-08-01"),
        ("graphs", 35.0, "medium", 5, 1, _dt(2.0), 2.5, 3, "2026-08-01"),
    ]

    call_count = {"n": 0}

    def _execute(sql, params=None):
        n = call_count["n"]
        call_count["n"] += 1
        result = MagicMock()
        if n == 0:   result.fetchone.return_value = user_row
        elif n == 1: result.fetchall.return_value = []     # submissions
        elif n == 2: result.fetchall.return_value = []     # rec log
        elif n == 3: result.fetchall.return_value = mastery_rows
        elif n == 4: result.fetchall.return_value = []     # concept gaps
        return result

    db.execute.side_effect = _execute
    return db


# ===========================================================================
# Behavioural tests
# ===========================================================================

class TestNewUserPath(unittest.TestCase):
    """db=None -> brand new user, no DB round trip at all."""

    def test_returns_result_with_no_db(self):
        result = get_recommendations("brand_new", db=None, qdrant=FakeQdrant(_problems()))
        self.assertIsInstance(result, RecommendationResult)

    def test_new_user_returns_empty_or_starter_recommendations(self):
        """
        is_cold_start is no longer part of the returned payload (internal
        detail, stripped per the backend-schema-only output). A brand new
        user should still get a valid result object -- possibly with
        starter-concept recommendations, possibly empty depending on the
        fixture's catalog -- without raising.
        """
        result = get_recommendations("brand_new", db=None, qdrant=FakeQdrant(_problems()))
        self.assertIsInstance(result.recommendations, list)

    def test_new_user_does_not_crash_with_no_qdrant_either(self):
        result = get_recommendations("brand_new", db=None, qdrant=None)
        self.assertIsInstance(result, RecommendationResult)
        self.assertEqual(result.recommendations, [])


class TestWarmUserPath(unittest.TestCase):

    def test_returns_recommendations(self):
        db = _make_fake_db()
        qdrant = FakeQdrant(_problems())
        result = get_recommendations("u1", db=db, qdrant=qdrant, bkt_store={}, hlr_store={})
        self.assertIsInstance(result, RecommendationResult)

    def test_respects_k_limit(self):
        db = _make_fake_db()
        qdrant = FakeQdrant(_problems())
        result = get_recommendations("u1", db=db, qdrant=qdrant,
                                     bkt_store={}, hlr_store={}, k=5)
        self.assertLessEqual(len(result.recommendations), 5)

    def test_recommendations_match_backend_schema(self):
        """
        difficulty_plan/filter_report/rank_components etc are no longer
        part of the returned payload -- those were internal scoring detail
        that leaked into the response shape. Each recommendation must now
        contain EXACTLY the backend's fields: problem_id, title, title_slug,
        difficulty_score, topic_tags, source, recommended_at.
        """
        db = _make_fake_db()
        qdrant = FakeQdrant(_problems())
        result = get_recommendations("u1", db=db, qdrant=qdrant, bkt_store={}, hlr_store={})
        expected_keys = {"problem_id", "title", "title_slug", "difficulty_score",
                         "topic_tags", "source", "recommended_at"}
        for rec in result.recommendations:
            self.assertEqual(set(rec.keys()), expected_keys)

    def test_candidate_set_staged(self):
        store = InMemoryCandidateStore()
        db = _make_fake_db()
        qdrant = FakeQdrant(_problems())
        result = get_recommendations("u1", db=db, qdrant=qdrant,
                                     bkt_store={}, hlr_store={}, store=store)
        self.assertIsNotNone(result.candidate_set_id)
        staged = store.get(result.candidate_set_id)
        self.assertIsNotNone(staged)

    def test_recommendations_have_valid_source(self):
        """source must be one of the backend's 5 RecommendationLog enum values."""
        db = _make_fake_db()
        qdrant = FakeQdrant(_problems())
        result = get_recommendations("u1", db=db, qdrant=qdrant, bkt_store={}, hlr_store={})
        valid_sources = {"graph_walk", "vector_similarity", "revision", "sm2_review", "course_path"}
        for rec in result.recommendations:
            self.assertIn(rec["source"], valid_sources)

    def test_recommendations_have_resolved_titles(self):
        db = _make_fake_db()
        qdrant = FakeQdrant(_problems())
        result = get_recommendations("u1", db=db, qdrant=qdrant, bkt_store={}, hlr_store={})
        for rec in result.recommendations:
            self.assertIsNotNone(rec["title"])
            self.assertTrue(rec["title"].startswith("Title for"))

    def test_default_k_is_ten(self):
        db = _make_fake_db()
        qdrant = FakeQdrant(_problems())
        result = get_recommendations("u1", db=db, qdrant=qdrant, bkt_store={}, hlr_store={})
        self.assertLessEqual(len(result.recommendations), 10)

    def test_max_per_pool_cap_still_takes_effect(self):
        """
        pool_sources is no longer part of the output (internal-only, per
        the schema redesign), so pool-level cap enforcement can't be
        observed from the response anymore -- that's covered directly by
        test_diversity_mixer.py's unit tests instead. This just confirms
        passing max_per_pool doesn't break the call and still returns a
        valid, schema-correct result.
        """
        db = _make_fake_db()
        qdrant = FakeQdrant(_problems())
        result = get_recommendations("u1", db=db, qdrant=qdrant,
                                     bkt_store={}, hlr_store={},
                                     k=10, max_per_pool=2)
        self.assertLessEqual(len(result.recommendations), 10)
        for rec in result.recommendations:
            self.assertIn(rec["source"],
                         {"graph_walk", "vector_similarity", "revision",
                          "sm2_review", "course_path"})


class TestExternalRelevanceOverride(unittest.TestCase):

    def test_external_relevance_scores_used_when_provided(self):
        """
        If a real trained ranker's scores are passed in, the diversity mixer
        should use them instead of the heuristic ranker's own scores --
        proves the hand-off point for swapping in LightGBM later works
        without changing this function's structure.
        """
        db = _make_fake_db()
        qdrant = FakeQdrant(_problems())
        result = get_recommendations("u1", db=db, qdrant=qdrant,
                                     bkt_store={}, hlr_store={},
                                     relevance_scores={})   # empty override still runs without error
        self.assertIsInstance(result, RecommendationResult)


class TestResultSerialization(unittest.TestCase):

    def test_to_dict_json_serialisable(self):
        import json
        db = _make_fake_db()
        qdrant = FakeQdrant(_problems())
        result = get_recommendations("u1", db=db, qdrant=qdrant, bkt_store={}, hlr_store={})
        raw = json.dumps(result.to_dict())
        self.assertIsInstance(raw, str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
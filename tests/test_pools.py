"""
tests/test_pools.py

Tests all seven candidate pools without a real Qdrant. A fake client returns
tagged/difficulty-scored problems so each pool's selection logic can be checked.

Run:
    python -m pytest tests/test_pools.py -v
"""

from __future__ import annotations

import time
import unittest
from datetime import datetime, timezone, timedelta

from pipeline.recommender.models.user_graph import (
    UserGraph, UserNode, ConceptEdge, ConceptConceptEdge, EdgeType,
)
from pipeline.recommender.pools.pools import (
    CoursePathPool, TransferPool, WeaknessPool, SpacedReviewPool,
    StretchPool, NoveltyPool, VectorPool, build_pools, POOL_CLASSES,
)

NOW = time.time()


def _iso(days):
    return (datetime.fromtimestamp(NOW, tz=timezone.utc) + timedelta(days=days)).isoformat()


# --------------------------------------------------------------------------
# Fake Qdrant
# --------------------------------------------------------------------------

class _Pt:
    def __init__(self, pid, tags, diff, score=0.9):
        self.id = pid
        self.payload = {"problem_id": pid, "topic_tags": tags, "difficulty_score": diff}
        self.score = score
        self.vector = None


class FakeQdrant:
    """
    Minimal stand-in. `problems` is a list of _Pt.
    scroll() filters by topic_tags MatchAny and difficulty Range.
    query_points() returns everything sorted by score (ANN stand-in).
    """
    def __init__(self, problems):
        self.problems = problems

    def scroll(self, collection_name, scroll_filter=None, limit=10,
               with_payload=True, with_vectors=False):
        want_tags = None
        diff_range = None
        if scroll_filter is not None:
            for cond in scroll_filter.must:
                key = getattr(cond, "key", None)
                if key == "topic_tags" and getattr(cond, "match", None) is not None:
                    want_tags = set(cond.match.any)
                if key == "difficulty_score" and getattr(cond, "range", None) is not None:
                    diff_range = cond.range
        out = []
        for p in self.problems:
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


def _problems():
    return [
        _Pt("p_arrays_easy",  ["arrays"],   0.2),
        _Pt("p_arrays_med",   ["arrays"],   0.5),
        _Pt("p_arrays_hard",  ["arrays"],   0.8),
        _Pt("p_graphs_easy",  ["graphs"],   0.2),
        _Pt("p_graphs_hard",  ["graphs"],   0.85),
        _Pt("p_dp_med",       ["dp"],       0.5),
        _Pt("p_trees_easy",   ["trees"],    0.25),
        _Pt("p_solved",       ["arrays"],   0.3),
    ]


class _StubState:
    """Minimal UserStateVector stand-in for pool tests."""
    def __init__(self, qv, graph):
        self._qv = qv
        self.solved_ids = set(graph.solved_ids)
        self.is_cold_start = qv is None
    def to_query_vector(self):
        return self._qv


def _graph(concepts=None, solved=None, cc=None):
    g = UserGraph(user=UserNode(user_id="u1"))
    for c in (concepts or []):
        g.add_concept_edge(c)
    for pid in (solved or []):
        g.solved_ids.add(pid)
    for e in (cc or []):
        g.add_cc_edge(e)
    return g


def _concept(slug, mastery=0.5, urgency=0.0, severity=0.0,
             edge_type=EdgeType.LEARNING, next_review_date=None):
    return ConceptEdge(slug, edge_type, mastery_score=mastery,
                       urgency=urgency, severity=severity,
                       next_review_date=next_review_date)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

class TestCoursePath(unittest.TestCase):
    def test_returns_in_progress_concepts(self):
        g = _graph([_concept("arrays", mastery=0.5)], solved=[])
        pool = CoursePathPool(qdrant=FakeQdrant(_problems()))
        cands = pool.generate(g, _StubState(None, g), n=10)
        self.assertTrue(any("arrays" in c.problem_id for c in cands))

    def test_excludes_solved(self):
        g = _graph([_concept("arrays", mastery=0.5)], solved=["p_solved"])
        pool = CoursePathPool(qdrant=FakeQdrant(_problems()))
        cands = pool.generate(g, _StubState(None, g), n=10)
        self.assertNotIn("p_solved", [c.problem_id for c in cands])


class TestTransfer(unittest.TestCase):
    def test_uses_ann_when_vector_present(self):
        g = _graph([_concept("arrays", mastery=0.6)], solved=[])
        pool = TransferPool(qdrant=FakeQdrant(_problems()))
        cands = pool.generate(g, _StubState([1.0]*1920, g), n=5)
        self.assertGreater(len(cands), 0)


class TestWeakness(unittest.TestCase):
    def test_targets_weak_concepts(self):
        g = _graph([
            _concept("graphs", mastery=0.3, severity=0.8, edge_type=EdgeType.WEAK),
        ], solved=[])
        pool = WeaknessPool(qdrant=FakeQdrant(_problems()))
        cands = pool.generate(g, _StubState(None, g), n=10)
        self.assertTrue(any("graphs" in c.problem_id for c in cands))

    def test_draws_easier_difficulty(self):
        g = _graph([
            _concept("arrays", mastery=0.3, severity=0.7, edge_type=EdgeType.WEAK),
        ], solved=[])
        pool = WeaknessPool(qdrant=FakeQdrant(_problems()))
        cands = pool.generate(g, _StubState(None, g), n=10)
        # hard arrays problem (0.8) should be excluded by the easy-med band
        self.assertNotIn("p_arrays_hard", [c.problem_id for c in cands])

    def test_empty_when_no_weak(self):
        g = _graph([_concept("arrays", mastery=0.9)], solved=[])
        pool = WeaknessPool(qdrant=FakeQdrant(_problems()))
        cands = pool.generate(g, _StubState(None, g), n=10)
        self.assertEqual(cands, [])


class TestSpacedReview(unittest.TestCase):
    def test_overdue_concept_included(self):
        g = _graph([
            _concept("arrays", mastery=0.6, next_review_date=_iso(-3)),
        ], solved=[])
        pool = SpacedReviewPool(qdrant=FakeQdrant(_problems()))
        cands = pool.generate(g, _StubState(None, g), n=10)
        self.assertTrue(any("arrays" in c.problem_id for c in cands))

    def test_urgent_concept_included(self):
        g = _graph([
            _concept("graphs", mastery=0.6, urgency=0.8),
        ], solved=[])
        pool = SpacedReviewPool(qdrant=FakeQdrant(_problems()))
        cands = pool.generate(g, _StubState(None, g), n=10)
        self.assertTrue(any("graphs" in c.problem_id for c in cands))

    def test_future_review_excluded(self):
        g = _graph([
            _concept("arrays", mastery=0.6, next_review_date=_iso(+5)),
        ], solved=[])
        pool = SpacedReviewPool(qdrant=FakeQdrant(_problems()))
        cands = pool.generate(g, _StubState(None, g), n=10)
        self.assertEqual(cands, [])


class TestStretch(unittest.TestCase):
    def test_targets_partial_mastery(self):
        g = _graph([_concept("dp", mastery=0.5)], solved=[])
        pool = StretchPool(qdrant=FakeQdrant(_problems()))
        cands = pool.generate(g, _StubState(None, g), n=10)
        self.assertTrue(any("dp" in c.problem_id for c in cands))

    def test_excludes_easy(self):
        g = _graph([_concept("arrays", mastery=0.5)], solved=[])
        pool = StretchPool(qdrant=FakeQdrant(_problems()))
        cands = pool.generate(g, _StubState(None, g), n=10)
        self.assertNotIn("p_arrays_easy", [c.problem_id for c in cands])


class TestNovelty(unittest.TestCase):
    def test_unseen_concept_from_mastered(self):
        g = _graph(
            [_concept("arrays", mastery=0.8, edge_type=EdgeType.MASTERED)],
            solved=[],
            cc=[ConceptConceptEdge("arrays", "trees", EdgeType.PREREQ, 1.0)],
        )
        pool = NoveltyPool(qdrant=FakeQdrant(_problems()))
        cands = pool.generate(g, _StubState(None, g), n=10)
        self.assertTrue(any("trees" in c.problem_id for c in cands))

    def test_empty_when_no_reachable_novel(self):
        g = _graph([_concept("arrays", mastery=0.8, edge_type=EdgeType.MASTERED)],
                   solved=[])
        pool = NoveltyPool(qdrant=FakeQdrant(_problems()))
        cands = pool.generate(g, _StubState(None, g), n=10)
        self.assertEqual(cands, [])


class TestVector(unittest.TestCase):
    def test_ann_returns_candidates(self):
        g = _graph([_concept("arrays", mastery=0.6)], solved=[])
        pool = VectorPool(qdrant=FakeQdrant(_problems()))
        cands = pool.generate(g, _StubState([1.0]*1920, g), n=5)
        self.assertGreater(len(cands), 0)

    def test_cold_start_returns_empty(self):
        g = _graph([_concept("arrays", mastery=0.6)], solved=[])
        pool = VectorPool(qdrant=FakeQdrant(_problems()))
        cands = pool.generate(g, _StubState(None, g), n=5)
        self.assertEqual(cands, [])


class TestRegistry(unittest.TestCase):
    def test_build_pools_has_all_seven(self):
        pools = build_pools(qdrant=FakeQdrant(_problems()))
        self.assertEqual(set(pools.keys()),
                         {"A", "B_C", "D", "E", "F", "G", "vector"})

    def test_all_pools_exclude_solved(self):
        g = _graph([
            _concept("arrays", mastery=0.5),
            _concept("graphs", mastery=0.3, severity=0.8, edge_type=EdgeType.WEAK),
        ], solved=["p_solved"])
        pools = build_pools(qdrant=FakeQdrant(_problems()))
        state = _StubState([1.0]*1920, g)
        for name, pool in pools.items():
            cands = pool.generate(g, state, n=10)
            self.assertNotIn("p_solved", [c.problem_id for c in cands],
                             msg=f"pool {name} leaked a solved problem")

    def test_candidates_carry_pool_name(self):
        g = _graph([_concept("arrays", mastery=0.5)], solved=[])
        pool = CoursePathPool(qdrant=FakeQdrant(_problems()))
        cands = pool.generate(g, _StubState(None, g), n=5)
        for c in cands:
            self.assertEqual(c.pool, "A")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestColdStartFallback(unittest.TestCase):
    """
    Regression test for a real bug: CoursePathPool's and NoveltyPool's
    cold-start branches used to read graph.concept_edges / mastered
    concepts to build their fallback list -- which is exactly what's
    EMPTY for a genuinely brand-new user, so both pools silently
    returned zero candidates for every new signup.
    """

    def test_course_path_returns_candidates_for_totally_cold_user(self):
        problems = [
            _Pt("array_0", ["array"], 0.2),
            _Pt("string_0", ["string"], 0.2),
        ]
        pool = CoursePathPool(qdrant=FakeQdrant(problems))
        graph = _graph()   # zero concepts, zero cc_edges, zero solved
        cands = pool.generate(graph, _StubState(None, graph), n=10)
        self.assertGreater(len(cands), 0,
                           "CoursePathPool returned nothing for a cold-start "
                           "user -- starter concept fallback is broken")

    def test_novelty_returns_candidates_for_totally_cold_user(self):
        problems = [
            _Pt("array_0", ["array"], 0.2),
            _Pt("string_0", ["string"], 0.2),
        ]
        pool = NoveltyPool(qdrant=FakeQdrant(problems))
        graph = _graph()
        cands = pool.generate(graph, _StubState(None, graph), n=10)
        self.assertGreater(len(cands), 0,
                           "NoveltyPool returned nothing for a cold-start "
                           "user -- starter concept fallback is broken")
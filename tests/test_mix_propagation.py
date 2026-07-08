"""
tests/test_mix_propagation.py

Verifies the actual fix: the adaptive difficulty controller's per-pool
easy/medium/hard MIX now reaches each pool's Qdrant queries in the right
proportions (not just a flat n with one fixed band), and that the new
hard ceilings (MAX_TOTAL_CANDIDATES, MAX_PER_POOL_ABSOLUTE) are respected.

Run:
    python -m pytest tests/test_mix_propagation.py -v
"""

from __future__ import annotations

import sys
import types
import unittest

# Stub qdrant_client so base_pool.py's `from qdrant_client.models import ...`
# resolves without the real package installed. Self-contained here rather
# than relying on another test file happening to run first and leaving the
# stub in sys.modules -- that's an ordering-dependent accident, not a fix.
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

from pipeline.recommender.models.user_graph import (
    UserGraph, UserNode, ConceptEdge, EdgeType,
)
from pipeline.recommender.pools.pools import WeaknessPool, StretchPool, CoursePathPool
from pipeline.recommender.pools.base_pool import EASY_BAND, MED_BAND, HARD_BAND
from pipeline.recommender.services.pool_generation import (
    PoolGenerationOrchestrator, MAX_TOTAL_CANDIDATES, MAX_PER_POOL_ABSOLUTE,
)


class _Pt:
    def __init__(self, pid, tags, diff):
        self.id = pid
        self.payload = {"problem_id": pid, "topic_tags": tags, "difficulty_score": diff}
        self.score = 0.9


class RecordingQdrant:
    """
    Fake Qdrant that RECORDS every difficulty range it was scrolled with,
    so tests can assert on exactly what bands/counts were requested --
    not just what came back.
    """
    def __init__(self, problems):
        self.problems = problems
        self.scroll_calls = []   # list of (difficulty_range, limit)

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

        self.scroll_calls.append({
            "range": (diff_range.gte, diff_range.lte) if diff_range else None,
            "limit": limit,
        })

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
        return out[:limit], None


def _many_problems(tag, count=100):
    """Spread `count` problems evenly across the full 0-1 difficulty range."""
    return [_Pt(f"{tag}_{i}", [tag], round(i / max(count - 1, 1), 4)) for i in range(count)]


def _concept(slug, mastery=0.5):
    return ConceptEdge(slug, EdgeType.LEARNING, mastery_score=mastery)


class TestMixReachesQdrant(unittest.TestCase):
    """
    Confirms mix is not just accepted as a parameter but actually changes
    the difficulty ranges and per-band counts sent to Qdrant.
    """

    def test_weakness_pool_requests_only_easy_and_medium_bands(self):
        qdrant = RecordingQdrant(_many_problems("dp", 100))
        pool = WeaknessPool(qdrant=qdrant)
        graph = UserGraph(user=UserNode(user_id="u1"))
        graph.add_concept_edge(ConceptEdge("dp", EdgeType.WEAK, mastery_score=0.3, severity=0.8))

        mix = {"easy": 0.3, "medium": 0.3, "hard": 0.4}   # global mix HAS hard
        pool.generate(graph, None, n=20, mix=mix)

        ranges_requested = {c["range"] for c in qdrant.scroll_calls if c["range"]}
        # WeaknessPool.ALLOWED_BANDS = ("easy", "medium") -- HARD_BAND must
        # never appear, even though the global mix included 40% hard.
        self.assertNotIn(HARD_BAND, ranges_requested)

    def test_stretch_pool_requests_only_medium_and_hard_bands(self):
        qdrant = RecordingQdrant(_many_problems("graphs", 100))
        pool = StretchPool(qdrant=qdrant)
        graph = UserGraph(user=UserNode(user_id="u1"))
        graph.add_concept_edge(ConceptEdge("graphs", EdgeType.LEARNING, mastery_score=0.5))

        mix = {"easy": 0.5, "medium": 0.3, "hard": 0.2}   # global mix HAS easy
        pool.generate(graph, None, n=20, mix=mix)

        ranges_requested = {c["range"] for c in qdrant.scroll_calls if c["range"]}
        self.assertNotIn(EASY_BAND, ranges_requested)

    def test_mix_proportions_change_per_band_counts(self):
        """
        A mix heavily weighted toward 'easy' should request more easy-band
        candidates than a mix heavily weighted toward 'hard', for the same
        pool and total n.
        """
        qdrant_a = RecordingQdrant(_many_problems("arrays", 100))
        pool_a = CoursePathPool(qdrant=qdrant_a)
        graph = UserGraph(user=UserNode(user_id="u1"))
        graph.add_concept_edge(_concept("arrays", mastery=0.5))

        easy_heavy_mix = {"easy": 0.8, "medium": 0.15, "hard": 0.05}
        pool_a.generate(graph, None, n=20, mix=easy_heavy_mix)
        easy_call = next(c for c in qdrant_a.scroll_calls if c["range"] == EASY_BAND)

        qdrant_b = RecordingQdrant(_many_problems("arrays", 100))
        pool_b = CoursePathPool(qdrant=qdrant_b)
        hard_heavy_mix = {"easy": 0.05, "medium": 0.15, "hard": 0.8}
        pool_b.generate(graph, None, n=20, mix=hard_heavy_mix)
        easy_call_2 = next(c for c in qdrant_b.scroll_calls if c["range"] == EASY_BAND)

        # easy-heavy mix should have requested a bigger easy-band limit
        self.assertGreater(easy_call["limit"], easy_call_2["limit"])

    def test_band_counts_sum_to_requested_n(self):
        """
        The three per-band counts (when all three bands are allowed) should
        sum to n. Note: _problems_by_concept intentionally over-fetches at
        limit=count*3 to leave headroom for exclusion filtering, so the
        recorded Qdrant `limit` per call is 3x the per-band count -- divide
        back out before summing.
        """
        qdrant = RecordingQdrant(_many_problems("arrays", 100))
        pool = CoursePathPool(qdrant=qdrant)
        graph = UserGraph(user=UserNode(user_id="u1"))
        graph.add_concept_edge(_concept("arrays", mastery=0.5))

        mix = {"easy": 0.5, "medium": 0.3, "hard": 0.2}
        pool.generate(graph, None, n=20, mix=mix)

        total_requested = sum(c["limit"] // 3 for c in qdrant.scroll_calls if c["range"])
        self.assertEqual(total_requested, 20)

    def test_none_mix_falls_back_to_full_allowed_range(self):
        """
        Backward compatibility: no mix passed -> single query spanning this
        pool's full allowed difficulty range. The code still attaches an
        explicit Range(0.0, 1.0) condition even though it matches everything
        -- that's harmless and correct, not something to avoid.
        """
        qdrant = RecordingQdrant(_many_problems("arrays", 100))
        pool = CoursePathPool(qdrant=qdrant)
        graph = UserGraph(user=UserNode(user_id="u1"))
        graph.add_concept_edge(_concept("arrays", mastery=0.5))

        pool.generate(graph, None, n=20, mix=None)
        ranged_calls = [c for c in qdrant.scroll_calls if c["range"]]
        self.assertEqual(len(ranged_calls), 1)
        lo, hi = ranged_calls[0]["range"]
        self.assertEqual((lo, hi), (EASY_BAND[0], HARD_BAND[1]))


class TestHardCeilings(unittest.TestCase):

    def test_total_n_clamped_to_max_total_candidates(self):
        qdrant = RecordingQdrant(_many_problems("arrays", 500))
        graph = UserGraph(user=UserNode(user_id="u1"))
        graph.add_concept_edge(_concept("arrays", mastery=0.5))
        graph.solved_ids.add("dummy")

        orchestrator = PoolGenerationOrchestrator(qdrant=qdrant)

        class _StubState:
            def to_query_vector(self): return None

        result = orchestrator.generate(graph, _StubState(), total_n=100000)
        total_requested = sum(result.requested_counts.values())
        self.assertLessEqual(total_requested, MAX_TOTAL_CANDIDATES)

    def test_single_pool_never_exceeds_absolute_cap(self):
        qdrant = RecordingQdrant(_many_problems("arrays", 500))
        # a user whose entire signal points at one pool (heavy weakness -> pool D)
        graph = UserGraph(user=UserNode(user_id="u1"))
        for i in range(10):
            graph.add_concept_edge(ConceptEdge(f"weak{i}", EdgeType.WEAK,
                                               mastery_score=0.2, severity=0.9))
        graph.solved_ids.add("dummy")

        orchestrator = PoolGenerationOrchestrator(qdrant=qdrant)

        class _StubState:
            def to_query_vector(self): return None

        result = orchestrator.generate(graph, _StubState(), total_n=MAX_TOTAL_CANDIDATES)
        for pool_name, n in result.requested_counts.items():
            self.assertLessEqual(n, MAX_PER_POOL_ABSOLUTE)

    def test_requested_counts_exposed_in_result(self):
        qdrant = RecordingQdrant(_many_problems("arrays", 100))
        graph = UserGraph(user=UserNode(user_id="u1"))
        graph.add_concept_edge(_concept("arrays", mastery=0.5))
        graph.solved_ids.add("dummy")

        orchestrator = PoolGenerationOrchestrator(qdrant=qdrant)

        class _StubState:
            def to_query_vector(self): return None

        result = orchestrator.generate(graph, _StubState(), total_n=30)
        self.assertIn("A", result.requested_counts)
        self.assertIsInstance(result.requested_counts["A"], int)


if __name__ == "__main__":
    unittest.main(verbosity=2)

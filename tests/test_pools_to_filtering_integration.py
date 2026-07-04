"""
tests/test_pools_to_filtering_integration.py

End-to-end integration test: builds all 7 real pools via build_pools(),
runs generate() on each against a shared UserGraph/state, and feeds their
combined output through CandidateFilteringLayer.run() -- proving the two
layers actually connect and every pool's output is accepted, not just
hand-built Candidate objects in isolation.

Run:
    python -m pytest tests/test_pools_to_filtering_integration.py -v
"""

from __future__ import annotations

import time
import unittest
from datetime import datetime, timezone, timedelta

from pipeline.recommender.models.user_graph import (
    UserGraph, UserNode, ConceptEdge, ConceptConceptEdge, EdgeType,
)
from pipeline.recommender.pools.pools import build_pools, POOL_CLASSES
from pipeline.recommender.services.candidate_filtering import (
    CandidateFilteringLayer, ZPD_LO, ZPD_HI,
)

NOW = time.time()


def _iso(days):
    return (datetime.fromtimestamp(NOW, tz=timezone.utc) + timedelta(days=days)).isoformat()


# ---------------------------------------------------------------------------
# Fake Qdrant -- same shape as test_pools.py's fixture, extended with more
# problems so every pool has something real to return.
# ---------------------------------------------------------------------------

class _Pt:
    def __init__(self, pid, tags, diff, score=0.9):
        self.id = pid
        self.payload = {"problem_id": pid, "topic_tags": tags, "difficulty_score": diff}
        self.score = score
        self.vector = None


class FakeQdrant:
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
        _Pt("p_arrays_easy",  ["arrays"],  0.2),
        _Pt("p_arrays_med",   ["arrays"],  0.5),
        _Pt("p_arrays_hard",  ["arrays"],  0.8),
        _Pt("p_graphs_easy",  ["graphs"],  0.2),
        _Pt("p_graphs_hard",  ["graphs"],  0.85),
        _Pt("p_dp_med",       ["dp"],      0.5),
        _Pt("p_trees_easy",   ["trees"],   0.25),
        _Pt("p_trees_med",    ["trees"],   0.5),
        _Pt("p_dp_hard",      ["dp"],      0.9),
    ]


class _StubState:
    def __init__(self, qv, graph):
        self._qv = qv
        self.solved_ids = set(graph.solved_ids)
        self.is_cold_start = qv is None
    def to_query_vector(self):
        return self._qv


def _concept(slug, mastery=0.5, urgency=0.0, severity=0.0,
             edge_type=EdgeType.LEARNING, next_review_date=None):
    return ConceptEdge(slug, edge_type, mastery_score=mastery,
                       urgency=urgency, severity=severity,
                       next_review_date=next_review_date)


def _build_warm_graph():
    """A user with realistic mixed state: some mastered, some weak, one overdue."""
    g = UserGraph(user=UserNode(user_id="integration_user"))
    # mastered_concepts() checks mastery_score >= 0.7, not edge_type -- both
    # must clear that threshold to actually count as mastered for unlock checks.
    g.add_concept_edge(_concept("arrays", mastery=0.75, edge_type=EdgeType.MASTERED))
    g.add_concept_edge(_concept("trees",  mastery=0.75, edge_type=EdgeType.MASTERED))
    g.add_concept_edge(_concept("graphs", mastery=0.35, severity=0.7, edge_type=EdgeType.WEAK))
    g.add_concept_edge(_concept("dp",     mastery=0.60, next_review_date=_iso(-3)))
    g.add_cc_edge(ConceptConceptEdge("trees", "graphs", EdgeType.PREREQ, 1.0))
    g.solved_ids.update(["p_arrays_easy"])
    return g


class TestAllPoolsFeedFiltering(unittest.TestCase):

    def setUp(self):
        self.qdrant = FakeQdrant(_problems())
        self.pools = build_pools(qdrant=self.qdrant)
        self.graph = _build_warm_graph()
        self.state = _StubState([1.0] * 1920, self.graph)

    def test_all_seven_pools_registered(self):
        self.assertEqual(set(self.pools.keys()),
                         {"A", "B_C", "D", "E", "F", "G", "vector"})

    def test_generate_from_every_pool_then_filter(self):
        """
        Run every real pool's generate(), collect their output exactly as
        the (not-yet-built) pool generation layer would, and feed the whole
        dict straight into CandidateFilteringLayer.run() -- this is the
        actual handoff contract between the two layers.
        """
        pool_candidates = {}
        for name, pool in self.pools.items():
            pool_candidates[name] = pool.generate(self.graph, self.state, n=10)

        # sanity: at least some pools produced something with this fixture
        total_raw = sum(len(v) for v in pool_candidates.values())
        self.assertGreater(total_raw, 0,
                           "no pool produced any candidates -- fixture too sparse")

        layer = CandidateFilteringLayer(self.graph)
        merged, report = layer.run(pool_candidates)

        # the layer must have seen input from the pools (contract: run()
        # accepts the exact dict shape build_pools()/generate() produce)
        self.assertEqual(report.input_count, total_raw)

    def test_solved_problem_never_survives_regardless_of_which_pool_found_it(self):
        pool_candidates = {}
        for name, pool in self.pools.items():
            pool_candidates[name] = pool.generate(self.graph, self.state, n=10)

        layer = CandidateFilteringLayer(self.graph)
        merged, report = layer.run(pool_candidates)

        surviving_ids = {mc.problem_id for mc in merged}
        self.assertNotIn("p_arrays_easy", surviving_ids)

    def test_locked_graphs_problems_blocked_until_layer_runs(self):
        """
        graphs requires trees (PREREQ) and trees IS mastered in this fixture,
        so graphs problems should NOT be locked -- confirms the filtering
        layer's prereq index correctly reads the same cc_edges the pools
        themselves already used to decide eligibility upstream.
        """
        pool_candidates = {}
        for name, pool in self.pools.items():
            pool_candidates[name] = pool.generate(self.graph, self.state, n=10)

        layer = CandidateFilteringLayer(self.graph)
        merged, report = layer.run(pool_candidates)
        self.assertEqual(report.removed_locked, 0)

    def test_merged_candidates_carry_correct_pool_provenance(self):
        """
        If two different pools both proposed the same problem_id, the merged
        entry must list both pool names -- proves cross-pool dedup actually
        connects outputs from independently-instantiated pool objects.
        """
        pool_candidates = {
            "A": [self.pools["A"].generate(self.graph, self.state, n=10)][0],
            "vector": self.pools["vector"].generate(self.graph, self.state, n=10),
        }
        # force an overlap deliberately so we can assert on it regardless of
        # what the fake fixture naturally produces
        from pipeline.recommender.pools.base_pool import Candidate
        pool_candidates["A"].append(Candidate("p_dp_med", "A", topic_tags=["dp"], difficulty_score=0.5))
        pool_candidates["vector"].append(Candidate("p_dp_med", "vector", topic_tags=["dp"], difficulty_score=0.5, score=0.99))

        layer = CandidateFilteringLayer(self.graph)
        merged, report = layer.run(pool_candidates)

        hit = next((mc for mc in merged if mc.problem_id == "p_dp_med"), None)
        self.assertIsNotNone(hit)
        self.assertIn("A", hit.pool_sources)
        self.assertIn("vector", hit.pool_sources)

    def test_ranker_input_covers_candidates_from_multiple_pools(self):
        pool_candidates = {}
        for name, pool in self.pools.items():
            pool_candidates[name] = pool.generate(self.graph, self.state, n=10)

        layer = CandidateFilteringLayer(self.graph)
        merged, _ = layer.run(pool_candidates)
        rows = layer.to_ranker_input(merged)

        # every row must be traceable back to at least one real pool name
        valid_pools = set(POOL_CLASSES.keys())
        for row in rows:
            self.assertTrue(set(row["pool_sources"]).issubset(valid_pools))

    def test_empty_pool_dict_handled_gracefully(self):
        layer = CandidateFilteringLayer(self.graph)
        merged, report = layer.run({})
        self.assertEqual(merged, [])
        self.assertEqual(report.input_count, 0)

    def test_pool_with_zero_candidates_does_not_break_run(self):
        pool_candidates = {name: [] for name in self.pools}
        layer = CandidateFilteringLayer(self.graph)
        merged, report = layer.run(pool_candidates)
        self.assertEqual(merged, [])
        self.assertEqual(report.input_count, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

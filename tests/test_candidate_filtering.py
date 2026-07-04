"""
tests/test_candidate_filtering.py

Tests the candidate filtering layer: per-pool filtering (solved, locked),
merge/dedup across pools, and the ZPD band filter -- all without a real
database or Qdrant.

Run:
    python -m pytest tests/test_candidate_filtering.py -v
"""

from __future__ import annotations

import unittest

from pipeline.recommender.models.user_graph import (
    UserGraph, UserNode, ConceptEdge, ConceptConceptEdge, EdgeType,
)
from pipeline.recommender.pools.base_pool import Candidate
from pipeline.recommender.services.candidate_filtering import (
    CandidateFilteringLayer, MergedCandidate, FilterReport,
    ZPD_LO, ZPD_HI, ZPD_OPTIMAL,
)


def _graph(concepts=None, solved=None, deprioritised=None, cc=None):
    g = UserGraph(user=UserNode(user_id="u1"))
    for c in (concepts or []):
        g.add_concept_edge(c)
    for pid in (solved or []):
        g.solved_ids.add(pid)
    for pid in (deprioritised or []):
        g.deprioritised_ids.add(pid)
    for e in (cc or []):
        g.add_cc_edge(e)
    return g


def _concept(slug, mastery=0.5, urgency=0.0, edge_type=EdgeType.LEARNING):
    return ConceptEdge(slug, edge_type, mastery_score=mastery, urgency=urgency)


def _cand(pid, pool, tags=None, difficulty=0.5, score=0.9):
    return Candidate(pid, pool, score=score, topic_tags=tags or [], difficulty_score=difficulty)


# ===========================================================================
# Per-pool filtering: solved / deprioritised / locked
# ===========================================================================

class TestPerPoolFiltering(unittest.TestCase):

    def test_solved_removed(self):
        g = _graph(solved=["p1"])
        layer = CandidateFilteringLayer(g)
        merged, report = layer.run({"A": [_cand("p1", "A", ["arrays"])]})
        self.assertEqual(merged, [])
        self.assertEqual(report.removed_solved, 1)

    def test_deprioritised_removed(self):
        g = _graph(deprioritised=["p1"])
        layer = CandidateFilteringLayer(g)
        merged, report = layer.run({"A": [_cand("p1", "A", ["arrays"])]})
        self.assertEqual(merged, [])
        self.assertEqual(report.removed_deprioritised, 1)

    def test_locked_removed(self):
        # "graphs" requires "trees" (PREREQ), user hasn't mastered trees
        g = _graph(
            concepts=[_concept("arrays", mastery=0.9, edge_type=EdgeType.MASTERED)],
            cc=[ConceptConceptEdge("trees", "graphs", EdgeType.PREREQ, 1.0)],
        )
        layer = CandidateFilteringLayer(g)
        merged, report = layer.run({"A": [_cand("p1", "A", ["graphs"], difficulty=0.5)]})
        self.assertEqual(merged, [])
        self.assertEqual(report.removed_locked, 1)

    def test_unlocked_when_prereq_mastered(self):
        g = _graph(
            concepts=[
                _concept("trees", mastery=0.9, edge_type=EdgeType.MASTERED),
                # mastery=0.6 vs difficulty=0.5 lands inside the ZPD band
                # (this test verifies unlock logic, not the ZPD filter --
                # test_zpd_filter tests below cover that separately)
                _concept("graphs", mastery=0.6),
            ],
            cc=[ConceptConceptEdge("trees", "graphs", EdgeType.PREREQ, 1.0)],
        )
        layer = CandidateFilteringLayer(g)
        merged, report = layer.run({"A": [_cand("p1", "A", ["graphs"], difficulty=0.5)]})
        self.assertEqual(report.removed_locked, 0)
        self.assertEqual(len(merged), 1)

    def test_no_prereq_never_locked(self):
        g = _graph(concepts=[_concept("arrays", mastery=0.5)])
        layer = CandidateFilteringLayer(g)
        merged, report = layer.run({"A": [_cand("p1", "A", ["arrays"], difficulty=0.5)]})
        self.assertEqual(report.removed_locked, 0)

    def test_candidate_with_no_tags_never_locked(self):
        g = _graph(cc=[ConceptConceptEdge("trees", "graphs", EdgeType.PREREQ, 1.0)])
        layer = CandidateFilteringLayer(g)
        merged, report = layer.run({"A": [_cand("p1", "A", tags=[], difficulty=0.5)]})
        self.assertEqual(report.removed_locked, 0)


# ===========================================================================
# Merge / dedup across pools
# ===========================================================================

class TestMergeDedup(unittest.TestCase):

    def test_same_problem_from_two_pools_merges(self):
        g = _graph(concepts=[_concept("arrays", mastery=0.68)])
        layer = CandidateFilteringLayer(g)
        merged, report = layer.run({
            "A":      [_cand("p1", "A", ["arrays"], difficulty=0.5)],
            "vector": [_cand("p1", "vector", ["arrays"], difficulty=0.5)],
        })
        self.assertEqual(len(merged), 1)
        self.assertEqual(report.removed_duplicates, 1)
        self.assertIn("A", merged[0].pool_sources)
        self.assertIn("vector", merged[0].pool_sources)

    def test_pool_count_reflects_number_of_sources(self):
        g = _graph(concepts=[_concept("arrays", mastery=0.68)])
        layer = CandidateFilteringLayer(g)
        merged, _ = layer.run({
            "A": [_cand("p1", "A", ["arrays"], difficulty=0.5)],
            "D": [_cand("p1", "D", ["arrays"], difficulty=0.5)],
            "G": [_cand("p1", "G", ["arrays"], difficulty=0.5)],
        })
        self.assertEqual(merged[0].pool_count, 3)

    def test_best_score_is_max_across_pools(self):
        g = _graph(concepts=[_concept("arrays", mastery=0.68)])
        layer = CandidateFilteringLayer(g)
        merged, _ = layer.run({
            "A":      [_cand("p1", "A", ["arrays"], difficulty=0.5, score=0.3)],
            "vector": [_cand("p1", "vector", ["arrays"], difficulty=0.5, score=0.95)],
        })
        self.assertAlmostEqual(merged[0].best_score, 0.95)

    def test_different_problems_stay_separate(self):
        g = _graph(concepts=[_concept("arrays", mastery=0.68)])
        layer = CandidateFilteringLayer(g)
        merged, report = layer.run({
            "A": [_cand("p1", "A", ["arrays"], difficulty=0.5)],
            "D": [_cand("p2", "D", ["arrays"], difficulty=0.5)],
        })
        self.assertEqual(len(merged), 2)
        self.assertEqual(report.removed_duplicates, 0)


# ===========================================================================
# ZPD band filter
# ===========================================================================

class TestZPDFilter(unittest.TestCase):

    def test_matched_mastery_and_difficulty_passes(self):
        # mastery == difficulty -> success ~0.5, just below ZPD_LO (0.55)
        # use mastery slightly above difficulty to land inside the band
        g = _graph(concepts=[_concept("arrays", mastery=0.55)])
        layer = CandidateFilteringLayer(g)
        merged, report = layer.run({"A": [_cand("p1", "A", ["arrays"], difficulty=0.5)]})
        self.assertEqual(len(merged), 1)
        self.assertGreaterEqual(merged[0].predicted_success, ZPD_LO)
        self.assertLessEqual(merged[0].predicted_success, ZPD_HI)

    def test_way_too_hard_filtered(self):
        # mastery far below difficulty -> success near 0
        g = _graph(concepts=[_concept("arrays", mastery=0.05)])
        layer = CandidateFilteringLayer(g)
        merged, report = layer.run({"A": [_cand("p1", "A", ["arrays"], difficulty=0.95)]})
        self.assertEqual(merged, [])
        self.assertEqual(report.removed_zpd, 1)

    def test_way_too_easy_filtered(self):
        # mastery far above difficulty -> success near 1
        g = _graph(concepts=[_concept("arrays", mastery=0.95)])
        layer = CandidateFilteringLayer(g)
        merged, report = layer.run({"A": [_cand("p1", "A", ["arrays"], difficulty=0.05)]})
        self.assertEqual(merged, [])
        self.assertEqual(report.removed_zpd, 1)

    def test_cold_start_defaults_to_optimal_and_passes(self):
        # no mastery data at all for this tag -> defaults to ZPD_OPTIMAL (0.68)
        g = _graph()   # no concept edges
        layer = CandidateFilteringLayer(g)
        merged, report = layer.run({"A": [_cand("p1", "A", ["arrays"], difficulty=0.5)]})
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0].predicted_success, ZPD_OPTIMAL)

    def test_no_tags_defaults_to_optimal(self):
        g = _graph(concepts=[_concept("arrays", mastery=0.9)])
        layer = CandidateFilteringLayer(g)
        merged, report = layer.run({"A": [_cand("p1", "A", tags=[], difficulty=0.5)]})
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0].predicted_success, ZPD_OPTIMAL)

    def test_custom_success_estimator_used(self):
        g = _graph(concepts=[_concept("arrays", mastery=0.5)])

        def always_optimal(mc, graph):
            return ZPD_OPTIMAL

        layer = CandidateFilteringLayer(g, success_estimator=always_optimal)
        merged, report = layer.run({"A": [_cand("p1", "A", ["arrays"], difficulty=0.99)]})
        # would normally be filtered as too hard, but custom estimator overrides
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0].predicted_success, ZPD_OPTIMAL)


# ===========================================================================
# Ranker bridge (Shraddha's recommendation engine integration point)
# ===========================================================================

class TestRankerBridge(unittest.TestCase):

    def test_to_ranker_input_shape(self):
        g = _graph(concepts=[_concept("arrays", mastery=0.6, urgency=0.3)])
        layer = CandidateFilteringLayer(g)
        merged, _ = layer.run({"A": [_cand("p1", "A", ["arrays"], difficulty=0.5)]})
        rows = layer.to_ranker_input(merged)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        for key in ("problem_id", "pool_sources", "pool_count", "topic_tags",
                    "difficulty_score", "avg_mastery", "max_urgency",
                    "predicted_success", "best_pool_score"):
            self.assertIn(key, row)

    def test_ranker_input_avg_mastery_matches_graph(self):
        g = _graph(concepts=[_concept("arrays", mastery=0.6)])
        layer = CandidateFilteringLayer(g)
        merged, _ = layer.run({"A": [_cand("p1", "A", ["arrays"], difficulty=0.5)]})
        rows = layer.to_ranker_input(merged)
        self.assertAlmostEqual(rows[0]["avg_mastery"], 0.6)

    def test_ranker_input_urgency_matches_graph(self):
        g = _graph(concepts=[_concept("arrays", mastery=0.6, urgency=0.4)])
        layer = CandidateFilteringLayer(g)
        merged, _ = layer.run({"A": [_cand("p1", "A", ["arrays"], difficulty=0.5)]})
        rows = layer.to_ranker_input(merged)
        self.assertAlmostEqual(rows[0]["max_urgency"], 0.4)

    def test_ranker_input_json_serialisable(self):
        import json
        g = _graph(concepts=[_concept("arrays", mastery=0.6)])
        layer = CandidateFilteringLayer(g)
        merged, _ = layer.run({"A": [_cand("p1", "A", ["arrays"], difficulty=0.5)]})
        rows = layer.to_ranker_input(merged)
        raw = json.dumps(rows)
        self.assertIsInstance(raw, str)


# ===========================================================================
# Report
# ===========================================================================

class TestFilterReport(unittest.TestCase):

    def test_report_counts_add_up(self):
        g = _graph(
            solved=["p_solved"],
            deprioritised=["p_deprio"],
            concepts=[_concept("arrays", mastery=0.68)],
        )
        layer = CandidateFilteringLayer(g)
        merged, report = layer.run({
            "A": [
                _cand("p_solved", "A", ["arrays"], difficulty=0.5),
                _cand("p_deprio", "A", ["arrays"], difficulty=0.5),
                _cand("p_ok", "A", ["arrays"], difficulty=0.5),
            ],
        })
        self.assertEqual(report.input_count, 3)
        self.assertEqual(report.removed_solved, 1)
        self.assertEqual(report.removed_deprioritised, 1)
        self.assertEqual(report.output_count, len(merged))

    def test_report_to_dict(self):
        g = _graph()
        layer = CandidateFilteringLayer(g)
        _, report = layer.run({"A": []})
        d = report.to_dict()
        self.assertIn("input_count", d)
        self.assertIn("output_count", d)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""
tests/test_feature_extractor.py

Tests training/feature_extractor.py against real (in-memory) UserGraph /
DifficultyPlan / MergedCandidate objects -- no database, Qdrant, or Redis.
Cross-checks every reused value against the actual recommender functions
(HeuristicRanker, AdaptiveDifficultyController) rather than hardcoded
expectations, so a future change to those formulas is caught here too.

Run:
    python -m pytest tests/test_feature_extractor.py -v
"""

from __future__ import annotations

import unittest

from pipeline.recommender.models.user_graph import (
    ConceptConceptEdge, ConceptEdge, EdgeType, UserGraph, UserNode,
)
from pipeline.recommender.services.adaptive_difficulty import (
    AdaptiveDifficultyController,
)
from pipeline.recommender.services.candidate_filtering import (
    CandidateFilteringLayer, MergedCandidate,
)
from pipeline.recommender.services.heuristic_ranker import HeuristicRanker
from training.feature_extractor import (
    _difficulty_alignment,
    extract_features,
    extract_graph_features,
    extract_pair_features,
    extract_pool_features,
    extract_problem_features,
    extract_user_features,
)
from training.feature_registry import build_default_registry


def _graph(concepts=None, solved=None, cc=None, user_id="u1"):
    g = UserGraph(user=UserNode(user_id=user_id))
    for c in (concepts or []):
        g.add_concept_edge(c)
    for pid in (solved or []):
        g.solved_ids.add(pid)
    for e in (cc or []):
        g.add_cc_edge(e)
    return g


def _concept(slug, mastery=0.5, urgency=0.0, severity=0.0, edge_type=EdgeType.LEARNING):
    return ConceptEdge(concept_slug=slug, edge_type=edge_type, mastery_score=mastery,
                        urgency=urgency, severity=severity)


def _candidate(problem_id="p1", pools=("A",), topic_tags=("arrays",),
               difficulty=0.5, best_score=0.0, predicted_success=0.68):
    return MergedCandidate(
        problem_id=problem_id, pool_sources=list(pools), best_score=best_score,
        topic_tags=list(topic_tags), difficulty_score=difficulty,
        predicted_success=predicted_success,
    )


class TestExtractUserFeatures(unittest.TestCase):

    def test_matches_difficulty_plan_directly(self):
        g = _graph([_concept("arrays", mastery=0.6), _concept("graphs", mastery=0.4)],
                    solved=["s1"])
        plan = AdaptiveDifficultyController(now=0).build_plan(g)
        f = extract_user_features(g, plan)
        self.assertEqual(f.cold_start, plan.is_cold_start)
        self.assertEqual(f.user_level, plan.level)
        self.assertAlmostEqual(f.average_mastery, plan.avg_mastery, places=5)
        self.assertEqual(f.weak_topic_count, plan.n_weak)
        self.assertEqual(f.urgent_topic_count, plan.n_urgent)
        self.assertEqual(f.overdue_review_count, plan.n_overdue)
        self.assertEqual(f.solved_count, 1)
        self.assertEqual(f.mastered_topic_count, len(g.mastered_concepts()))

    def test_mastery_variance_zero_for_uniform_mastery(self):
        g = _graph([_concept("a", mastery=0.5), _concept("b", mastery=0.5)])
        plan = AdaptiveDifficultyController(now=0).build_plan(g)
        f = extract_user_features(g, plan)
        self.assertAlmostEqual(f.mastery_variance, 0.0, places=6)

    def test_mastery_variance_positive_for_spread_mastery(self):
        g = _graph([_concept("a", mastery=0.1), _concept("b", mastery=0.9)])
        plan = AdaptiveDifficultyController(now=0).build_plan(g)
        f = extract_user_features(g, plan)
        self.assertGreater(f.mastery_variance, 0.0)

    def test_mastery_variance_zero_for_no_concepts(self):
        g = _graph()
        plan = AdaptiveDifficultyController(now=0).build_plan(g)
        f = extract_user_features(g, plan)
        self.assertEqual(f.mastery_variance, 0.0)

    def test_seeded_user_true_when_mastery_but_no_solves(self):
        g = _graph([_concept("arrays", mastery=0.5)], solved=None)
        plan = AdaptiveDifficultyController(now=0).build_plan(g)
        f = extract_user_features(g, plan)
        # seeded_user is feature_extractor.py's own computation (concept_edges
        # present, solved_ids empty) and must be True regardless of how
        # DifficultyPlan.is_cold_start currently classifies this graph --
        # that classification belongs to adaptive_difficulty.py, out of
        # scope for this test.
        self.assertTrue(f.seeded_user)

    def test_seeded_user_false_with_platform_solves(self):
        g = _graph([_concept("arrays", mastery=0.5)], solved=["p1"])
        plan = AdaptiveDifficultyController(now=0).build_plan(g)
        f = extract_user_features(g, plan)
        self.assertFalse(f.seeded_user)

    def test_seeded_user_false_for_genuinely_cold_start(self):
        g = _graph()
        plan = AdaptiveDifficultyController(now=0).build_plan(g)
        f = extract_user_features(g, plan)
        self.assertFalse(f.seeded_user)
        self.assertTrue(f.cold_start)


class TestExtractGraphFeatures(unittest.TestCase):

    def test_none_when_no_prereq_edges(self):
        g = _graph([_concept("arrays", mastery=0.8, edge_type=EdgeType.MASTERED)])
        f = extract_graph_features(g)
        self.assertIsNone(f.prerequisite_completion_ratio)

    def test_cooccurs_edges_do_not_count(self):
        g = _graph(
            [_concept("arrays", mastery=0.8, edge_type=EdgeType.MASTERED)],
            cc=[ConceptConceptEdge("arrays", "graphs", EdgeType.COOCCURS, 1.0)],
        )
        f = extract_graph_features(g)
        self.assertIsNone(f.prerequisite_completion_ratio)

    def test_ratio_reflects_mastered_prereq_sources(self):
        g = _graph(
            [_concept("arrays", mastery=0.8, edge_type=EdgeType.MASTERED),
             _concept("strings", mastery=0.2, edge_type=EdgeType.LEARNING)],
            cc=[
                ConceptConceptEdge("arrays", "trees", EdgeType.PREREQ, 1.0),
                ConceptConceptEdge("strings", "dp", EdgeType.PREREQ, 1.0),
            ],
        )
        f = extract_graph_features(g)
        # arrays is mastered (source of a PREREQ edge), strings is not -> 1/2
        self.assertAlmostEqual(f.prerequisite_completion_ratio, 0.5, places=6)


class TestExtractProblemFeatures(unittest.TestCase):

    def test_defaults_to_none_without_catalog_metadata(self):
        row = {"difficulty_score": 0.6, "topic_tags": ["arrays", "hash_map"]}
        f = extract_problem_features(row)
        self.assertEqual(f.difficulty_score, 0.6)
        self.assertEqual(f.topic_tag_count, 2)
        self.assertIsNone(f.company_tag_count)
        self.assertIsNone(f.frequency)
        self.assertIsNone(f.rating)
        self.assertIsNone(f.asked_by_faang)

    def test_populates_from_catalog_metadata(self):
        row = {"difficulty_score": 0.6, "topic_tags": ["arrays"]}
        meta = {"companies": ["Google", "Meta"], "frequency": 0.7, "rating": 0.9, "asked_by_faang": True}
        f = extract_problem_features(row, catalog_metadata=meta)
        self.assertEqual(f.company_tag_count, 2.0)
        self.assertAlmostEqual(f.frequency, 0.7)
        self.assertAlmostEqual(f.rating, 0.9)
        self.assertTrue(f.asked_by_faang)

    def test_empty_topic_tags_handled(self):
        row = {"difficulty_score": None, "topic_tags": []}
        f = extract_problem_features(row)
        self.assertEqual(f.topic_tag_count, 0)
        self.assertIsNone(f.difficulty_score)


class TestDifficultyAlignment(unittest.TestCase):

    def test_matches_manual_formula(self):
        self.assertAlmostEqual(_difficulty_alignment(0.5, 0.5), 1.0)
        self.assertAlmostEqual(_difficulty_alignment(0.2, 0.8), 1.0 - 0.6)

    def test_none_inputs_default_to_zero(self):
        self.assertEqual(_difficulty_alignment(None, 0.5), 0.0)
        self.assertEqual(_difficulty_alignment(0.5, None), 0.0)


class TestExtractPairFeatures(unittest.TestCase):

    def _graph_and_row(self, mastery=0.6, urgency=0.4, difficulty=0.5):
        g = _graph([_concept("arrays", mastery=mastery, urgency=urgency)])
        layer = CandidateFilteringLayer(g)
        mc = _candidate(topic_tags=["arrays"], difficulty=difficulty)
        row = layer.to_ranker_input([mc])[0]
        return g, row

    def test_sub_scores_match_heuristic_ranker_exactly(self):
        g, row = self._graph_and_row()
        ranker = HeuristicRanker()
        ranked = ranker.score_one(row)
        f = extract_pair_features(row, ranked)
        self.assertAlmostEqual(f.proximity, ranked.proximity_score, places=6)
        self.assertAlmostEqual(f.pool_agreement, ranked.pool_agreement, places=6)
        self.assertAlmostEqual(f.urgency_boost, ranked.urgency_boost, places=6)
        self.assertAlmostEqual(f.similarity_score, ranked.similarity_score, places=6)

    def test_mastery_difficulty_gap_sign(self):
        g, row = self._graph_and_row(mastery=0.8, difficulty=0.3)
        ranked = HeuristicRanker().score_one(row)
        f = extract_pair_features(row, ranked)
        self.assertGreater(f.mastery_difficulty_gap, 0)   # mastery > difficulty

        g2, row2 = self._graph_and_row(mastery=0.2, difficulty=0.9)
        ranked2 = HeuristicRanker().score_one(row2)
        f2 = extract_pair_features(row2, ranked2)
        self.assertLess(f2.mastery_difficulty_gap, 0)   # difficulty > mastery

    def test_gap_none_when_avg_mastery_missing(self):
        row = {"difficulty_score": 0.5, "avg_mastery": None, "max_urgency": None,
               "predicted_success": 0.68, "best_pool_score": 0.0}
        ranked = HeuristicRanker().score_one(row)
        f = extract_pair_features(row, ranked)
        self.assertIsNone(f.mastery_difficulty_gap)


class TestExtractPoolFeatures(unittest.TestCase):

    def test_boolean_flags_match_pool_sources(self):
        plan = AdaptiveDifficultyController(now=0).build_plan(_graph())
        mc = _candidate(pools=("A", "G"))
        f = extract_pool_features(mc, plan)
        self.assertTrue(f.from_pool_A)
        self.assertTrue(f.from_pool_G)
        self.assertFalse(f.from_pool_D)
        self.assertFalse(f.from_pool_vector)
        self.assertEqual(f.pool_count, 2)

    def test_max_pool_weight_matches_plan(self):
        g = _graph()
        plan = AdaptiveDifficultyController(now=0).build_plan(g)
        mc = _candidate(pools=("A", "D"))
        f = extract_pool_features(mc, plan)
        self.assertAlmostEqual(f.max_pool_weight, max(plan.weight_of("A"), plan.weight_of("D")), places=6)

    def test_zero_when_no_pool_sources(self):
        plan = AdaptiveDifficultyController(now=0).build_plan(_graph())
        mc = MergedCandidate(problem_id="p1", pool_sources=[], best_score=0.0,
                              topic_tags=[], difficulty_score=0.5)
        f = extract_pool_features(mc, plan)
        self.assertEqual(f.max_pool_weight, 0.0)
        self.assertEqual(f.pool_count, 0)


class TestExtractFeaturesIntegration(unittest.TestCase):

    def test_full_extraction_is_deterministic(self):
        g = _graph([_concept("arrays", mastery=0.6, urgency=0.3)], solved=["s1"])
        plan = AdaptiveDifficultyController(now=0).build_plan(g)
        mc1 = _candidate(topic_tags=["arrays"])
        mc2 = _candidate(topic_tags=["arrays"])

        result1 = extract_features(g, plan, mc1, query_id="q1", recommended_at=1000.0)
        result2 = extract_features(g, plan, mc2, query_id="q1", recommended_at=1000.0)

        self.assertEqual(result1.to_flat_dict(), result2.to_flat_dict())

    def test_computes_predicted_success_when_missing(self):
        g = _graph([_concept("arrays", mastery=0.6)])
        plan = AdaptiveDifficultyController(now=0).build_plan(g)
        mc = MergedCandidate(problem_id="p1", pool_sources=["A"], best_score=0.0,
                              topic_tags=["arrays"], difficulty_score=0.5,
                              predicted_success=None)
        self.assertIsNone(mc.predicted_success)
        extract_features(g, plan, mc)
        self.assertIsNotNone(mc.predicted_success)

    def test_flat_dict_keys_match_registry_exactly(self):
        """Every non-label registry name must appear in to_flat_dict()'s
        output, and to_flat_dict() must not produce any column the
        registry doesn't declare -- keeps the extractor and registry from
        silently drifting apart."""
        registry = build_default_registry()
        g = _graph([_concept("arrays", mastery=0.6, urgency=0.3)], solved=["s1"])
        plan = AdaptiveDifficultyController(now=0).build_plan(g)
        mc = _candidate(topic_tags=["arrays"])
        flat = extract_features(g, plan, mc, query_id="q1").to_flat_dict()

        registry_names = set(registry.names()) - {"label"}
        self.assertEqual(set(flat.keys()), registry_names)

    def test_identifiers_populated_correctly(self):
        g = _graph([_concept("arrays", mastery=0.6)], user_id="user_42")
        plan = AdaptiveDifficultyController(now=0).build_plan(g)
        mc = _candidate(problem_id="prob_7", topic_tags=["arrays"])
        result = extract_features(g, plan, mc, query_id="q1", recommended_at=500.0)
        self.assertEqual(result.query_id, "q1")
        self.assertEqual(result.candidate_id, "prob_7")
        self.assertEqual(result.user_id, "user_42")
        self.assertEqual(result.recommended_at, 500.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

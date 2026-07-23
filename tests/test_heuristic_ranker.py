"""
tests/test_heuristic_ranker.py

Tests the hand-tuned weighted heuristic ranker: proximity to ZPD_OPTIMAL,
pool agreement saturation, urgency boost, similarity clamp, weight
correctness, and top_k behaviour.

Run:
    python -m pytest tests/test_heuristic_ranker.py -v
"""

from __future__ import annotations

import unittest

from pipeline.recommender.services.heuristic_ranker import (
    HeuristicRanker, RankedCandidate, rank_top_k,
    ZPD_OPTIMAL, ZPD_LO, ZPD_HI, POOL_AGREEMENT_SATURATION,
    WEIGHT_PROXIMITY, WEIGHT_POOL_AGREE, WEIGHT_URGENCY, WEIGHT_SIMILARITY,
    WEIGHT_DIFFICULTY_ALIGNMENT,
)


def _row(pid, predicted_success=ZPD_OPTIMAL, pool_count=1,
         max_urgency=0.0, best_pool_score=0.0):
    return {
        "problem_id": pid,
        "pool_sources": ["A"] * pool_count,
        "pool_count": pool_count,
        "topic_tags": ["arrays"],
        "difficulty_score": 0.5,
        "avg_mastery": 0.6,
        "max_urgency": max_urgency,
        "predicted_success": predicted_success,
        "best_pool_score": best_pool_score,
    }


class TestWeightsSumToOne(unittest.TestCase):

    def test_default_weights_sum_to_one(self):
        """All five weights that score_one() actually applies must sum to
        1.0 -- WEIGHT_DIFFICULTY_ALIGNMENT is a real term in the score, not
        a bonus added on top of an already-complete 1.0."""
        total = (
            WEIGHT_PROXIMITY + WEIGHT_POOL_AGREE + WEIGHT_URGENCY
            + WEIGHT_SIMILARITY + WEIGHT_DIFFICULTY_ALIGNMENT
        )
        self.assertAlmostEqual(total, 1.0, places=6)


class TestProximityScore(unittest.TestCase):

    def test_exact_optimal_gets_max_proximity(self):
        ranker = HeuristicRanker()
        rc = ranker.score_one(_row("p1", predicted_success=ZPD_OPTIMAL))
        self.assertAlmostEqual(rc.proximity_score, 1.0, places=4)

    def test_band_edge_gets_lower_proximity(self):
        ranker = HeuristicRanker()
        rc_lo = ranker.score_one(_row("p1", predicted_success=ZPD_LO))
        rc_optimal = ranker.score_one(_row("p2", predicted_success=ZPD_OPTIMAL))
        self.assertLess(rc_lo.proximity_score, rc_optimal.proximity_score)

    def test_other_band_edge_also_lower(self):
        ranker = HeuristicRanker()
        rc_hi = ranker.score_one(_row("p1", predicted_success=ZPD_HI))
        rc_optimal = ranker.score_one(_row("p2", predicted_success=ZPD_OPTIMAL))
        self.assertLess(rc_hi.proximity_score, rc_optimal.proximity_score)

    def test_missing_predicted_success_defaults_to_optimal(self):
        ranker = HeuristicRanker()
        row = _row("p1")
        row["predicted_success"] = None
        rc = ranker.score_one(row)
        self.assertAlmostEqual(rc.proximity_score, 1.0, places=4)

    def test_proximity_never_negative(self):
        ranker = HeuristicRanker()
        # even a value technically outside the band (shouldn't happen given
        # upstream ZPD filtering, but defend anyway) shouldn't go negative
        rc = ranker.score_one(_row("p1", predicted_success=0.0))
        self.assertGreaterEqual(rc.proximity_score, 0.0)


class TestPoolAgreement(unittest.TestCase):

    def test_single_pool_partial_credit(self):
        ranker = HeuristicRanker()
        rc = ranker.score_one(_row("p1", pool_count=1))
        self.assertAlmostEqual(rc.pool_agreement, 1 / POOL_AGREEMENT_SATURATION, places=4)

    def test_saturation_point_gets_full_credit(self):
        ranker = HeuristicRanker()
        rc = ranker.score_one(_row("p1", pool_count=POOL_AGREEMENT_SATURATION))
        self.assertAlmostEqual(rc.pool_agreement, 1.0, places=4)

    def test_beyond_saturation_stays_capped_at_one(self):
        ranker = HeuristicRanker()
        rc = ranker.score_one(_row("p1", pool_count=7))
        self.assertAlmostEqual(rc.pool_agreement, 1.0, places=4)

    def test_more_pools_scores_higher_than_fewer(self):
        ranker = HeuristicRanker()
        rc_one = ranker.score_one(_row("p1", pool_count=1))
        rc_three = ranker.score_one(_row("p2", pool_count=3))
        self.assertGreater(rc_three.score, rc_one.score)


class TestUrgencyBoost(unittest.TestCase):

    def test_higher_urgency_scores_higher(self):
        ranker = HeuristicRanker()
        rc_low = ranker.score_one(_row("p1", max_urgency=0.1))
        rc_high = ranker.score_one(_row("p2", max_urgency=0.9))
        self.assertGreater(rc_high.score, rc_low.score)

    def test_missing_urgency_defaults_to_zero(self):
        ranker = HeuristicRanker()
        row = _row("p1")
        row["max_urgency"] = None
        rc = ranker.score_one(row)
        self.assertEqual(rc.urgency_boost, 0.0)


class TestSimilarityClamp(unittest.TestCase):

    def test_similarity_clamped_above_one(self):
        ranker = HeuristicRanker()
        rc = ranker.score_one(_row("p1", best_pool_score=1.5))
        self.assertLessEqual(rc.similarity_score, 1.0)

    def test_similarity_clamped_below_zero(self):
        ranker = HeuristicRanker()
        rc = ranker.score_one(_row("p1", best_pool_score=-0.3))
        self.assertGreaterEqual(rc.similarity_score, 0.0)

    def test_missing_similarity_defaults_to_zero(self):
        ranker = HeuristicRanker()
        row = _row("p1")
        row["best_pool_score"] = None
        rc = ranker.score_one(row)
        self.assertEqual(rc.similarity_score, 0.0)


class TestRankingOrder(unittest.TestCase):

    def test_rank_sorts_best_first(self):
        ranker = HeuristicRanker()
        rows = [
            _row("low", predicted_success=ZPD_LO, pool_count=1),
            _row("high", predicted_success=ZPD_OPTIMAL, pool_count=3, max_urgency=0.5),
        ]
        ranked = ranker.rank(rows)
        self.assertEqual(ranked[0].problem_id, "high")
        self.assertEqual(ranked[1].problem_id, "low")

    def test_top_k_limits_result_size(self):
        rows = [_row(f"p{i}") for i in range(20)]
        result = rank_top_k(rows, k=10)
        self.assertEqual(len(result), 10)

    def test_top_k_smaller_input_returns_all(self):
        rows = [_row(f"p{i}") for i in range(3)]
        result = rank_top_k(rows, k=10)
        self.assertEqual(len(result), 3)

    def test_empty_input_returns_empty(self):
        result = rank_top_k([], k=10)
        self.assertEqual(result, [])

    def test_top_k_dicts_have_rank_score(self):
        rows = [_row("p1")]
        result = rank_top_k(rows, k=10)
        self.assertIn("rank_score", result[0])
        self.assertIn("rank_components", result[0])

    def test_original_row_fields_preserved_in_output(self):
        rows = [_row("p1")]
        result = rank_top_k(rows, k=10)
        self.assertEqual(result[0]["problem_id"], "p1")
        self.assertIn("topic_tags", result[0])


class TestCustomWeights(unittest.TestCase):

    def test_custom_weights_change_ranking(self):
        """
        With urgency weighted to dominate everything else, a high-urgency/
        low-proximity candidate should outrank a perfect-proximity/zero-urgency
        one -- proves weights are actually used, not hardcoded internally.
        """
        ranker = HeuristicRanker(
            weight_proximity=0.05, weight_pool_agree=0.05,
            weight_urgency=0.85, weight_similarity=0.05,
        )
        rows = [
            _row("perfect_proximity", predicted_success=ZPD_OPTIMAL, max_urgency=0.0),
            _row("urgent", predicted_success=ZPD_LO, max_urgency=0.95),
        ]
        ranked = ranker.rank(rows)
        self.assertEqual(ranked[0].problem_id, "urgent")


if __name__ == "__main__":
    unittest.main(verbosity=2)

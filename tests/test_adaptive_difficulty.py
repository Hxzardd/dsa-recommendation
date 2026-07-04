"""
tests/test_adaptive_difficulty.py

Tests the adaptive difficulty controller without any database or Qdrant.
Builds UserGraph objects in memory and checks the resulting difficulty plan.

Run:
    python -m pytest tests/test_adaptive_difficulty.py -v
"""

from __future__ import annotations

import time
import unittest
from datetime import datetime, timezone, timedelta

from pipeline.recommender.models.user_graph import (
    UserGraph, UserNode, ConceptEdge, EdgeType,
)
from pipeline.recommender.services.adaptive_difficulty import (
    AdaptiveDifficultyController, build_difficulty_plan,
    POOLS, LOW_MASTERY, HIGH_MASTERY,
)


NOW = time.time()


def _iso(days_from_now: float) -> str:
    dt = datetime.fromtimestamp(NOW, tz=timezone.utc) + timedelta(days=days_from_now)
    return dt.isoformat()


def _graph(concepts=None, solved=None):
    g = UserGraph(user=UserNode(user_id="u1", username="tester"))
    for c in (concepts or []):
        g.add_concept_edge(c)
    for pid in (solved or []):
        g.solved_ids.add(pid)
    return g


def _concept(slug, mastery=0.5, urgency=0.0, severity=0.0,
             edge_type=EdgeType.LEARNING, next_review_date=None):
    return ConceptEdge(
        concept_slug=slug,
        edge_type=edge_type,
        mastery_score=mastery,
        urgency=urgency,
        severity=severity,
        next_review_date=next_review_date,
    )


class TestWeightsSumToOne(unittest.TestCase):

    def test_normalised_weights(self):
        g = _graph([_concept("arrays", mastery=0.5)], solved=["p1"])
        plan = build_difficulty_plan(g, now=NOW)
        total = sum(plan.weight_of(p) for p in POOLS)
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_cold_start_weights_sum_to_one(self):
        g = _graph()  # no concepts, no solves
        plan = build_difficulty_plan(g, now=NOW)
        total = sum(plan.weight_of(p) for p in POOLS)
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_all_pools_present(self):
        g = _graph([_concept("arrays", mastery=0.5)], solved=["p1"])
        plan = build_difficulty_plan(g, now=NOW)
        for p in POOLS:
            self.assertIn(p, plan.directives)


class TestMixSumsToOne(unittest.TestCase):

    def test_every_pool_mix_sums_to_one(self):
        g = _graph([_concept("arrays", mastery=0.5)], solved=["p1"])
        plan = build_difficulty_plan(g, now=NOW)
        for p in POOLS:
            mix = plan.mix_of(p)
            self.assertAlmostEqual(
                mix["easy"] + mix["medium"] + mix["hard"], 1.0, places=3,
                msg=f"pool {p} mix does not sum to 1",
            )


class TestLevelDetection(unittest.TestCase):

    def test_beginner_level(self):
        g = _graph([_concept("arrays", mastery=0.2)], solved=["p1"])
        plan = build_difficulty_plan(g, now=NOW)
        self.assertEqual(plan.level, "beginner")

    def test_mid_level(self):
        g = _graph([_concept("arrays", mastery=0.5)], solved=["p1"])
        plan = build_difficulty_plan(g, now=NOW)
        self.assertEqual(plan.level, "mid")

    def test_advanced_level(self):
        g = _graph([_concept("arrays", mastery=0.8)], solved=["p1"])
        plan = build_difficulty_plan(g, now=NOW)
        self.assertEqual(plan.level, "advanced")

    def test_avg_mastery_across_concepts(self):
        g = _graph([
            _concept("arrays", mastery=0.2),
            _concept("graphs", mastery=0.8),
        ], solved=["p1"])
        plan = build_difficulty_plan(g, now=NOW)
        self.assertAlmostEqual(plan.avg_mastery, 0.5, places=3)


class TestDifficultyMixShifts(unittest.TestCase):

    def test_beginner_mix_favours_easy(self):
        g = _graph([_concept("arrays", mastery=0.2)], solved=["p1"])
        plan = build_difficulty_plan(g, now=NOW)
        mix = plan.mix_of("A")
        self.assertGreater(mix["easy"], mix["hard"])

    def test_advanced_mix_favours_hard(self):
        g = _graph([_concept("arrays", mastery=0.85)], solved=["p1"])
        plan = build_difficulty_plan(g, now=NOW)
        mix = plan.mix_of("A")
        self.assertGreater(mix["hard"], mix["easy"])

    def test_stretch_pool_harder_than_weakness_pool(self):
        g = _graph([_concept("arrays", mastery=0.5)], solved=["p1"])
        plan = build_difficulty_plan(g, now=NOW)
        f_mix = plan.mix_of("F")   # stretch
        d_mix = plan.mix_of("D")   # weakness
        self.assertGreater(f_mix["hard"], d_mix["hard"])

    def test_weakness_pool_easier_than_stretch(self):
        g = _graph([_concept("arrays", mastery=0.5)], solved=["p1"])
        plan = build_difficulty_plan(g, now=NOW)
        d_mix = plan.mix_of("D")
        f_mix = plan.mix_of("F")
        self.assertGreater(d_mix["easy"], f_mix["easy"])


class TestReviewPressure(unittest.TestCase):

    def test_overdue_boosts_spaced_review_pool(self):
        # baseline: no overdue
        g_base = _graph([_concept("arrays", mastery=0.5)], solved=["p1"])
        plan_base = build_difficulty_plan(g_base, now=NOW)

        # overdue: next_review_date in the past
        g_over = _graph([
            _concept("arrays", mastery=0.5, next_review_date=_iso(-5)),
            _concept("graphs", mastery=0.5, next_review_date=_iso(-3)),
        ], solved=["p1"])
        plan_over = build_difficulty_plan(g_over, now=NOW)

        self.assertGreater(plan_over.n_overdue, 0)
        self.assertGreater(plan_over.weight_of("E"), plan_base.weight_of("E"))

    def test_urgent_concepts_boost_review(self):
        g = _graph([
            _concept("arrays", mastery=0.5, urgency=0.8),
            _concept("graphs", mastery=0.5, urgency=0.7),
        ], solved=["p1"])
        plan = build_difficulty_plan(g, now=NOW)
        self.assertEqual(plan.n_urgent, 2)

    def test_future_review_not_overdue(self):
        g = _graph([
            _concept("arrays", mastery=0.5, next_review_date=_iso(+5)),
        ], solved=["p1"])
        plan = build_difficulty_plan(g, now=NOW)
        self.assertEqual(plan.n_overdue, 0)


class TestWeaknessPressure(unittest.TestCase):

    def test_weak_concepts_boost_weakness_pool(self):
        g_base = _graph([_concept("arrays", mastery=0.5)], solved=["p1"])
        plan_base = build_difficulty_plan(g_base, now=NOW)

        g_weak = _graph([
            _concept("arrays", mastery=0.5, severity=0.8, edge_type=EdgeType.WEAK),
            _concept("graphs", mastery=0.5, severity=0.7, edge_type=EdgeType.WEAK),
        ], solved=["p1"])
        plan_weak = build_difficulty_plan(g_weak, now=NOW)

        self.assertEqual(plan_weak.n_weak, 2)
        self.assertGreater(plan_weak.weight_of("D"), plan_base.weight_of("D"))


class TestColdStart(unittest.TestCase):

    def test_cold_start_flagged(self):
        g = _graph()
        plan = build_difficulty_plan(g, now=NOW)
        self.assertTrue(plan.is_cold_start)

    def test_cold_start_favours_course_path(self):
        g = _graph()
        plan = build_difficulty_plan(g, now=NOW)
        # course path (A) should be the heaviest pool at cold start
        weights = {p: plan.weight_of(p) for p in POOLS}
        self.assertEqual(max(weights, key=weights.get), "A")

    def test_no_solves_is_cold_even_with_concepts(self):
        g = _graph([_concept("arrays", mastery=0.5)], solved=None)
        plan = build_difficulty_plan(g, now=NOW)
        self.assertTrue(plan.is_cold_start)


class TestSerialization(unittest.TestCase):

    def test_to_dict_structure(self):
        g = _graph([_concept("arrays", mastery=0.5)], solved=["p1"])
        plan = build_difficulty_plan(g, now=NOW)
        d = plan.to_dict()
        self.assertIn("pools", d)
        self.assertIn("level", d)
        self.assertIn("avg_mastery", d)
        for p in POOLS:
            self.assertIn(p, d["pools"])
            self.assertIn("weight", d["pools"][p])
            self.assertIn("mix", d["pools"][p])

    def test_to_dict_json_serialisable(self):
        import json
        g = _graph([_concept("arrays", mastery=0.5)], solved=["p1"])
        plan = build_difficulty_plan(g, now=NOW)
        raw = json.dumps(plan.to_dict())
        self.assertIsInstance(raw, str)


if __name__ == "__main__":
    unittest.main(verbosity=2)

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
    POOLS, LOW_MASTERY, MIX_BEGINNER,
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

    def test_seeded_user_with_no_platform_solves_is_not_cold_start(self):
        """
        Regression test: a user with imported LeetCode/Codeforces history
        (seeding_controller.py) has real concept_edges (mastery/HLR seeded
        from external history) but zero solved_ids -- solved_ids is
        populated exclusively from the platform `submission` table
        (UserGraphService._load_submissions), which seeding never touches.
        Such a user must NOT be classified cold-start: they have real
        per-topic mastery signal to build a level-appropriate plan from,
        even though they've never submitted anything on this platform yet.
        """
        g = _graph([_concept("arrays", mastery=0.5)], solved=None)
        plan = build_difficulty_plan(g, now=NOW)
        self.assertFalse(plan.is_cold_start)

    def test_seeded_user_gets_level_appropriate_weights_not_cold_start_weights(self):
        """A seeded user's pool weights should reflect their actual seeded
        mastery level (here: mid, since 0.5 is between LOW_MASTERY and
        HIGH_MASTERY), not the cold-start weight table."""
        g = _graph([_concept("arrays", mastery=0.5)], solved=None)
        plan = build_difficulty_plan(g, now=NOW)
        self.assertEqual(plan.level, "mid")
        self.assertFalse(plan.is_cold_start)

    def test_true_cold_start_mix_is_extremely_easy_weighted(self):
        """A user's very first-ever recommendations should require
        extremely low difficulty -- more so than the general beginner mix."""
        g = _graph()
        plan = build_difficulty_plan(g, now=NOW)
        mix = plan.mix_of("A")
        self.assertGreaterEqual(mix["easy"], 0.8)
        self.assertEqual(mix["hard"], 0.0)
        self.assertGreater(mix["easy"], MIX_BEGINNER[0])

    def test_second_and_third_question_ease_gradually_not_a_sudden_jump(self):
        """
        Regression test: _base_mix() used to be a flat plateau at
        MIX_BEGINNER (30% medium / 10% hard) across the ENTIRE
        0..LOW_MASTERY range, so the moment is_cold_start flipped False
        (right after the user's first solve), question 2 got the SAME
        medium/hard exposure as an almost-established 0.34-mastery
        beginner -- a sudden jump, not a gradual ramp. The mix should now
        ease in smoothly: each successive low-mastery step's medium share
        should be small and only gradually increasing, never jumping
        straight to a large fraction.
        """
        g_q1 = _graph()   # true cold start
        plan_q1 = build_difficulty_plan(g_q1, now=NOW)
        mix_q1 = plan_q1.mix_of("A")

        # question 2: one easy solve landed, mastery still very low
        g_q2 = _graph([_concept("arrays", mastery=0.20)], solved=["p1"])
        plan_q2 = build_difficulty_plan(g_q2, now=NOW)
        mix_q2 = plan_q2.mix_of("A")

        # question 3: mastery a little higher still
        g_q3 = _graph([_concept("arrays", mastery=0.27)], solved=["p1", "p2"])
        plan_q3 = build_difficulty_plan(g_q3, now=NOW)
        mix_q3 = plan_q3.mix_of("A")

        # monotonic, gradual easing -- easy share decreases step by step,
        # never cliffs down to MIX_BEGINNER's fixed 60% in one jump
        self.assertGreater(mix_q1["easy"], mix_q2["easy"])
        self.assertGreater(mix_q2["easy"], mix_q3["easy"])
        # medium share stays small and climbs gradually, not a jump to 30%
        self.assertLess(mix_q2["medium"], 0.30)
        self.assertLess(mix_q3["medium"], 0.30)
        # no hard problems at all this early
        self.assertLess(mix_q2["hard"], 0.10)
        self.assertLess(mix_q3["hard"], 0.15)

    def test_base_mix_reproduces_beginner_exactly_at_low_mastery_boundary(self):
        """At avg_mastery==LOW_MASTERY exactly, the three-segment
        interpolation should hand off cleanly to MIX_BEGINNER."""
        ctrl = AdaptiveDifficultyController()
        mix = ctrl._base_mix(LOW_MASTERY)
        for i in range(3):
            self.assertAlmostEqual(mix[i], MIX_BEGINNER[i], places=6)


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

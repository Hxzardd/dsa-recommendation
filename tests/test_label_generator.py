"""
tests/test_label_generator.py

Tests training/label_generator.py entirely with InMemoryOutcomeProvider --
no database connection needed.

Run:
    python -m pytest tests/test_label_generator.py -v
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from training.label_generator import (
    NEVER_ATTEMPTED,
    STRATEGIES,
    BinarySolvedLabelStrategy,
    InMemoryOutcomeProvider,
    LabelGenerator,
    RecommendationOutcome,
    TernaryEngagementLabelStrategy,
    get_strategy,
)


def _solved(attempt_count=1, first_attempt_at=100.0, solved_at=100.0):
    return RecommendationOutcome(
        was_attempted=True, attempt_count=attempt_count,
        first_attempt_at=first_attempt_at, solved=True, solved_at=solved_at,
    )


def _failed(attempt_count=2, first_attempt_at=100.0):
    return RecommendationOutcome(
        was_attempted=True, attempt_count=attempt_count,
        first_attempt_at=first_attempt_at, solved=False, solved_at=None,
    )


def _base_row(query_id="q1", candidate_id="p1", user_id="u1", recommended_at=50.0):
    return {"query_id": query_id, "candidate_id": candidate_id,
            "user_id": user_id, "recommended_at": recommended_at, "label": float("nan")}


class TestRecommendationOutcome(unittest.TestCase):

    def test_never_attempted_sentinel(self):
        self.assertFalse(NEVER_ATTEMPTED.was_attempted)
        self.assertFalse(NEVER_ATTEMPTED.solved)
        self.assertEqual(NEVER_ATTEMPTED.attempt_count, 0)


class TestInMemoryOutcomeProvider(unittest.TestCase):

    def test_returns_never_attempted_for_unknown_key(self):
        provider = InMemoryOutcomeProvider({})
        outcome = provider.get_outcome("u1", "p1", 50.0, 14.0)
        self.assertEqual(outcome, NEVER_ATTEMPTED)

    def test_returns_supplied_outcome(self):
        outcome = _solved()
        provider = InMemoryOutcomeProvider({("u1", "p1"): outcome})
        self.assertEqual(provider.get_outcome("u1", "p1", 50.0, 14.0), outcome)


class TestLabelStrategies(unittest.TestCase):

    def test_binary_solved(self):
        s = BinarySolvedLabelStrategy()
        self.assertEqual(s.compute_label(_solved()), 1.0)
        self.assertEqual(s.compute_label(_failed()), 0.0)
        self.assertEqual(s.compute_label(NEVER_ATTEMPTED), 0.0)
        self.assertEqual(s.max_label, 1)

    def test_ternary_engagement(self):
        s = TernaryEngagementLabelStrategy()
        self.assertEqual(s.compute_label(_solved()), 2.0)
        self.assertEqual(s.compute_label(_failed()), 1.0)
        self.assertEqual(s.compute_label(NEVER_ATTEMPTED), 0.0)
        self.assertEqual(s.max_label, 2)

    def test_get_strategy_returns_correct_instance(self):
        self.assertIsInstance(get_strategy("binary_solved"), BinarySolvedLabelStrategy)
        self.assertIsInstance(get_strategy("ternary_engagement"), TernaryEngagementLabelStrategy)

    def test_get_strategy_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            get_strategy("not_a_real_strategy")

    def test_all_registered_strategies_are_valid(self):
        for name, cls in STRATEGIES.items():
            instance = cls()
            self.assertEqual(instance.name, name)


class TestLabelDataframe(unittest.TestCase):

    def test_solved_ignored_and_failed_labeled_correctly(self):
        df = pd.DataFrame([
            _base_row(query_id="q1", candidate_id="solved_problem"),
            _base_row(query_id="q1", candidate_id="failed_problem"),
            _base_row(query_id="q1", candidate_id="ignored_problem"),
        ])
        provider = InMemoryOutcomeProvider({
            ("u1", "solved_problem"): _solved(),
            ("u1", "failed_problem"): _failed(),
        })
        gen = LabelGenerator(provider=provider)
        labeled = gen.label_dataframe(df)

        by_id = labeled.set_index("candidate_id")["label"].to_dict()
        self.assertEqual(by_id["solved_problem"], 2.0)
        self.assertEqual(by_id["failed_problem"], 1.0)
        self.assertEqual(by_id["ignored_problem"], 0.0)

    def test_diagnostic_columns_populated(self):
        df = pd.DataFrame([_base_row(candidate_id="solved_problem")])
        provider = InMemoryOutcomeProvider({("u1", "solved_problem"): _solved(attempt_count=3)})
        gen = LabelGenerator(provider=provider)
        labeled = gen.label_dataframe(df)
        self.assertTrue(labeled.loc[0, "_outcome_solved"])
        self.assertTrue(labeled.loc[0, "_outcome_attempted"])
        self.assertEqual(labeled.loc[0, "_outcome_attempt_count"], 3)

    def test_missing_required_columns_raises(self):
        df = pd.DataFrame([{"query_id": "q1"}])
        gen = LabelGenerator(provider=InMemoryOutcomeProvider({}))
        with self.assertRaises(ValueError):
            gen.label_dataframe(df)

    def test_nan_identifier_produces_nan_label_not_crash(self):
        df = pd.DataFrame([_base_row(recommended_at=float("nan"))])
        gen = LabelGenerator(provider=InMemoryOutcomeProvider({}))
        labeled = gen.label_dataframe(df)
        self.assertTrue(math.isnan(labeled.loc[0, "label"]))

    def test_does_not_mutate_input_dataframe(self):
        df = pd.DataFrame([_base_row(candidate_id="solved_problem")])
        provider = InMemoryOutcomeProvider({("u1", "solved_problem"): _solved()})
        gen = LabelGenerator(provider=provider)
        gen.label_dataframe(df)
        self.assertTrue(math.isnan(df.loc[0, "label"]))   # original untouched

    def test_deterministic_repeated_labeling(self):
        df = pd.DataFrame([
            _base_row(query_id="q1", candidate_id="p1"),
            _base_row(query_id="q1", candidate_id="p2"),
        ])
        provider = InMemoryOutcomeProvider({("u1", "p1"): _solved()})
        gen = LabelGenerator(provider=provider)
        labeled1 = gen.label_dataframe(df)
        labeled2 = gen.label_dataframe(df)
        pd.testing.assert_frame_equal(labeled1, labeled2)

    def test_repeated_recommendation_same_problem_different_events(self):
        """Same user+problem recommended in two different events (different
        query_id) -- e.g. a review pool re-surfacing it later -- must be
        labeled independently (correctly attributing each event's own
        outcome, not a shared/duplicated one), not treated as a duplicate
        row. A real (time-scoped) provider would return the historical
        solve only for the event it followed; this fake mirrors that by
        keying on recommended_at, unlike InMemoryOutcomeProvider's
        (user_id, problem_id)-only key, which can't express "the same
        problem had two different outcomes across two different events."""
        df = pd.DataFrame([
            _base_row(query_id="q1", candidate_id="p1", recommended_at=10.0),
            _base_row(query_id="q2", candidate_id="p1", recommended_at=999999.0),
        ])

        class TimeScopedProvider:
            def get_outcome(self, user_id, problem_id, recommended_at, window_days):
                if recommended_at == 10.0:
                    return _solved(first_attempt_at=20.0)   # solved shortly after the FIRST event
                return NEVER_ATTEMPTED   # nothing new happened after the second (re-)recommendation

        gen = LabelGenerator(provider=TimeScopedProvider())
        labeled = gen.label_dataframe(df)
        # both rows are valid, independently-labeled -- not flagged as duplicates
        self.assertEqual(gen.validate(labeled), [])
        by_query = labeled.set_index("query_id")["label"].to_dict()
        self.assertEqual(by_query["q1"], 2.0)   # solved
        self.assertEqual(by_query["q2"], 0.0)   # ignored this time around

    def test_window_days_forwarded_to_provider(self):
        seen_windows = []

        class RecordingProvider:
            def get_outcome(self, user_id, problem_id, recommended_at, window_days):
                seen_windows.append(window_days)
                return NEVER_ATTEMPTED

        gen = LabelGenerator(provider=RecordingProvider(), window_days=30.0)
        gen.label_dataframe(pd.DataFrame([_base_row()]))
        self.assertEqual(seen_windows, [30.0])


class TestValidate(unittest.TestCase):

    def _labeled(self, provider_map=None):
        df = pd.DataFrame([_base_row(query_id="q1", candidate_id="p1")])
        gen = LabelGenerator(provider=InMemoryOutcomeProvider(provider_map or {}))
        return gen, gen.label_dataframe(df)

    def test_valid_dataframe_has_no_problems(self):
        gen, labeled = self._labeled()
        self.assertEqual(gen.validate(labeled), [])

    def test_duplicate_query_candidate_pair_detected(self):
        gen, labeled = self._labeled()
        dup = pd.concat([labeled, labeled], ignore_index=True)
        problems = gen.validate(dup)
        self.assertTrue(any("duplicate" in p for p in problems))

    def test_missing_identifier_detected(self):
        gen, labeled = self._labeled()
        labeled.loc[0, "user_id"] = None
        problems = gen.validate(labeled)
        self.assertTrue(any("user_id" in p for p in problems))

    def test_non_positive_timestamp_detected(self):
        gen, labeled = self._labeled()
        labeled.loc[0, "recommended_at"] = 0.0
        problems = gen.validate(labeled)
        self.assertTrue(any("non-positive" in p for p in problems))

    def test_impossible_attribution_detected(self):
        gen, labeled = self._labeled()
        # attempt supposedly happened BEFORE the recommendation
        labeled.loc[0, "_outcome_first_attempt_at"] = labeled.loc[0, "recommended_at"] - 100
        problems = gen.validate(labeled)
        self.assertTrue(any("impossible attribution" in p for p in problems))

    def test_inconsistent_recommendation_event_detected(self):
        gen, labeled = self._labeled()
        second = labeled.copy()
        second.loc[0, "candidate_id"] = "p2"
        second.loc[0, "user_id"] = "different_user"   # same query_id, different user
        combined = pd.concat([labeled, second], ignore_index=True)
        problems = gen.validate(combined)
        self.assertTrue(any("inconsistent" in p for p in problems))

    def test_out_of_range_label_detected(self):
        gen, labeled = self._labeled()
        labeled.loc[0, "label"] = 99.0
        problems = gen.validate(labeled)
        self.assertTrue(any("outside the valid range" in p for p in problems))


class TestComputeStatistics(unittest.TestCase):

    def test_distribution_and_percentages(self):
        df = pd.DataFrame([
            _base_row(query_id="q1", candidate_id="solved"),
            _base_row(query_id="q1", candidate_id="failed"),
            _base_row(query_id="q1", candidate_id="ignored1"),
            _base_row(query_id="q1", candidate_id="ignored2"),
        ])
        provider = InMemoryOutcomeProvider({
            ("u1", "solved"): _solved(),
            ("u1", "failed"): _failed(),
        })
        gen = LabelGenerator(provider=provider)
        labeled = gen.label_dataframe(df)
        stats = gen.compute_statistics(labeled)

        self.assertEqual(stats["total_rows"], 4)
        self.assertEqual(stats["percent_solved"], 25.0)
        self.assertEqual(stats["percent_attempted_not_solved"], 25.0)
        self.assertEqual(stats["percent_ignored"], 50.0)
        self.assertEqual(stats["unlabeled_rows"], 0)
        self.assertEqual(stats["label_distribution"], {"0.0": 2, "1.0": 1, "2.0": 1})
        self.assertIsNotNone(stats["label_imbalance_ratio"])

    def test_empty_dataframe(self):
        gen = LabelGenerator(provider=InMemoryOutcomeProvider({}))
        empty = pd.DataFrame(columns=["query_id", "candidate_id", "user_id",
                                       "recommended_at", "label"])
        stats = gen.compute_statistics(empty)
        self.assertEqual(stats["total_rows"], 0)
        self.assertIsNone(stats["label_imbalance_ratio"])


class TestExport(unittest.TestCase):

    def test_writes_parquet_and_statistics_json(self):
        train_df = pd.DataFrame([
            _base_row(query_id="q1", candidate_id="p1"),
            _base_row(query_id="q1", candidate_id="p2"),
        ])
        val_df = pd.DataFrame([_base_row(query_id="q2", candidate_id="p3")])
        provider = InMemoryOutcomeProvider({("u1", "p1"): _solved()})
        gen = LabelGenerator(provider=provider)

        with tempfile.TemporaryDirectory() as td:
            train_path = Path(td) / "labeled_train.parquet"
            val_path = Path(td) / "labeled_validation.parquet"
            stats_path = Path(td) / "label_statistics.json"
            result = gen.export(train_df, val_df, train_path, val_path, stats_path)

            self.assertTrue(train_path.exists())
            self.assertTrue(val_path.exists())
            self.assertTrue(stats_path.exists())
            reloaded = pd.read_parquet(train_path)
            self.assertEqual(len(reloaded), 2)
            self.assertIn("train", result.statistics)
            self.assertIn("validation", result.statistics)

    def test_raises_and_writes_nothing_on_invalid_data(self):
        train_df = pd.DataFrame([_base_row(query_id="q1", candidate_id="p1")])
        train_df = pd.concat([train_df, train_df], ignore_index=True)   # force duplicate
        gen = LabelGenerator(provider=InMemoryOutcomeProvider({}))

        with tempfile.TemporaryDirectory() as td:
            train_path = Path(td) / "labeled_train.parquet"
            val_path = Path(td) / "labeled_validation.parquet"
            stats_path = Path(td) / "label_statistics.json"
            with self.assertRaises(ValueError):
                gen.export(train_df, pd.DataFrame([_base_row(query_id="q2")]),
                           train_path, val_path, stats_path)
            self.assertFalse(train_path.exists())
            self.assertFalse(val_path.exists())
            self.assertFalse(stats_path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)

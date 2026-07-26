"""
tests/test_dataset_generator.py

Tests training/dataset_generator.py end-to-end against real (in-memory)
UserGraph / DifficultyPlan / MergedCandidate objects -- no database,
Qdrant, or Redis. Exercises actual parquet read/write via tempfile.

Run:
    python -m pytest tests/test_dataset_generator.py -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pipeline.recommender.models.user_graph import ConceptEdge, EdgeType, UserGraph, UserNode
from pipeline.recommender.services.adaptive_difficulty import AdaptiveDifficultyController
from pipeline.recommender.services.candidate_filtering import MergedCandidate
from training.dataset_generator import DatasetGenerator, RecommendationEvent
from training.feature_registry import build_default_registry


def _graph(concepts=None, solved=None, user_id="u1"):
    g = UserGraph(user=UserNode(user_id=user_id))
    for c in (concepts or []):
        g.add_concept_edge(c)
    for pid in (solved or []):
        g.solved_ids.add(pid)
    return g


def _concept(slug, mastery=0.5, urgency=0.0):
    return ConceptEdge(concept_slug=slug, edge_type=EdgeType.LEARNING,
                        mastery_score=mastery, urgency=urgency)


def _candidate(problem_id, pools=("A",), topic_tags=("arrays",), difficulty=0.5):
    return MergedCandidate(
        problem_id=problem_id, pool_sources=list(pools), best_score=0.0,
        topic_tags=list(topic_tags), difficulty_score=difficulty,
        predicted_success=0.68,
    )


def _event(query_id="q1", user_id="u1", recommended_at=1000.0, n_candidates=3,
           mastery=0.6, solved=("s1",)):
    g = _graph([_concept("arrays", mastery=mastery)], solved=list(solved), user_id=user_id)
    plan = AdaptiveDifficultyController(now=0).build_plan(g)
    candidates = [
        _candidate(f"p{i}", pools=("A",) if i % 2 == 0 else ("G",))
        for i in range(n_candidates)
    ]
    return RecommendationEvent(
        query_id=query_id, user_id=user_id, recommended_at=recommended_at,
        graph=g, plan=plan, candidates=candidates,
    )


class TestGenerateRows(unittest.TestCase):

    def test_one_row_per_candidate(self):
        gen = DatasetGenerator()
        event = _event(n_candidates=4)
        rows = gen.generate_rows(event)
        self.assertEqual(len(rows), 4)

    def test_every_row_carries_the_event_query_id(self):
        gen = DatasetGenerator()
        event = _event(query_id="q_abc", n_candidates=3)
        rows = gen.generate_rows(event)
        self.assertTrue(all(r["query_id"] == "q_abc" for r in rows))

    def test_candidate_ids_match_input_candidates(self):
        gen = DatasetGenerator()
        event = _event(n_candidates=3)
        rows = gen.generate_rows(event)
        self.assertEqual([r["candidate_id"] for r in rows], ["p0", "p1", "p2"])

    def test_label_is_placeholder_nan(self):
        gen = DatasetGenerator()
        event = _event(n_candidates=2)
        rows = gen.generate_rows(event)
        for r in rows:
            self.assertTrue(pd.isna(r["label"]))

    def test_empty_candidate_list_produces_no_rows(self):
        gen = DatasetGenerator()
        event = _event(n_candidates=0)
        self.assertEqual(gen.generate_rows(event), [])


class TestGenerateDataframe(unittest.TestCase):

    def test_row_count_matches_total_candidates_across_events(self):
        gen = DatasetGenerator()
        events = [_event(query_id="q1", n_candidates=3), _event(query_id="q2", n_candidates=5)]
        df = gen.generate_dataframe(events)
        self.assertEqual(len(df), 8)

    def test_columns_match_registry_exactly(self):
        gen = DatasetGenerator()
        df = gen.generate_dataframe([_event()])
        registry_names = set(build_default_registry().names())
        self.assertEqual(set(df.columns), registry_names)
        # column ORDER must also match the registry's deterministic order
        self.assertEqual(list(df.columns), gen.registry.names())

    def test_label_column_is_float64_and_all_nan(self):
        gen = DatasetGenerator()
        df = gen.generate_dataframe([_event(n_candidates=2)])
        self.assertEqual(df["label"].dtype, "float64")
        self.assertTrue(df["label"].isna().all())

    def test_deterministic_regardless_of_event_input_order(self):
        gen = DatasetGenerator()
        e1 = _event(query_id="q1", user_id="u1", recommended_at=100.0)
        e2 = _event(query_id="q2", user_id="u2", recommended_at=50.0)
        df_a = gen.generate_dataframe([e1, e2])
        df_b = gen.generate_dataframe([e2, e1])
        pd.testing.assert_frame_equal(df_a, df_b)

    def test_empty_events_produces_empty_dataframe_with_correct_columns(self):
        gen = DatasetGenerator()
        df = gen.generate_dataframe([])
        self.assertEqual(len(df), 0)
        self.assertEqual(list(df.columns), gen.registry.names())

    def test_repeated_generation_is_byte_identical(self):
        gen = DatasetGenerator()
        events = [_event(query_id="q1"), _event(query_id="q2", user_id="u2")]
        df1 = gen.generate_dataframe(events)
        df2 = gen.generate_dataframe(events)
        pd.testing.assert_frame_equal(df1, df2)


class TestValidateSchema(unittest.TestCase):

    def test_valid_dataframe_has_no_problems(self):
        gen = DatasetGenerator()
        df = gen.generate_dataframe([_event()])
        self.assertEqual(gen.validate_schema(df), [])

    def test_missing_column_detected(self):
        gen = DatasetGenerator()
        df = gen.generate_dataframe([_event()]).drop(columns=["difficulty_score"])
        problems = gen.validate_schema(df)
        self.assertTrue(any("difficulty_score" in p for p in problems))

    def test_extra_column_detected(self):
        gen = DatasetGenerator()
        df = gen.generate_dataframe([_event()])
        df["totally_unregistered_column"] = 1
        problems = gen.validate_schema(df)
        self.assertTrue(any("totally_unregistered_column" in p for p in problems))


class TestSplitTrainValidation(unittest.TestCase):

    def test_no_query_id_appears_in_both_splits(self):
        gen = DatasetGenerator()
        events = [_event(query_id=f"q{i}", n_candidates=2) for i in range(10)]
        df = gen.generate_dataframe(events)
        train_df, val_df = gen.split_train_validation(df, validation_fraction=0.3, seed=1)
        train_ids = set(train_df["query_id"])
        val_ids = set(val_df["query_id"])
        self.assertEqual(train_ids & val_ids, set())

    def test_all_rows_accounted_for(self):
        gen = DatasetGenerator()
        events = [_event(query_id=f"q{i}", n_candidates=3) for i in range(6)]
        df = gen.generate_dataframe(events)
        train_df, val_df = gen.split_train_validation(df, validation_fraction=0.5, seed=1)
        self.assertEqual(len(train_df) + len(val_df), len(df))

    def test_deterministic_given_same_seed(self):
        gen = DatasetGenerator()
        events = [_event(query_id=f"q{i}", n_candidates=2) for i in range(10)]
        df = gen.generate_dataframe(events)
        train1, val1 = gen.split_train_validation(df, validation_fraction=0.3, seed=7)
        train2, val2 = gen.split_train_validation(df, validation_fraction=0.3, seed=7)
        self.assertEqual(set(train1["query_id"]), set(train2["query_id"]))
        self.assertEqual(set(val1["query_id"]), set(val2["query_id"]))

    def test_query_group_never_split_within_itself(self):
        """Every row of a given query_id must land entirely in one split."""
        gen = DatasetGenerator()
        events = [_event(query_id=f"q{i}", n_candidates=4) for i in range(8)]
        df = gen.generate_dataframe(events)
        train_df, val_df = gen.split_train_validation(df, validation_fraction=0.25, seed=3)
        for qid in df["query_id"].unique():
            in_train = (train_df["query_id"] == qid).sum()
            in_val = (val_df["query_id"] == qid).sum()
            self.assertTrue(in_train == 0 or in_val == 0,
                             f"query_id {qid} split across both train and validation")


class TestExport(unittest.TestCase):

    def test_writes_readable_parquet_files(self):
        gen = DatasetGenerator()
        events = [_event(query_id=f"q{i}", n_candidates=3) for i in range(4)]
        df = gen.generate_dataframe(events)
        with tempfile.TemporaryDirectory() as td:
            train_path = Path(td) / "train.parquet"
            val_path = Path(td) / "validation.parquet"
            result = gen.export(df, validation_fraction=0.25, seed=1,
                                 train_path=train_path, validation_path=val_path)
            self.assertTrue(train_path.exists())
            self.assertTrue(val_path.exists())

            reloaded_train = pd.read_parquet(train_path)
            reloaded_val = pd.read_parquet(val_path)
            self.assertEqual(len(reloaded_train) + len(reloaded_val), len(df))
            self.assertEqual(result.train_query_count + result.validation_query_count,
                              df["query_id"].nunique())

    def test_raises_and_writes_nothing_on_schema_mismatch(self):
        gen = DatasetGenerator()
        df = gen.generate_dataframe([_event()]).drop(columns=["difficulty_score"])
        with tempfile.TemporaryDirectory() as td:
            train_path = Path(td) / "train.parquet"
            val_path = Path(td) / "validation.parquet"
            with self.assertRaises(ValueError):
                gen.export(df, train_path=train_path, validation_path=val_path)
            self.assertFalse(train_path.exists())
            self.assertFalse(val_path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)

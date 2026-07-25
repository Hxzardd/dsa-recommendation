"""
tests/test_feature_registry.py

Tests training/feature_registry.py: the canonical feature declarations for
the LightGBM LambdaRank training dataset. No database, Qdrant, or pandas
needed -- this is pure metadata.

Run:
    python -m pytest tests/test_feature_registry.py -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pipeline.recommender.services.adaptive_difficulty import POOLS
from training.feature_registry import (
    ComputationalCost,
    DataType,
    FeatureGroup,
    FeatureRegistry,
    FeatureSpec,
    LeakageRisk,
    REJECTED_FEATURES,
    build_default_registry,
)


class TestFeatureSpecValidation(unittest.TestCase):

    def test_engineered_feature_without_justification_raises(self):
        with self.assertRaises(ValueError):
            FeatureSpec(
                name="bad", group=FeatureGroup.USER, dtype=DataType.NUMERICAL,
                source_file="x.py", source_function="y",
                reused=False, reason_for_inclusion="because",
            )

    def test_reused_feature_without_justification_is_fine(self):
        spec = FeatureSpec(
            name="ok", group=FeatureGroup.USER, dtype=DataType.NUMERICAL,
            source_file="x.py", source_function="y",
            reused=True, reason_for_inclusion="because",
        )
        self.assertIsNone(spec.engineering_justification)

    def test_high_leakage_risk_in_model_matrix_raises(self):
        with self.assertRaises(ValueError):
            FeatureSpec(
                name="leaky", group=FeatureGroup.PAIR, dtype=DataType.NUMERICAL,
                source_file="x.py", source_function="y",
                reused=True, reason_for_inclusion="because",
                leakage_risk=LeakageRisk.HIGH, include_in_model_matrix=True,
            )

    def test_high_leakage_risk_excluded_from_matrix_is_fine(self):
        spec = FeatureSpec(
            name="leaky_diagnostic", group=FeatureGroup.DIAGNOSTIC, dtype=DataType.NUMERICAL,
            source_file="x.py", source_function="y",
            reused=True, reason_for_inclusion="comparison only",
            leakage_risk=LeakageRisk.MEDIUM, include_in_model_matrix=False,
        )
        self.assertFalse(spec.include_in_model_matrix)

    def test_to_dict_round_trips_every_field(self):
        spec = FeatureSpec(
            name="f1", group=FeatureGroup.PROBLEM, dtype=DataType.NUMERICAL,
            source_file="a.py", source_function="b",
            reused=True, reason_for_inclusion="r",
            expected_range="[0,1]", computational_cost=ComputationalCost.LOW,
        )
        d = spec.to_dict()
        self.assertEqual(d["name"], "f1")
        self.assertEqual(d["computational_cost"], "low")
        self.assertEqual(d["expected_range"], "[0,1]")


class TestFeatureRegistry(unittest.TestCase):

    def test_duplicate_registration_raises(self):
        r = FeatureRegistry()
        spec = FeatureSpec(
            name="dup", group=FeatureGroup.USER, dtype=DataType.NUMERICAL,
            source_file="a.py", source_function="b", reused=True, reason_for_inclusion="r",
        )
        r.register(spec)
        with self.assertRaises(ValueError):
            r.register(spec)

    def test_by_group_filters_correctly(self):
        r = FeatureRegistry()
        r.register(FeatureSpec(
            name="u1", group=FeatureGroup.USER, dtype=DataType.NUMERICAL,
            source_file="a.py", source_function="b", reused=True, reason_for_inclusion="r",
        ))
        r.register(FeatureSpec(
            name="p1", group=FeatureGroup.PROBLEM, dtype=DataType.NUMERICAL,
            source_file="a.py", source_function="b", reused=True, reason_for_inclusion="r",
        ))
        self.assertEqual([f.name for f in r.by_group(FeatureGroup.USER)], ["u1"])
        self.assertEqual([f.name for f in r.by_group(FeatureGroup.PROBLEM)], ["p1"])

    def test_model_matrix_features_excludes_identifiers(self):
        r = FeatureRegistry()
        r.register(FeatureSpec(
            name="qid", group=FeatureGroup.IDENTIFIER, dtype=DataType.IDENTIFIER,
            source_file="a.py", source_function="b", reused=True,
            reason_for_inclusion="r", include_in_model_matrix=False,
        ))
        r.register(FeatureSpec(
            name="feat", group=FeatureGroup.USER, dtype=DataType.NUMERICAL,
            source_file="a.py", source_function="b", reused=True, reason_for_inclusion="r",
        ))
        matrix = [f.name for f in r.model_matrix_features()]
        self.assertNotIn("qid", matrix)
        self.assertIn("feat", matrix)

    def test_export_writes_valid_json(self):
        r = FeatureRegistry()
        r.register(FeatureSpec(
            name="f1", group=FeatureGroup.USER, dtype=DataType.NUMERICAL,
            source_file="a.py", source_function="b", reused=True, reason_for_inclusion="r",
        ))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "registry.json"
            r.export(path)
            self.assertTrue(path.exists())
            with open(path) as fh:
                data = json.load(fh)
            self.assertEqual(data["feature_count"], 1)
            self.assertEqual(data["features"][0]["name"], "f1")
            self.assertIn("rejected_features", data)

    def test_names_are_sorted_and_deterministic(self):
        r = FeatureRegistry()
        for name in ("zebra", "apple", "mango"):
            r.register(FeatureSpec(
                name=name, group=FeatureGroup.USER, dtype=DataType.NUMERICAL,
                source_file="a.py", source_function="b", reused=True, reason_for_inclusion="r",
            ))
        self.assertEqual(r.names(), ["apple", "mango", "zebra"])


class TestDefaultRegistry(unittest.TestCase):
    """The actual canonical registry used by the rest of the training pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.registry = build_default_registry()

    def test_validate_reports_no_problems(self):
        problems = self.registry.validate()
        self.assertEqual(problems, [], f"Registry has unresolved problems: {problems}")

    def test_identifiers_excluded_from_model_matrix(self):
        matrix_names = {f.name for f in self.registry.model_matrix_features()}
        for identifier in ("query_id", "candidate_id", "user_id", "recommended_at", "label"):
            self.assertIn(identifier, self.registry.names())
            self.assertNotIn(identifier, matrix_names)

    def test_diagnostic_rank_score_excluded_from_model_matrix(self):
        spec = self.registry.get("current_heuristic_rank_score")
        self.assertFalse(spec.include_in_model_matrix)
        self.assertEqual(spec.group, FeatureGroup.DIAGNOSTIC)

    def test_engineered_features_all_have_justification(self):
        for spec in self.registry.all():
            if not spec.reused:
                self.assertTrue(
                    spec.engineering_justification,
                    f"{spec.name} is engineered but has no justification",
                )

    def test_pool_boolean_features_match_live_pool_registry(self):
        """Regression guard: if adaptive_difficulty.POOLS ever changes, the
        registry's from_pool_* features must be updated to match -- this
        test fails loudly instead of silently going stale."""
        expected = {f"from_pool_{p}" for p in POOLS}
        actual = {f.name for f in self.registry.by_group(FeatureGroup.POOL) if f.name.startswith("from_pool_")}
        self.assertEqual(expected, actual)

    def test_no_high_or_medium_leakage_risk_in_model_matrix(self):
        for spec in self.registry.model_matrix_features():
            self.assertNotIn(
                spec.leakage_risk, (LeakageRisk.HIGH, LeakageRisk.MEDIUM),
                f"{spec.name} has {spec.leakage_risk.value} leakage risk but is trainable",
            )

    def test_rejected_features_have_reasons(self):
        self.assertGreater(len(REJECTED_FEATURES), 0)
        for rejected in REJECTED_FEATURES:
            self.assertTrue(rejected.reason.strip())

    def test_raw_state_vector_and_review_load_are_rejected_not_registered(self):
        rejected_names = {r.name for r in REJECTED_FEATURES}
        self.assertIn("raw_user_state_vector_1920d", rejected_names)
        self.assertIn("review_load", rejected_names)
        self.assertIn("topic_entropy", rejected_names)
        # and confirm they were NOT silently also registered as real features
        self.assertNotIn("review_load", self.registry.names())
        self.assertNotIn("topic_entropy", self.registry.names())

    def test_export_produces_json_matching_registry_size(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "registry.json"
            self.registry.export(path)
            with open(path) as fh:
                data = json.load(fh)
            self.assertEqual(data["feature_count"], len(self.registry))
            self.assertEqual(
                data["model_matrix_feature_count"],
                len(self.registry.model_matrix_features()),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

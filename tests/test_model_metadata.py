"""
tests/test_model_metadata.py

Tests training/model_metadata.py: metadata generation (build_metadata),
save/load round-tripping, dataset-version/model-version determinism, and
the missing-metadata-file fallback (load_metadata returns None, never a
fabricated placeholder dict).

Run:
    python -m pytest tests/test_model_metadata.py -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training import model_metadata as mm
from training.feature_registry import build_default_registry


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


class TestComputeDatasetVersion(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.train_path = Path(self._tmpdir.name) / "train.parquet"
        self.validation_path = Path(self._tmpdir.name) / "validation.parquet"

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_same_file_bytes_produce_same_dataset_version(self):
        _write_bytes(self.train_path, b"train-content-A")
        _write_bytes(self.validation_path, b"validation-content-A")
        v1 = mm.compute_dataset_version(self.train_path, self.validation_path)
        v2 = mm.compute_dataset_version(self.train_path, self.validation_path)
        self.assertIsNotNone(v1)
        self.assertEqual(v1, v2)

    def test_different_content_produces_different_dataset_version(self):
        _write_bytes(self.train_path, b"train-content-A")
        _write_bytes(self.validation_path, b"validation-content-A")
        v1 = mm.compute_dataset_version(self.train_path, self.validation_path)

        _write_bytes(self.train_path, b"train-content-B-different")
        v2 = mm.compute_dataset_version(self.train_path, self.validation_path)
        self.assertNotEqual(v1, v2)

    def test_missing_file_returns_none(self):
        # validation_path was never written
        _write_bytes(self.train_path, b"train-content-A")
        self.assertIsNone(mm.compute_dataset_version(self.train_path, self.validation_path))


class TestBuildModelVersion(unittest.TestCase):

    def test_deterministic_given_same_inputs(self):
        v1 = mm.build_model_version("20260101T000000Z", "abc123", "deadbeef")
        v2 = mm.build_model_version("20260101T000000Z", "abc123", "deadbeef")
        self.assertEqual(v1, v2)

    def test_changes_when_dataset_version_changes(self):
        v1 = mm.build_model_version("20260101T000000Z", "abc123", "deadbeef")
        v2 = mm.build_model_version("20260101T000000Z", "xyz789", "deadbeef")
        self.assertNotEqual(v1, v2)

    def test_none_safe_for_missing_git_and_dataset(self):
        version = mm.build_model_version("20260101T000000Z", None, None)
        self.assertIn("nodataset", version)
        self.assertIn("nogit", version)


class TestGitCommitHash(unittest.TestCase):

    def test_returns_a_string_or_none_never_raises(self):
        result = mm.get_git_commit_hash()
        self.assertTrue(result is None or isinstance(result, str))


class TestBuildAndSaveMetadata(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.train_path = Path(self._tmpdir.name) / "train.parquet"
        self.validation_path = Path(self._tmpdir.name) / "validation.parquet"
        _write_bytes(self.train_path, b"fake-train-bytes")
        _write_bytes(self.validation_path, b"fake-validation-bytes")
        self.metadata_path = Path(self._tmpdir.name) / "model_metadata.json"

        registry = build_default_registry()
        self.feature_names = [f.name for f in registry.model_matrix_features()]

    def tearDown(self):
        self._tmpdir.cleanup()

    def _build(self):
        return mm.build_metadata(
            feature_names=self.feature_names,
            training_rows=100, validation_rows=25,
            query_groups_train=20, query_groups_validation=5,
            random_seed=42, best_iteration=17,
            training_parameters={"objective": "lambdarank"},
            training_metrics={"validation": {"ndcg@5": 0.81}},
            model_file_name="lightgbm_model.txt",
            train_path=self.train_path, validation_path=self.validation_path,
        )

    def test_build_metadata_includes_all_required_fields(self):
        metadata = self._build().to_dict()
        required = [
            "model_version", "training_timestamp", "git_commit_hash",
            "lightgbm_version", "feature_registry_version", "feature_count",
            "dataset_version", "training_rows", "validation_rows",
            "query_groups_train", "query_groups_validation", "random_seed",
            "best_iteration", "training_parameters", "training_metrics",
            "model_file_name",
        ]
        for field in required:
            self.assertIn(field, metadata, f"missing required metadata field: {field}")

    def test_build_metadata_uses_the_values_passed_in(self):
        metadata = self._build().to_dict()
        self.assertEqual(metadata["training_rows"], 100)
        self.assertEqual(metadata["validation_rows"], 25)
        self.assertEqual(metadata["query_groups_train"], 20)
        self.assertEqual(metadata["query_groups_validation"], 5)
        self.assertEqual(metadata["random_seed"], 42)
        self.assertEqual(metadata["best_iteration"], 17)
        self.assertEqual(metadata["feature_count"], len(self.feature_names))
        self.assertEqual(metadata["model_file_name"], "lightgbm_model.txt")

    def test_feature_registry_version_matches_registry(self):
        registry = build_default_registry()
        metadata = self._build().to_dict()
        self.assertEqual(metadata["feature_registry_version"], registry.to_dict()["version"])

    def test_dataset_version_matches_compute_dataset_version(self):
        metadata = self._build().to_dict()
        expected = mm.compute_dataset_version(self.train_path, self.validation_path)
        self.assertEqual(metadata["dataset_version"], expected)

    def test_save_and_load_round_trip(self):
        metadata = self._build()
        saved_path = mm.save_metadata(metadata, path=self.metadata_path)
        self.assertEqual(saved_path, self.metadata_path)
        self.assertTrue(self.metadata_path.exists())

        loaded = mm.load_metadata(self.metadata_path)
        self.assertEqual(loaded, metadata.to_dict())

    def test_saved_file_is_valid_json(self):
        metadata = self._build()
        mm.save_metadata(metadata, path=self.metadata_path)
        with open(self.metadata_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)   # raises if not valid JSON
        self.assertEqual(raw["model_version"], metadata.model_version)


class TestLoadMetadataMissingFallback(unittest.TestCase):

    def test_load_metadata_returns_none_for_nonexistent_path(self):
        result = mm.load_metadata(Path("/nonexistent/model_metadata.json"))
        self.assertIsNone(result)

    def test_load_metadata_returns_none_for_corrupt_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "corrupt.json"
            _write_bytes(path, b"{ this is not valid json ")
            result = mm.load_metadata(path)
            self.assertIsNone(result)

    def test_load_metadata_does_not_raise(self):
        # The whole point: a missing/broken metadata sidecar must never
        # surface as an exception -- callers (LightGBMRanker.load_model)
        # depend on this to keep serving the model without its provenance.
        try:
            mm.load_metadata(Path("/nonexistent/model_metadata.json"))
        except Exception as exc:   # noqa: BLE001
            self.fail(f"load_metadata raised {exc!r} instead of returning None")


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""
training/config.py

Central, explicit configuration for the LightGBM LambdaRank training data
pipeline. No hardcoded paths/constants inside the other training/ modules --
everything they need lives here, so the whole pipeline's behaviour is
visible and changeable from one place.

Grows incrementally alongside the rest of training/ -- only what the
components built so far (feature_registry.py, dataset_generator.py,
label_generator.py) need is defined here.
"""

from __future__ import annotations

from pathlib import Path

# Repo root (this file lives at <repo_root>/training/config.py).
REPO_ROOT = Path(__file__).resolve().parent.parent

# Where every training-pipeline artifact gets written. Kept out of the
# package itself (not committed) -- purely generated output.
ARTIFACTS_DIR = REPO_ROOT / "training" / "artifacts"

FEATURE_REGISTRY_PATH = ARTIFACTS_DIR / "feature_registry.json"
TRAIN_DATASET_PATH = ARTIFACTS_DIR / "train.parquet"
VALIDATION_DATASET_PATH = ARTIFACTS_DIR / "validation.parquet"
LABELED_TRAIN_DATASET_PATH = ARTIFACTS_DIR / "labeled_train.parquet"
LABELED_VALIDATION_DATASET_PATH = ARTIFACTS_DIR / "labeled_validation.parquet"
LABEL_STATISTICS_PATH = ARTIFACTS_DIR / "label_statistics.json"
DATASET_STATISTICS_PATH = ARTIFACTS_DIR / "dataset_statistics.json"

LIGHTGBM_MODEL_PATH = ARTIFACTS_DIR / "lightgbm_model.txt"
FEATURE_IMPORTANCE_CSV_PATH = ARTIFACTS_DIR / "feature_importance.csv"
FEATURE_IMPORTANCE_JSON_PATH = ARTIFACTS_DIR / "feature_importance.json"
TRAINING_METRICS_PATH = ARTIFACTS_DIR / "training_metrics.json"
TRAINING_CONFIG_PATH = ARTIFACTS_DIR / "training_config.json"
MODEL_METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

# Determinism: every random operation in this pipeline (train/validation
# splitting, any future sampling) must seed from this single constant, so
# two runs against the same source data produce byte-identical output.
RANDOM_SEED = 42

# Fraction of recommendation events (not rows -- see dataset_generator.py's
# split_train_validation, which splits by query_id) held out for validation.
DEFAULT_VALIDATION_FRACTION = 0.2

# How many days after a recommendation to look for a qualifying submission
# before treating it as ignored. See training/LABELING_STRATEGY.md for why
# 14 days was chosen as the default.
DEFAULT_OUTCOME_WINDOW_DAYS = 14.0

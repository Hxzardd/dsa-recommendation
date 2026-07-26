"""
training/model_metadata.py

Builds, saves, and loads the reproducibility metadata that accompanies
every trained LightGBM model (training/artifacts/model_metadata.json).

Written once by train_lightgbm.py right after a training run, using values
that run already computed (row counts, query groups, params, best
iteration/scores) -- nothing here recomputes or duplicates that work, it
only assembles and serializes it plus a few identity fields (git commit,
library version, dataset content hash) that belong at the metadata layer,
not inside the training script itself.

Read by pipeline/recommender/services/lightgbm_ranker.py alongside the
model file, so a served model's provenance -- what code trained it, on
what data, with what config -- is always inspectable, not just the raw
booster. Missing metadata (e.g. a model file copied without its sidecar
JSON) is a real, expected condition, not an error: load_metadata() returns
None rather than fabricating placeholder values, and callers decide how to
degrade.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import lightgbm

from training.config import (
    MODEL_METADATA_PATH,
    REPO_ROOT,
    TRAIN_DATASET_PATH,
    VALIDATION_DATASET_PATH,
)
from training.feature_registry import build_default_registry


@dataclass
class ModelMetadata:
    model_version: str
    training_timestamp: str
    git_commit_hash: Optional[str]
    lightgbm_version: str
    feature_registry_version: int
    feature_count: int
    dataset_version: Optional[str]
    training_rows: int
    validation_rows: int
    query_groups_train: int
    query_groups_validation: int
    random_seed: int
    best_iteration: int
    training_parameters: dict
    training_metrics: dict
    model_file_name: str

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Identity helpers -- each is best-effort and returns None (never a
# fabricated placeholder) when the real value genuinely isn't available.
# ---------------------------------------------------------------------------

def get_git_commit_hash() -> Optional[str]:
    """None (not "unknown"/"" or any other placeholder) when this isn't a
    git checkout, git isn't installed, or the call fails for any reason --
    e.g. a packaged/deployed copy of the repo with no .git directory."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _file_sha256(path: Path, chunk_size: int = 1 << 20) -> Optional[str]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_dataset_version(
    train_path: Path = TRAIN_DATASET_PATH,
    validation_path: Path = VALIDATION_DATASET_PATH,
) -> Optional[str]:
    """
    A content-derived dataset identity, not an arbitrary incrementing
    number disconnected from what data actually produced a model: sha256
    of train.parquet's bytes combined with validation.parquet's, truncated
    to 16 hex chars. Identical dataset bytes always produce the same
    dataset_version; any change to either file (a new simulator run,
    different user count/seed) changes it. Returns None if either file is
    missing, rather than hashing a partial/absent dataset.
    """
    train_hash = _file_sha256(train_path)
    validation_hash = _file_sha256(validation_path)
    if train_hash is None or validation_hash is None:
        return None
    return hashlib.sha256((train_hash + validation_hash).encode("utf-8")).hexdigest()[:16]


def _training_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_model_version(training_timestamp: str, dataset_version: Optional[str],
                         git_commit_hash: Optional[str]) -> str:
    """
    e.g. "lgbm-20260725T163000Z-a1b2c3d4e5f6g7h8-1a2b3c4d" -- sortable by
    time, and its dataset/commit suffixes tie a specific model file back
    to the exact data and code that produced it (both None-safe: falls
    back to an explicit "nodataset"/"nogit" token rather than silently
    dropping the segment, so the format is always parseable the same way).
    """
    dataset_part = dataset_version or "nodataset"
    commit_part = (git_commit_hash or "nogit")[:8]
    return f"lgbm-{training_timestamp}-{dataset_part}-{commit_part}"


# ---------------------------------------------------------------------------
# Build / save / load
# ---------------------------------------------------------------------------

def build_metadata(
    *,
    feature_names: list[str],
    training_rows: int,
    validation_rows: int,
    query_groups_train: int,
    query_groups_validation: int,
    random_seed: int,
    best_iteration: int,
    training_parameters: dict,
    training_metrics: dict,
    model_file_name: str,
    train_path: Path = TRAIN_DATASET_PATH,
    validation_path: Path = VALIDATION_DATASET_PATH,
) -> ModelMetadata:
    """Assembles metadata from values the training run already computed
    (passed in -- not recomputed here) plus identity fields this module
    owns (git commit, library version, feature registry version, dataset
    content hash, timestamp, model_version)."""
    registry = build_default_registry()
    dataset_version = compute_dataset_version(train_path, validation_path)
    git_commit_hash = get_git_commit_hash()
    training_timestamp = _training_timestamp()

    return ModelMetadata(
        model_version=build_model_version(training_timestamp, dataset_version, git_commit_hash),
        training_timestamp=training_timestamp,
        git_commit_hash=git_commit_hash,
        lightgbm_version=lightgbm.__version__,
        feature_registry_version=registry.to_dict()["version"],
        feature_count=len(feature_names),
        dataset_version=dataset_version,
        training_rows=training_rows,
        validation_rows=validation_rows,
        query_groups_train=query_groups_train,
        query_groups_validation=query_groups_validation,
        random_seed=random_seed,
        best_iteration=best_iteration,
        training_parameters=training_parameters,
        training_metrics=training_metrics,
        model_file_name=model_file_name,
    )


def save_metadata(metadata: ModelMetadata, path: Path = MODEL_METADATA_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metadata.to_dict(), fh, indent=2, default=str)
    return path


def load_metadata(path: Path = MODEL_METADATA_PATH) -> Optional[dict]:
    """Returns None (never a fabricated/placeholder dict) if the metadata
    file doesn't exist or fails to parse -- a model file can legitimately
    be present without its sidecar metadata (e.g. copied by hand), and
    callers must decide how to degrade rather than being handed fake
    values that look real."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None

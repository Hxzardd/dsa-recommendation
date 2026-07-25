"""
training/smoke_test_model.py

Lightweight developer sanity check for the baseline LightGBM model
(training/artifacts/lightgbm_model.txt) against the validation dataset
(training/artifacts/validation.parquet). NOT the project's evaluation
framework -- just enough to confirm the model loads, predicts, and ranks
sensibly before that framework lands. Only NDCG@5 / NDCG@10 (via sklearn,
not reimplemented), a few manual-inspection samples, and basic sanity
checks.

Feature list/order and categorical columns come from
training/feature_registry.py, and feature preparation / query-group
reconstruction reuse training/train_lightgbm.py's own helpers -- no
FeatureRegistry logic is duplicated here.

Run:
    python -m training.smoke_test_model
"""
from __future__ import annotations

import random

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from training.config import LIGHTGBM_MODEL_PATH, RANDOM_SEED, VALIDATION_DATASET_PATH
from training.feature_registry import build_default_registry
from training.train_lightgbm import _prepare_features, _query_groups

N_SAMPLE_QUERIES = 5
TOP_N_PROBLEMS = 5


def _group_slices(groups: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) row-index bounds for each contiguous query group."""
    bounds = []
    start = 0
    for size in groups:
        bounds.append((start, start + size))
        start += size
    return bounds


def main():
    registry = build_default_registry()
    feature_names = [f.name for f in registry.model_matrix_features()]
    categorical_names = [f.name for f in registry.categorical_features()]

    df = pd.read_parquet(VALIDATION_DATASET_PATH)

    # --------------------------------------------------------------- validation
    missing = set(feature_names) - set(df.columns)
    assert not missing, f"Missing features required by FeatureRegistry: {sorted(missing)}"

    booster = lgb.Booster(model_file=str(LIGHTGBM_MODEL_PATH))
    print("Model summary")
    print(f"  loaded successfully from {LIGHTGBM_MODEL_PATH}")

    assert list(booster.feature_name()) == feature_names, (
        "Model's stored feature order does not match FeatureRegistry.model_matrix_features() -- "
        "predictions would be silently wrong if features were fed in a different order."
    )
    print(f"  feature ordering matches FeatureRegistry ({len(feature_names)} features)")

    groups = _query_groups(df)
    assert groups.sum() == len(df), "Query group sizes do not cover every row"
    n_groups = len(groups)

    print("\nDataset summary")
    print(f"  validation rows: {len(df)}")
    print(f"  query groups: {n_groups}")
    print(f"  average group size: {groups.mean():.2f}")

    # --------------------------------------------------------------- scoring
    X, _ = _prepare_features(df, feature_names, categorical_names)
    preds = booster.predict(X)
    assert not np.isnan(preds).any(), "Model produced NaN predictions"
    assert len(preds) == len(df), "Prediction count does not match row count"

    labels = df["label"].to_numpy()
    slices = _group_slices(groups)
    assert len(slices) == n_groups, "Every query group must produce predictions"

    ndcg5_scores, ndcg10_scores = [], []
    for start, end in slices:
        y_true = labels[start:end].reshape(1, -1)
        y_score = preds[start:end].reshape(1, -1)
        ndcg5_scores.append(ndcg_score(y_true, y_score, k=5))
        ndcg10_scores.append(ndcg_score(y_true, y_score, k=10))

    avg_ndcg5 = float(np.mean(ndcg5_scores))
    avg_ndcg10 = float(np.mean(ndcg10_scores))

    print("\nRanking summary")
    print(f"  average NDCG@5:  {avg_ndcg5:.4f}")
    print(f"  average NDCG@10: {avg_ndcg10:.4f}")

    # --------------------------------------------------------------- manual inspection samples
    rng = random.Random(RANDOM_SEED)
    sample_indices = rng.sample(range(n_groups), min(N_SAMPLE_QUERIES, n_groups))

    print("\nSampled query groups (manual inspection only):")
    for idx in sample_indices:
        start, end = slices[idx]
        query_id = df["query_id"].iloc[start]
        true_labels = labels[start:end]
        scores = preds[start:end]
        problem_ids = df["candidate_id"].iloc[start:end].to_numpy()

        ranking = np.argsort(-scores)   # descending
        ranked_problems = problem_ids[ranking]
        ranked_scores = scores[ranking]
        ranked_labels = true_labels[ranking]

        print("=" * 50)
        print(f"Query ID: {query_id}")
        print(f"True Labels: {true_labels.tolist()}")
        print(f"Predicted Scores: {np.round(scores, 4).tolist()}")
        print(f"Predicted Ranking (by candidate index, best first): {ranking.tolist()}")
        print(f"Top {min(TOP_N_PROBLEMS, len(ranked_problems))} Problems: "
              f"{ranked_problems[:TOP_N_PROBLEMS].tolist()}")
        print(f"  (their labels: {ranked_labels[:TOP_N_PROBLEMS].tolist()}, "
              f"scores: {np.round(ranked_scores[:TOP_N_PROBLEMS], 4).tolist()})")
    print("=" * 50)


if __name__ == "__main__":
    main()

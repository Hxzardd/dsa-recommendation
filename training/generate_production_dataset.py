"""
training/generate_production_dataset.py

Generates the first production-scale synthetic bootstrap dataset by running
the approved, unmodified pipeline end to end:

    SyntheticUserGenerator -> DatasetGenerator -> LabelGenerator

This is a new orchestration entrypoint, not a change to any existing
training/ component -- feature_registry.py, feature_extractor.py,
dataset_generator.py, label_generator.py, and synthetic_user_generator.py
are all called exactly as implemented and already validated (see the
bootstrap validation report this follows from).

Same real-catalog-seeded fake Qdrant this repo's validation script and test
suite already use (tests/test_progression_changes_ranking.py::FakeQdrant
pattern), because live Qdrant is unreachable in this environment -- real
Postgres `problem` catalog (topic_tags/difficulty_score), real `topic.slug`
values, and the real offline concept-concept graph
(synthetic_user_generator.load_real_concept_graph(), PREREQ from
topic_prerequisite + COOCCURS from the real offline graph loader) are used
throughout. No feature vector or dataset row is fabricated; nothing bypasses
the recommender.

Run:
    python training/generate_production_dataset.py
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter

import pandas as pd

from database.postgres import db
from training.config import (
    ARTIFACTS_DIR,
    DEFAULT_VALIDATION_FRACTION,
    LABEL_STATISTICS_PATH,
    RANDOM_SEED,
    TRAIN_DATASET_PATH,
    VALIDATION_DATASET_PATH,
)
from training.dataset_generator import DatasetGenerator
from training.feature_registry import build_default_registry
from training.label_generator import LabelGenerator
from training.synthetic_user_generator import (
    SimulatorConfig,
    SyntheticUserGenerator,
    load_real_concept_graph,
)

logging.basicConfig(level=logging.WARNING)   # quiet the expected "Centroid fetch failed" noise
log = logging.getLogger(__name__)

DATASET_STATISTICS_PATH = ARTIFACTS_DIR / "dataset_statistics.json"

NUM_USERS = 3000   # production-scale bootstrap: large enough for meaningful LambdaRank
                    # query-group volume, small enough to generate locally in under a
                    # few minutes with no live Qdrant/GPU dependency.


# ---------------------------------------------------------------------------
# Real catalog + real offline concept graph (same sources the approved
# validation run used -- see the validation report for why these are real,
# not fabricated, and why a fake Qdrant wrapper is necessary here).
# ---------------------------------------------------------------------------

class _Pt:
    __slots__ = ("id", "payload", "score")

    def __init__(self, pid, tags, diff, accept_rate):
        self.id = pid
        self.payload = {
            "problem_id": pid,
            "topic_tags": tags,
            "difficulty_score": diff,
            "source_acceptance_rate": accept_rate,
        }
        self.score = accept_rate if accept_rate is not None else 0.5


class RealCatalogFakeQdrant:
    """Same scroll/query_points interface
    tests/test_progression_changes_ranking.py::FakeQdrant exercises, seeded
    with the real catalog instead of synthetic problems -- necessary because
    live Qdrant is unreachable in this environment (confirmed:
    db_env.qdrant_client().get_collections() -> ConnectError)."""

    def __init__(self, problems):
        self.problems = problems

    def scroll(self, collection_name, scroll_filter=None, limit=10,
               with_payload=True, with_vectors=False):
        want_tags = None
        want_ids = None
        diff_range = None
        if scroll_filter is not None:
            for cond in scroll_filter.must:
                key = getattr(cond, "key", None)
                match = getattr(cond, "match", None)
                if key == "topic_tags" and match is not None and getattr(match, "any", None):
                    want_tags = set(match.any)
                if key == "problem_id" and match is not None and getattr(match, "any", None):
                    want_ids = set(match.any)
                if key == "difficulty_score" and getattr(cond, "range", None) is not None:
                    diff_range = cond.range
        out = []
        for p in self.problems:
            if want_ids is not None and p.id not in want_ids:
                continue
            if want_tags is not None and not (set(p.payload["topic_tags"]) & want_tags):
                continue
            if diff_range is not None:
                d = p.payload["difficulty_score"]
                if diff_range.gte is not None and d < diff_range.gte:
                    continue
                if diff_range.lte is not None and d > diff_range.lte:
                    continue
            out.append(p)
            if len(out) >= limit:
                break
        return out, None

    def query_points(self, collection_name, query, limit=10,
                      with_payload=True, with_vectors=False):
        class R:
            points = sorted(self.problems, key=lambda p: p.score, reverse=True)[:limit]
        return R()


def _load_real_catalog_and_topics():
    conn = db.get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            select problem_id, topic_tags, difficulty_score, source_acceptance_rate
            from problem
            where is_active = true and topic_tags is not null and difficulty_score is not null
        """)
        catalog_rows = cur.fetchall()
        cur.execute("select slug from topic")
        topic_slugs = [r[0] for r in cur.fetchall()]
    finally:
        db.release_connection(conn)
    catalog = [_Pt(pid, tags, diff, acc) for pid, tags, diff, acc in catalog_rows]
    return catalog, topic_slugs


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def main():
    report: dict = {}
    t0 = time.time()

    catalog, topic_slugs = _load_real_catalog_and_topics()
    print(f"Real catalog: {len(catalog)} problems, {len(topic_slugs)} topic slugs")
    qdrant = RealCatalogFakeQdrant(catalog)

    cc_edges = load_real_concept_graph()
    n_cc = sum(len(v) for v in cc_edges.values())
    print(f"Real concept-concept graph: {n_cc} edges")

    config = SimulatorConfig(
        num_users=NUM_USERS,
        total_n=30,
        k=10,
        random_seed=RANDOM_SEED,   # same seed the rest of the training pipeline uses -- deterministic generation
    )
    generator = SyntheticUserGenerator(
        qdrant=qdrant, topic_slugs=topic_slugs, config=config, cc_edges=cc_edges,
    )
    result = generator.generate()
    n_events = len(result.events)
    n_users = len({e.user_id for e in result.events})
    print(f"Generated {n_events} recommendation events for {n_users} users "
          f"(requested {NUM_USERS})")

    # -----------------------------------------------------------------
    # DatasetGenerator -- unmodified, exactly as implemented
    # -----------------------------------------------------------------
    dataset_gen = DatasetGenerator()
    df = dataset_gen.generate_dataframe(result.events)
    n_rows_raw = len(df)
    print(f"Raw dataset rows (pre single-candidate-group drop): {n_rows_raw}")

    schema_problems = dataset_gen.validate_schema(df)
    if schema_problems:
        raise RuntimeError(
            "Dataset failed FeatureRegistry schema validation:\n" +
            "\n".join(f"  - {p}" for p in schema_problems)
        )

    # -----------------------------------------------------------------
    # Drop single-candidate query groups. A LambdaRank query group needs
    # >=2 candidates to form any ranking pair; a group of exactly 1 can
    # never contribute a pairwise gradient, so it is dead weight for
    # training, not a usable row. Root cause of these groups is real
    # catalog scarcity within narrow (topic x difficulty-band x
    # already-shown) combinations -- confirmed during the approved
    # simulator validation -- not something to fabricate additional
    # candidates to paper over. Dropped from the dataframe, not from the
    # recommender's own output: the RecommendationEvent itself (and
    # anything else derived from it) is untouched.
    # -----------------------------------------------------------------
    group_sizes = df.groupby("query_id").size()
    single_candidate_ids = group_sizes[group_sizes == 1].index
    n_dropped_groups = len(single_candidate_ids)
    n_dropped_rows = int(group_sizes[group_sizes == 1].sum())
    df_filtered = df[~df["query_id"].isin(single_candidate_ids)].reset_index(drop=True)
    print(f"Dropped {n_dropped_groups} single-candidate query groups ({n_dropped_rows} rows)")

    # -----------------------------------------------------------------
    # Split by query_id (DatasetGenerator.split_train_validation, unmodified)
    # -----------------------------------------------------------------
    train_df, validation_df = dataset_gen.split_train_validation(
        df_filtered, validation_fraction=DEFAULT_VALIDATION_FRACTION, seed=RANDOM_SEED,
    )
    print(f"Split: {train_df['query_id'].nunique()} train query groups, "
          f"{validation_df['query_id'].nunique()} validation query groups")

    # -----------------------------------------------------------------
    # LabelGenerator -- unmodified, exactly as implemented. label_dataframe
    # is called separately on each split so each row's label is looked up
    # against the SAME outcome_provider the simulator produced (keyed by
    # (user_id, problem_id), independent of split assignment).
    # -----------------------------------------------------------------
    label_gen = LabelGenerator(provider=result.outcome_provider)
    labeled_train = label_gen.label_dataframe(train_df)
    labeled_validation = label_gen.label_dataframe(validation_df)

    validation_problems = label_gen.validate(labeled_train) + label_gen.validate(labeled_validation)
    if validation_problems:
        raise RuntimeError(
            "Label validation failed -- refusing to export:\n" +
            "\n".join(f"  - {p}" for p in validation_problems)
        )

    # no duplicate rows, across the whole exported dataset
    combined = pd.concat([labeled_train, labeled_validation], ignore_index=True)
    n_dup_rows = int(combined.duplicated().sum())
    if n_dup_rows:
        raise RuntimeError(f"{n_dup_rows} fully-duplicate rows found -- refusing to export")

    train_stats = label_gen.compute_statistics(labeled_train)
    validation_stats = label_gen.compute_statistics(labeled_validation)

    # -----------------------------------------------------------------
    # Export -- train.parquet / validation.parquet / label_statistics.json,
    # using the exact path constants training/config.py already defines
    # for this purpose (TRAIN_DATASET_PATH etc. name them "train.parquet" /
    # "validation.parquet" already -- DatasetGenerator.export()/
    # LabelGenerator.export() would write the LABELED frames to
    # differently-named paths, so the final labeled frames are written
    # directly to the paths already reserved for the final dataset).
    # -----------------------------------------------------------------
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    labeled_train.to_parquet(TRAIN_DATASET_PATH, index=False)
    labeled_validation.to_parquet(VALIDATION_DATASET_PATH, index=False)

    label_statistics = {
        "strategy": label_gen.strategy.name,
        "window_days": label_gen.window_days,
        "train": train_stats,
        "validation": validation_stats,
    }
    with open(LABEL_STATISTICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(label_statistics, fh, indent=2, default=str)

    # -----------------------------------------------------------------
    # dataset_statistics.json -- everything else requested for validation
    # -----------------------------------------------------------------
    registry = build_default_registry()
    trainable = [f.name for f in registry.model_matrix_features() if f.name in combined.columns]
    feat_df = combined[trainable]

    constant_features = []
    missing_counts = {}
    for c in trainable:
        col = feat_df[c]
        n_missing = int(col.isna().sum())
        if n_missing:
            missing_counts[c] = n_missing
        try:
            nunique = col.nunique(dropna=True)
        except TypeError:
            nunique = col.astype(str).nunique()
        if nunique <= 1:
            constant_features.append(c)

    group_sizes_final = combined.groupby("query_id").size()
    candidates_per_query_desc = group_sizes_final.describe().to_dict()
    query_group_distribution = {str(k): int(v) for k, v in Counter(group_sizes_final).items()}

    train_path_size = TRAIN_DATASET_PATH.stat().st_size
    validation_path_size = VALIDATION_DATASET_PATH.stat().st_size

    dataset_statistics = {
        "generated_at_unix": time.time(),
        "random_seed": RANDOM_SEED,
        "num_users_requested": NUM_USERS,
        "num_users_generated": n_users,
        "recommendation_events": n_events,
        "raw_dataset_rows": n_rows_raw,
        "dropped_single_candidate_groups": n_dropped_groups,
        "dropped_single_candidate_rows": n_dropped_rows,
        "final_dataset_rows": len(combined),
        "train_rows": len(labeled_train),
        "validation_rows": len(labeled_validation),
        "train_query_groups": int(train_df["query_id"].nunique()),
        "validation_query_groups": int(validation_df["query_id"].nunique()),
        "validation_fraction_target": DEFAULT_VALIDATION_FRACTION,
        "candidates_per_query_desc": candidates_per_query_desc,
        "query_group_size_distribution": query_group_distribution,
        "constant_features": constant_features,
        "missing_value_counts": missing_counts,
        "schema_validated_against_feature_registry": True,
        "label_validation_problems": [],
        "duplicate_rows_found": n_dup_rows,
        "train_dataset_bytes": train_path_size,
        "validation_dataset_bytes": validation_path_size,
        "generation_seconds": round(time.time() - t0, 1),
    }
    with open(DATASET_STATISTICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(dataset_statistics, fh, indent=2, default=str)

    print(json.dumps(dataset_statistics, indent=2, default=str))
    print(f"\nWrote:\n  {TRAIN_DATASET_PATH}\n  {VALIDATION_DATASET_PATH}\n"
          f"  {LABEL_STATISTICS_PATH}\n  {DATASET_STATISTICS_PATH}")


if __name__ == "__main__":
    main()

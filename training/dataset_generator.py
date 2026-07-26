"""
training/dataset_generator.py

Turns a stream of RecommendationEvents into a LightGBM LambdaRank-ready
DataFrame. Every feature value comes from training/feature_extractor.py --
this module only orchestrates iteration, schema/registry validation,
deterministic ordering, query-aware train/validation splitting, and parquet
export. No recommendation logic or feature formula is duplicated here.

Label generation is explicitly out of scope for this component -- every
row carries a `label` column populated with NaN, a placeholder for
training/label_generator.py (a later component) to fill in.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from pipeline.recommender.models.user_graph import UserGraph
from pipeline.recommender.services.adaptive_difficulty import DifficultyPlan
from pipeline.recommender.services.candidate_filtering import CandidateFilteringLayer
from pipeline.recommender.services.heuristic_ranker import HeuristicRanker

from training.config import (
    DEFAULT_VALIDATION_FRACTION,
    RANDOM_SEED,
    TRAIN_DATASET_PATH,
    VALIDATION_DATASET_PATH,
)
from training.feature_extractor import extract_features
from training.feature_registry import FeatureRegistry, build_default_registry


@dataclass(frozen=True)
class RecommendationEvent:
    """
    One recommendation event: the exact final slate of candidates shown to
    a user at a point in time, plus the graph/plan state that produced it.

    `candidates` MUST be the FINAL, post-ranking, post-diversity-mixed slate
    -- i.e. what DiversityMixer.mix() returned, the same set
    save_recommendation_log() would persist -- not the raw pre-filter
    candidate pool. This is deliberate, not an oversight: only candidates
    that were actually shown can ever receive a label later via
    recommendation_log.was_attempted, so a training row must correspond to
    something that could exist as a recommendation_log row. Including
    filtered-out or diversity-mixer-dropped candidates would produce rows
    that can never be labelled.
    """
    query_id: str
    user_id: str
    recommended_at: float
    graph: UserGraph
    plan: DifficultyPlan
    candidates: list          # list[MergedCandidate], in shown order
    catalog_metadata_by_problem_id: Optional[dict] = None   # problem_id -> offline catalog row


@dataclass
class DatasetExportResult:
    dataframe: pd.DataFrame
    train_path: Path
    validation_path: Path
    train_query_count: int
    validation_query_count: int


class DatasetGenerator:
    """
    Usage:
        gen = DatasetGenerator()
        df = gen.generate_dataframe(events)
        result = gen.export(df)
    """

    def __init__(self, registry: Optional[FeatureRegistry] = None,
                 ranker: Optional[HeuristicRanker] = None):
        self.registry = registry or build_default_registry()
        self.ranker = ranker or HeuristicRanker()

    # ------------------------------------------------------------- rows

    def generate_rows(self, event: RecommendationEvent) -> list[dict]:
        """One dict per candidate in this event, in the event's given
        (shown) order. A single CandidateFilteringLayer is reused across
        every candidate in the event (it's stateless per-call; the only
        caching that matters -- MergedCandidate._topic_stats_cache -- lives
        on each candidate object regardless of which layer instance touches
        it, per feature_extractor.py/candidate_filtering.py's own design)."""
        layer = CandidateFilteringLayer(event.graph)
        rows = []
        for candidate in event.candidates:
            catalog_row = None
            if event.catalog_metadata_by_problem_id is not None:
                catalog_row = event.catalog_metadata_by_problem_id.get(candidate.problem_id)

            extracted = extract_features(
                event.graph, event.plan, candidate,
                ranker=self.ranker,
                filtering_layer=layer,
                catalog_metadata=catalog_row,
                query_id=event.query_id,
                recommended_at=event.recommended_at,
            )
            row = extracted.to_flat_dict()
            row["label"] = float("nan")   # placeholder -- see label_generator.py (future component)
            rows.append(row)
        return rows

    # ------------------------------------------------------------- dataframe

    def generate_dataframe(self, events: Iterable[RecommendationEvent]) -> pd.DataFrame:
        """
        Deterministic regardless of the iteration order events are supplied
        in: events are sorted by (user_id, recommended_at, query_id) before
        flattening, and the resulting columns are always ordered to match
        the feature registry exactly.
        """
        events_sorted = sorted(
            events, key=lambda e: (e.user_id, e.recommended_at, e.query_id)
        )

        columns = self.registry.names()   # includes "label"; deterministic order

        all_rows: list[dict] = []
        for event in events_sorted:
            all_rows.extend(self.generate_rows(event))

        if not all_rows:
            return pd.DataFrame(columns=columns)

        df = pd.DataFrame(all_rows)
        df = df[columns]
        # Force a real numeric NaN dtype for the label placeholder rather
        # than leaving it to pandas' object-column inference -- keeps the
        # parquet schema clean and matches the registry's declared
        # DataType.NUMERICAL for "label".
        df["label"] = df["label"].astype("float64")
        return df

    # ------------------------------------------------------------- validation

    def validate_schema(self, df: pd.DataFrame) -> list[str]:
        """
        Export-time schema check against the registry: every declared
        column must be present, and no undeclared column may exist.
        Statistical/quality validation (ranges, duplicates, correlations)
        is dataset_validator.py's job -- a separate, later component --
        not this one's.
        """
        problems: list[str] = []

        registry_names = set(self.registry.names())
        df_names = set(df.columns)

        missing = registry_names - df_names
        if missing:
            problems.append(
                f"columns declared in the registry but missing from the dataframe: {sorted(missing)}"
            )

        extra = df_names - registry_names
        if extra:
            problems.append(
                f"dataframe columns not declared in the registry: {sorted(extra)}"
            )

        problems.extend(self.registry.validate())
        return problems

    # ------------------------------------------------------------- splitting

    def split_train_validation(
        self, df: pd.DataFrame,
        validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
        seed: int = RANDOM_SEED,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits by query_id, never by row -- splitting a query group across
        train/validation would corrupt LambdaRank's group structure (a
        partial group is not a valid ranking query) and leak query-level
        signal between the two sets. Deterministic: query_ids are sorted
        lexicographically (independent of row/event order) before a
        seeded shuffle, so the same dataframe always splits the same way.

        Rows with a missing query_id (should not occur for
        properly-constructed RecommendationEvents, since query_id is a
        required field) are excluded from both splits -- there is no valid
        group to assign them to.
        """
        query_ids = sorted(df["query_id"].dropna().unique().tolist())

        rng = random.Random(seed)
        rng.shuffle(query_ids)

        n_validation = int(round(len(query_ids) * validation_fraction))
        validation_ids = set(query_ids[:n_validation])
        train_ids = set(query_ids[n_validation:])

        train_df = df[df["query_id"].isin(train_ids)].reset_index(drop=True)
        validation_df = df[df["query_id"].isin(validation_ids)].reset_index(drop=True)
        return train_df, validation_df

    # ------------------------------------------------------------- export

    def export(
        self, df: pd.DataFrame,
        validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
        seed: int = RANDOM_SEED,
        train_path: Optional[Path] = None,
        validation_path: Optional[Path] = None,
    ) -> DatasetExportResult:
        """Validates the dataframe against the registry, splits by query_id,
        and writes train.parquet / validation.parquet. Raises ValueError
        (without writing anything) if schema validation fails."""
        problems = self.validate_schema(df)
        if problems:
            raise ValueError(
                "Dataset failed schema validation against the feature registry:\n"
                + "\n".join(f"  - {p}" for p in problems)
            )

        train_df, validation_df = self.split_train_validation(df, validation_fraction, seed)

        train_path = train_path or TRAIN_DATASET_PATH
        validation_path = validation_path or VALIDATION_DATASET_PATH
        train_path.parent.mkdir(parents=True, exist_ok=True)
        validation_path.parent.mkdir(parents=True, exist_ok=True)

        train_df.to_parquet(train_path, index=False)
        validation_df.to_parquet(validation_path, index=False)

        return DatasetExportResult(
            dataframe=df,
            train_path=train_path,
            validation_path=validation_path,
            train_query_count=train_df["query_id"].nunique(),
            validation_query_count=validation_df["query_id"].nunique(),
        )

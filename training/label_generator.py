"""
training/label_generator.py

Assigns a LambdaRank relevance label to every (query_id, candidate_id) row
produced by training/dataset_generator.py, using only data that already
exists: recommendation_log (was_attempted) and the submission table
(verdict, submitted_at) -- both already read elsewhere in this codebase
(user_graph_service.py::_load_submissions,
database.postgres.db::mark_recommendation_attempted).

See training/LABELING_STRATEGY.md for the full strategy comparison and
rationale. Summary: the default strategy (TernaryEngagementLabelStrategy)
grades each recommendation 0 (never attempted) / 1 (attempted, not solved
within the window) / 2 (solved within the window) -- deliberately NOT
graded further by solve speed or attempt count, because this recommender's
own ranking philosophy already treats "solved instantly" as no more
relevant than a productive-struggle solve (HeuristicRanker's proximity term
peaks at predicted_success=0.68, not 1.0); rewarding raw speed in the label
would train a future ranker away from the ZPD-appropriate recommendations
the heuristic already targets.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import pandas as pd

from training.config import (
    DEFAULT_OUTCOME_WINDOW_DAYS,
    LABEL_STATISTICS_PATH,
    LABELED_TRAIN_DATASET_PATH,
    LABELED_VALIDATION_DATASET_PATH,
)

SECONDS_PER_DAY = 86400.0


# ============================================================================
# Outcome: the raw, available signal for one (user, problem, recommended_at).
# ============================================================================

@dataclass(frozen=True)
class RecommendationOutcome:
    """Derived purely from recommendation_log + submission -- no invented data."""
    was_attempted: bool
    attempt_count: int
    first_attempt_at: Optional[float]
    solved: bool
    solved_at: Optional[float]


NEVER_ATTEMPTED = RecommendationOutcome(
    was_attempted=False, attempt_count=0, first_attempt_at=None,
    solved=False, solved_at=None,
)


class RecommendationOutcomeProvider(Protocol):
    """Abstraction over "what happened after this recommendation" so
    LabelGenerator never needs a live database to be tested."""

    def get_outcome(
        self, user_id: str, problem_id: str, recommended_at: float, window_days: float,
    ) -> RecommendationOutcome: ...


class InMemoryOutcomeProvider:
    """Test/offline provider -- outcomes supplied directly, keyed by
    (user_id, problem_id). Any key not present is treated as never
    attempted (the honest default -- absence of data here means no
    qualifying submission was supplied, not that one doesn't exist)."""

    def __init__(self, outcomes: dict[tuple[str, str], RecommendationOutcome]):
        self._outcomes = outcomes

    def get_outcome(self, user_id, problem_id, recommended_at, window_days) -> RecommendationOutcome:
        return self._outcomes.get((user_id, problem_id), NEVER_ATTEMPTED)


class PostgresOutcomeProvider:
    """
    Production provider. Queries the `submission` table -- the same table
    and columns user_graph_service.py::_load_submissions already reads --
    for submissions on (user_id, problem_id) between recommended_at and
    recommended_at + window_days. Reuses
    database.postgres.db.get_connection/release_connection, the same
    connection-pool utilities every other query in this repo uses, rather
    than opening a new connection path.
    """

    def __init__(self):
        from database.postgres.db import get_connection, release_connection
        self._get_connection = get_connection
        self._release_connection = release_connection

    def get_outcome(self, user_id, problem_id, recommended_at, window_days) -> RecommendationOutcome:
        window_end = recommended_at + window_days * SECONDS_PER_DAY
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT verdict, submitted_at
                    FROM   submission
                    WHERE  user_id = %s AND problem_id = %s
                       AND status = 'COMPLETED'
                       AND submitted_at >= to_timestamp(%s)
                       AND submitted_at <= to_timestamp(%s)
                    ORDER  BY submitted_at ASC
                    """,
                    (user_id, problem_id, recommended_at, window_end),
                )
                rows = cur.fetchall()
        finally:
            self._release_connection(conn)

        if not rows:
            return NEVER_ATTEMPTED

        def _ts(value) -> float:
            return value.timestamp() if hasattr(value, "timestamp") else float(value)

        first_attempt_at = _ts(rows[0][1])
        solved_row = next((r for r in rows if r[0] == "Accepted"), None)

        return RecommendationOutcome(
            was_attempted=True,
            attempt_count=len(rows),
            first_attempt_at=first_attempt_at,
            solved=solved_row is not None,
            solved_at=_ts(solved_row[1]) if solved_row is not None else None,
        )


# ============================================================================
# Label strategies -- pluggable, swappable without touching LabelGenerator.
# ============================================================================

class LabelStrategy(ABC):
    name: str = "base"
    max_label: int = 0

    @abstractmethod
    def compute_label(self, outcome: RecommendationOutcome) -> float: ...


class BinarySolvedLabelStrategy(LabelStrategy):
    """label = 1 if solved within the window, else 0.

    Considered and NOT the default: collapses "attempted but failed" and
    "completely ignored" into the same label, discarding a real,
    already-available distinction (was_attempted) that graded relevance
    could use. Kept for experimentation/comparison -- see
    LABELING_STRATEGY.md."""
    name = "binary_solved"
    max_label = 1

    def compute_label(self, outcome: RecommendationOutcome) -> float:
        return 1.0 if outcome.solved else 0.0


class TernaryEngagementLabelStrategy(LabelStrategy):
    """
    DEFAULT. 0 = never attempted (ignored), 1 = attempted but not solved
    within the window, 2 = solved within the window.

    Deliberately does not grade further by solve speed or attempt count --
    see this module's docstring for why: this recommender's own ranking
    philosophy already treats "solved instantly" as no more relevant than
    a productive-struggle solve, so rewarding raw speed would bake an
    unsupported "ideal attempt curve" assumption into the label and train
    a future model away from the ZPD-appropriate recommendations the
    current heuristic already targets.
    """
    name = "ternary_engagement"
    max_label = 2

    def compute_label(self, outcome: RecommendationOutcome) -> float:
        if outcome.solved:
            return 2.0
        if outcome.was_attempted:
            return 1.0
        return 0.0


STRATEGIES: dict[str, type[LabelStrategy]] = {
    "binary_solved": BinarySolvedLabelStrategy,
    "ternary_engagement": TernaryEngagementLabelStrategy,
}


def get_strategy(name: str) -> LabelStrategy:
    if name not in STRATEGIES:
        raise ValueError(f"Unknown label strategy {name!r}; available: {sorted(STRATEGIES)}")
    return STRATEGIES[name]()


# ============================================================================
# LabelGenerator
# ============================================================================

@dataclass
class LabelGenerationResult:
    labeled_train: pd.DataFrame
    labeled_validation: pd.DataFrame
    train_path: Path
    validation_path: Path
    statistics_path: Path
    statistics: dict


class LabelGenerator:
    """
    Usage:
        gen = LabelGenerator(provider=PostgresOutcomeProvider())
        result = gen.export(train_df, validation_df)
    """

    REQUIRED_COLUMNS = ("query_id", "candidate_id", "user_id", "recommended_at")

    def __init__(
        self,
        provider: RecommendationOutcomeProvider,
        strategy: Optional[LabelStrategy] = None,
        window_days: float = DEFAULT_OUTCOME_WINDOW_DAYS,
    ):
        self.provider = provider
        self.strategy = strategy or TernaryEngagementLabelStrategy()
        self.window_days = window_days

    # ------------------------------------------------------------- labeling

    def label_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a COPY of df (never mutates the input) with `label`
        overwritten from the placeholder NaN dataset_generator.py wrote,
        plus diagnostic `_outcome_*` columns (solved/attempted/attempt
        count/first-attempt timestamp) so validation and statistics don't
        need to re-query the provider.
        """
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"dataframe missing required columns: {sorted(missing)}")

        labeled = df.copy()
        labels: list[float] = []
        outcome_solved: list[bool] = []
        outcome_attempted: list[bool] = []
        outcome_attempt_count: list[int] = []
        outcome_first_attempt_at: list[Optional[float]] = []

        for _, row in labeled.iterrows():
            recommended_at = row["recommended_at"]
            user_id = row["user_id"]
            candidate_id = row["candidate_id"]

            if pd.isna(recommended_at) or pd.isna(user_id) or pd.isna(candidate_id):
                labels.append(float("nan"))
                outcome_solved.append(False)
                outcome_attempted.append(False)
                outcome_attempt_count.append(0)
                outcome_first_attempt_at.append(None)
                continue

            outcome = self.provider.get_outcome(
                user_id, candidate_id, float(recommended_at), self.window_days,
            )
            labels.append(self.strategy.compute_label(outcome))
            outcome_solved.append(outcome.solved)
            outcome_attempted.append(outcome.was_attempted)
            outcome_attempt_count.append(outcome.attempt_count)
            outcome_first_attempt_at.append(outcome.first_attempt_at)

        labeled["label"] = pd.Series(labels, index=labeled.index, dtype="float64")
        labeled["_outcome_solved"] = outcome_solved
        labeled["_outcome_attempted"] = outcome_attempted
        labeled["_outcome_attempt_count"] = outcome_attempt_count
        labeled["_outcome_first_attempt_at"] = outcome_first_attempt_at
        return labeled

    # ------------------------------------------------------------- validation

    def validate(self, labeled_df: pd.DataFrame) -> list[str]:
        """
        Detects: duplicate (query_id, candidate_id) rows, missing
        recommendation-history identifiers, invalid timestamps (including
        a submission timestamped before its own recommendation --
        impossible attribution), inconsistent recommendation events (a
        query_id spanning more than one user_id/recommended_at), and
        out-of-range ("impossible") label values for the active strategy.
        """
        problems: list[str] = []

        dup_mask = labeled_df.duplicated(subset=["query_id", "candidate_id"], keep=False)
        if dup_mask.any():
            problems.append(f"{int(dup_mask.sum())} duplicate (query_id, candidate_id) rows found")

        for col in self.REQUIRED_COLUMNS:
            n_missing = int(labeled_df[col].isna().sum())
            if n_missing:
                problems.append(f"{n_missing} rows missing {col!r} (missing recommendation history)")

        valid_ts = labeled_df["recommended_at"].dropna()
        n_bad_ts = int((valid_ts <= 0).sum())
        if n_bad_ts:
            problems.append(f"{n_bad_ts} rows have a non-positive recommended_at timestamp")

        if "_outcome_first_attempt_at" in labeled_df.columns:
            with_attempt = labeled_df.dropna(subset=["_outcome_first_attempt_at", "recommended_at"])
            impossible = with_attempt[
                with_attempt["_outcome_first_attempt_at"] < with_attempt["recommended_at"]
            ]
            if len(impossible):
                problems.append(
                    f"{len(impossible)} rows have a submission timestamped before "
                    "recommended_at (impossible attribution)"
                )

        grouped = labeled_df.dropna(subset=["query_id"]).groupby("query_id")[["user_id", "recommended_at"]].nunique()
        inconsistent = grouped[(grouped["user_id"] > 1) | (grouped["recommended_at"] > 1)]
        if len(inconsistent):
            problems.append(
                f"{len(inconsistent)} query_id group(s) have inconsistent "
                "user_id/recommended_at across their own rows"
            )

        labels = labeled_df["label"].dropna()
        out_of_range = labels[(labels < 0) | (labels > self.strategy.max_label)]
        if len(out_of_range):
            problems.append(
                f"{len(out_of_range)} rows have a label outside the valid range "
                f"[0, {self.strategy.max_label}] for strategy {self.strategy.name!r}"
            )

        return problems

    # ------------------------------------------------------------- statistics

    def compute_statistics(self, labeled_df: pd.DataFrame) -> dict:
        total = len(labeled_df)
        if total == 0:
            return {
                "strategy": self.strategy.name, "window_days": self.window_days,
                "total_rows": 0, "percent_solved": 0.0,
                "percent_attempted_not_solved": 0.0, "percent_ignored": 0.0,
                "unlabeled_rows": 0, "average_relevance": 0.0,
                "label_distribution": {}, "label_imbalance_ratio": None,
            }

        solved = int(labeled_df["_outcome_solved"].sum())
        attempted = int(labeled_df["_outcome_attempted"].sum())
        ignored = total - attempted
        unlabeled = int(labeled_df["label"].isna().sum())

        label_counts = labeled_df["label"].dropna().value_counts().sort_index()
        label_distribution = {str(k): int(v) for k, v in label_counts.items()}
        avg_relevance = float(labeled_df["label"].mean(skipna=True))

        imbalance_ratio = None
        if len(label_counts) >= 2:
            imbalance_ratio = round(float(label_counts.max()) / float(label_counts.min()), 4)

        return {
            "strategy": self.strategy.name,
            "window_days": self.window_days,
            "total_rows": total,
            "percent_solved": round(100.0 * solved / total, 2),
            "percent_attempted_not_solved": round(100.0 * (attempted - solved) / total, 2),
            "percent_ignored": round(100.0 * ignored / total, 2),
            "unlabeled_rows": unlabeled,
            "average_relevance": round(avg_relevance, 4),
            "label_distribution": label_distribution,
            "label_imbalance_ratio": imbalance_ratio,
        }

    # ------------------------------------------------------------- export

    def export(
        self,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        train_path: Optional[Path] = None,
        validation_path: Optional[Path] = None,
        statistics_path: Optional[Path] = None,
    ) -> LabelGenerationResult:
        labeled_train = self.label_dataframe(train_df)
        labeled_validation = self.label_dataframe(validation_df)

        problems = self.validate(labeled_train) + self.validate(labeled_validation)
        if problems:
            raise ValueError(
                "Label validation failed:\n" + "\n".join(f"  - {p}" for p in problems)
            )

        train_path = train_path or LABELED_TRAIN_DATASET_PATH
        validation_path = validation_path or LABELED_VALIDATION_DATASET_PATH
        statistics_path = statistics_path or LABEL_STATISTICS_PATH
        for p in (train_path, validation_path, statistics_path):
            p.parent.mkdir(parents=True, exist_ok=True)

        labeled_train.to_parquet(train_path, index=False)
        labeled_validation.to_parquet(validation_path, index=False)

        statistics = {
            "strategy": self.strategy.name,
            "window_days": self.window_days,
            "train": self.compute_statistics(labeled_train),
            "validation": self.compute_statistics(labeled_validation),
        }
        with open(statistics_path, "w", encoding="utf-8") as fh:
            json.dump(statistics, fh, indent=2)

        return LabelGenerationResult(
            labeled_train=labeled_train,
            labeled_validation=labeled_validation,
            train_path=train_path,
            validation_path=validation_path,
            statistics_path=statistics_path,
            statistics=statistics,
        )

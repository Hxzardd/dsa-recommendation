"""
training/feature_registry.py

The single source of truth for every feature the LightGBM LambdaRank
training dataset is allowed to produce. feature_extractor.py reads this
registry to know WHAT to compute and HOW to compute it (source_file /
source_function point at the exact existing recommender code being reused);
dataset_validator.py reads it to check the generated dataset actually
matches what was declared (right dtype, right range, right leakage
classification).

Design principle: adding a feature to the dataset means adding it HERE
first, with a justification. feature_extractor.py must not compute anything
that isn't registered -- this keeps "what features exist" auditable in one
place instead of scattered across extraction code.

Every feature below was evaluated against the quality checklist (exists at
recommendation time / no future info / not already computed elsewhere in a
way that's cheaper to reuse / generalizes / stable / interpretable / useful
for LambdaRank / SHAP-friendly) before being included. Features considered
and REJECTED are documented in REJECTED_FEATURES below with the reason --
that record is as important as the accepted list, so a future contributor
doesn't re-propose something already evaluated and declined.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from training.config import FEATURE_REGISTRY_PATH


class FeatureGroup(str, Enum):
    IDENTIFIER = "identifier"   # query_id, candidate_id, user_id, label -- never in the model matrix
    USER = "user"
    PROBLEM = "problem"
    PAIR = "pair"
    POOL = "pool"
    GRAPH = "graph"
    DIAGNOSTIC = "diagnostic"   # e.g. the current heuristic's own rank_score -- kept for comparison, not training


class DataType(str, Enum):
    NUMERICAL = "numerical"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    IDENTIFIER = "identifier"


class LeakageRisk(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ComputationalCost(str, Enum):
    FREE = "free"       # already computed by the live pipeline for every recommendation; zero extra cost
    LOW = "low"         # cheap aggregation over data the pipeline already loaded
    MEDIUM = "medium"   # requires one extra query/join beyond what the live pipeline does today
    HIGH = "high"       # expensive (e.g. rebuilding an embedding); avoid unless justified


@dataclass(frozen=True)
class FeatureSpec:
    """One row of the feature registry. See module docstring for intent."""

    name: str
    group: FeatureGroup
    dtype: DataType

    # Exactly where this value comes from in the EXISTING recommender code.
    # "new:<file>" for genuinely engineered features with no prior home.
    source_file: str
    source_function: str

    reused: bool   # True: read from an existing computation. False: newly engineered here.

    reason_for_inclusion: str
    engineering_justification: Optional[str] = None   # required when reused=False

    expected_range: Optional[str] = None
    missing_strategy: str = "native_nan"   # LightGBM's native missing-value handling, unless stated otherwise
    inference_available: bool = True       # computable at serving time, with data that exists today
    computational_cost: ComputationalCost = ComputationalCost.FREE
    leakage_risk: LeakageRisk = LeakageRisk.NONE
    include_in_model_matrix: bool = True   # False for identifiers and diagnostic-only columns

    def __post_init__(self) -> None:
        if not self.reused and not self.engineering_justification:
            raise ValueError(
                f"Feature {self.name!r} is newly engineered (reused=False) "
                "but has no engineering_justification -- every non-reused "
                "feature must justify why it was added instead of just "
                "reusing existing recommender output."
            )
        if self.leakage_risk in (LeakageRisk.HIGH, LeakageRisk.MEDIUM) and self.include_in_model_matrix:
            raise ValueError(
                f"Feature {self.name!r} has {self.leakage_risk.value} leakage "
                "risk but is marked include_in_model_matrix=True -- either "
                "resolve the leakage risk or exclude it from training."
            )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "group": self.group.value,
            "dtype": self.dtype.value,
            "source_file": self.source_file,
            "source_function": self.source_function,
            "reused": self.reused,
            "reason_for_inclusion": self.reason_for_inclusion,
            "engineering_justification": self.engineering_justification,
            "expected_range": self.expected_range,
            "missing_strategy": self.missing_strategy,
            "inference_available": self.inference_available,
            "computational_cost": self.computational_cost.value,
            "leakage_risk": self.leakage_risk.value,
            "include_in_model_matrix": self.include_in_model_matrix,
        }


class FeatureRegistry:
    """In-memory registry of every declared feature. Construct via
    build_default_registry() below rather than registering features ad hoc,
    so the canonical set stays in one reviewable place."""

    def __init__(self) -> None:
        self._features: dict[str, FeatureSpec] = {}

    def register(self, spec: FeatureSpec) -> None:
        if spec.name in self._features:
            raise ValueError(f"Duplicate feature registration: {spec.name!r}")
        self._features[spec.name] = spec

    def get(self, name: str) -> FeatureSpec:
        return self._features[name]

    def __contains__(self, name: str) -> bool:
        return name in self._features

    def __len__(self) -> int:
        return len(self._features)

    def all(self) -> list[FeatureSpec]:
        return sorted(self._features.values(), key=lambda f: f.name)

    def by_group(self, group: FeatureGroup) -> list[FeatureSpec]:
        return [f for f in self.all() if f.group == group]

    def model_matrix_features(self) -> list[FeatureSpec]:
        """Only the columns a LightGBM model should ever train on --
        identifiers and diagnostic-only columns are excluded."""
        return [f for f in self.all() if f.include_in_model_matrix]

    def categorical_features(self) -> list[FeatureSpec]:
        return [f for f in self.model_matrix_features() if f.dtype == DataType.CATEGORICAL]

    def names(self) -> list[str]:
        return [f.name for f in self.all()]

    def validate(self) -> list[str]:
        """Returns a list of problems found across the whole registry
        (empty list == valid). Duplicate names can't actually occur (register()
        already raises), but every other invariant is checked here so
        dataset_validator.py can assert the registry itself is sound before
        trusting it to validate a generated dataset."""
        problems: list[str] = []
        for f in self._features.values():
            if not f.reused and not f.engineering_justification:
                problems.append(f"{f.name}: engineered feature missing justification")
            if f.leakage_risk in (LeakageRisk.HIGH, LeakageRisk.MEDIUM) and f.include_in_model_matrix:
                problems.append(f"{f.name}: {f.leakage_risk.value} leakage risk but included in model matrix")
            if f.dtype == DataType.IDENTIFIER and f.include_in_model_matrix:
                problems.append(f"{f.name}: identifier column marked include_in_model_matrix=True")
        return problems

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "feature_count": len(self._features),
            "model_matrix_feature_count": len(self.model_matrix_features()),
            "features": [f.to_dict() for f in self.all()],
            "rejected_features": [r.to_dict() for r in REJECTED_FEATURES],
        }

    def export(self, path: Optional[Path] = None) -> Path:
        path = path or FEATURE_REGISTRY_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path


@dataclass(frozen=True)
class RejectedFeature:
    """A feature that was considered and explicitly declined. Kept as a
    permanent record so it isn't re-proposed without re-litigating the
    reason it was dropped."""
    name: str
    reason: str

    def to_dict(self) -> dict:
        return {"name": self.name, "reason": self.reason}


REJECTED_FEATURES: list[RejectedFeature] = [
    RejectedFeature(
        name="raw_user_state_vector_1920d",
        reason=(
            "UserStateVector.vector (models/user_state.py) is a dense 1920-d "
            "embedding. Exploding it into 1920 individual LightGBM columns "
            "fails multiple checklist items at once: not interpretable, not "
            "SHAP-friendly (1920 SHAP values per row is not explainable in "
            "any practical sense), and GBDTs generalize poorly over raw "
            "dense embedding dimensions compared to the scalar aggregates "
            "already available (mastered/weak/urgent concept counts, "
            "avg_mastery). The vector's actual predictive content for a "
            "USER-PROBLEM pair is better captured as a single scalar pair "
            "feature -- see user_problem_embedding_similarity below -- "
            "rather than 1920 raw dimensions."
        ),
    ),
    RejectedFeature(
        name="review_load",
        reason=(
            "Proposed as overdue_review_count + urgent_topic_count. Both "
            "addends are already separate registered features "
            "(overdue_review_count, urgent_topic_count); a GBDT can already "
            "learn any additive interaction between two existing columns in "
            "a single split combination. An explicit sum column adds no "
            "information the model can't already derive and only inflates "
            "feature count without improving ranking quality."
        ),
    ),
    RejectedFeature(
        name="topic_entropy",
        reason=(
            "Shannon entropy is only well-defined over a normalised "
            "probability distribution, not raw [0,1] mastery scores -- "
            "computing it without first normalising the scores to sum to 1 "
            "changes the mastery scores' meaning from 'how well is this "
            "topic known' to 'what fraction of this user's total mastery "
            "sits in this topic', a different, less interpretable quantity. "
            "mastery_variance already captures distributional spread in an "
            "interpretable, standard way; entropy would be redundant with it "
            "for the marginal benefit of a harder-to-explain feature."
        ),
    ),
    RejectedFeature(
        name="rank_score_as_training_feature",
        reason=(
            "The current HeuristicRanker's own final rank_score IS included "
            "in the dataset, but as a DIAGNOSTIC column only "
            "(include_in_model_matrix=False; see FeatureGroup.DIAGNOSTIC), "
            "not a training feature. Training a new ranker to reproduce the "
            "score of the ranker it's meant to improve upon is target "
            "leakage from the current production heuristic -- it teaches "
            "the model to imitate today's formula (including today's known "
            "weight-sum/redundancy characteristics) rather than learn from "
            "actual outcomes. Kept only so the new model's ranking can later "
            "be compared against the heuristic's on the same rows."
        ),
    ),
    RejectedFeature(
        name="time_of_day_cyclical_features",
        reason=(
            "No existing evidence in this recommender that time-of-day/day-"
            "of-week affects recommendation quality; adding cyclical sin/cos "
            "time features on recommended_at speculatively risks the model "
            "fitting noise rather than signal. Not included until a "
            "hypothesis for why it would matter is established."
        ),
    ),
]


def build_default_registry() -> FeatureRegistry:
    """The canonical registry for this recommendation engine's LambdaRank
    training dataset. See module docstring for the reuse-first principle."""
    r = FeatureRegistry()

    # ================================================================
    # Identifiers -- never part of the model's feature matrix.
    # ================================================================
    r.register(FeatureSpec(
        name="query_id", group=FeatureGroup.IDENTIFIER, dtype=DataType.IDENTIFIER,
        source_file="training/query_group_generator.py", source_function="TBD (next component)",
        reused=False,
        engineering_justification=(
            "Anchors one recommendation event as one LightGBM LambdaRank "
            "query group. No such identifier exists on any current "
            "recommender object -- CandidateStore.StoredCandidateSet.set_id "
            "is the closest existing analogue but is only ever held "
            "in-memory (InMemoryCandidateStore), never persisted, so "
            "historical query_ids must be synthesised from "
            "recommendation_log's (user_id, recommended_at) grouping."
        ),
        reason_for_inclusion="Required by LightGBM's group-file format.",
        include_in_model_matrix=False,
    ))
    r.register(FeatureSpec(
        name="candidate_id", group=FeatureGroup.IDENTIFIER, dtype=DataType.IDENTIFIER,
        source_file="pipeline/recommender/services/candidate_filtering.py",
        source_function="MergedCandidate.problem_id",
        reused=True,
        reason_for_inclusion="The problem_id being scored; needed to join a later-attached label.",
        include_in_model_matrix=False,
    ))
    r.register(FeatureSpec(
        name="user_id", group=FeatureGroup.IDENTIFIER, dtype=DataType.IDENTIFIER,
        source_file="pipeline/recommender/models/user_graph.py", source_function="UserNode.user_id",
        reused=True,
        reason_for_inclusion="Row provenance / joins; a raw user_id must never be a trained-on feature (unbounded cardinality, would let the model memorise individual users instead of generalising).",
        include_in_model_matrix=False,
    ))
    r.register(FeatureSpec(
        name="recommended_at", group=FeatureGroup.IDENTIFIER, dtype=DataType.IDENTIFIER,
        source_file="database/postgres/db.py", source_function="save_recommendation_log",
        reused=True,
        reason_for_inclusion="Timestamp of the recommendation event -- needed for deterministic ordering and for label_generator.py to enforce recommended_at < submitted_at when attaching outcomes.",
        include_in_model_matrix=False,
    ))
    r.register(FeatureSpec(
        name="label", group=FeatureGroup.IDENTIFIER, dtype=DataType.NUMERICAL,
        source_file="training/label_generator.py", source_function="TBD (future component)",
        reused=False,
        engineering_justification="Placeholder column only, per explicit instruction not to invent a labeling strategy in this component. Populated by label_generator.py once implemented.",
        reason_for_inclusion="LightGBM LambdaRank requires a label column; present but unpopulated (NaN) until label_generator.py runs.",
        missing_strategy="unpopulated_until_label_generator_runs",
        include_in_model_matrix=False,   # it's the target, not a feature
    ))

    # ================================================================
    # User features
    # ================================================================
    r.register(FeatureSpec(
        name="cold_start", group=FeatureGroup.USER, dtype=DataType.BOOLEAN,
        source_file="pipeline/recommender/services/adaptive_difficulty.py",
        source_function="AdaptiveDifficultyController.build_plan (DifficultyPlan.is_cold_start)",
        reused=True,
        reason_for_inclusion="Directly gates which pool-weight table and difficulty mix the live recommender used for this event -- a first-order signal for why a candidate was even proposed.",
    ))
    r.register(FeatureSpec(
        name="seeded_user", group=FeatureGroup.USER, dtype=DataType.BOOLEAN,
        source_file="pipeline/recommender/models/user_graph.py", source_function="new: len(concept_edges) > 0 and len(solved_ids) == 0",
        reused=False,
        engineering_justification=(
            "Not exposed as a named flag anywhere today. This is exactly "
            "the condition the seeded-user cold-start bugfix "
            "(adaptive_difficulty.py's is_cold correction) targeted: a user "
            "with imported LeetCode/Codeforces mastery but zero platform "
            "submissions. Both concept_edges and solved_ids are already "
            "loaded on every UserGraph -- this is a zero-cost boolean over "
            "data the pipeline already has, and it lets the training data "
            "distinguish 'genuinely blank user' from 'seeded user' even "
            "though both currently pass cold_start=False after the fix."
        ),
        reason_for_inclusion="Distinguishes two populations the live pipeline now treats identically (both non-cold-start) but which may have different label distributions worth learning separately.",
    ))
    r.register(FeatureSpec(
        name="user_level", group=FeatureGroup.USER, dtype=DataType.CATEGORICAL,
        source_file="pipeline/recommender/services/adaptive_difficulty.py",
        source_function="AdaptiveDifficultyController._level (DifficultyPlan.level)",
        reused=True,
        expected_range="{beginner, mid, advanced}",
        reason_for_inclusion="Coarse skill bucket already driving the live pool-weight table; low-cardinality categorical, ideal for LightGBM's native categorical handling.",
    ))
    r.register(FeatureSpec(
        name="solved_count", group=FeatureGroup.USER, dtype=DataType.NUMERICAL,
        source_file="pipeline/recommender/models/user_graph.py", source_function="len(UserGraph.solved_ids)",
        reused=True, expected_range="[0, inf)",
        reason_for_inclusion="Overall platform experience/tenure signal.",
    ))
    r.register(FeatureSpec(
        name="average_mastery", group=FeatureGroup.USER, dtype=DataType.NUMERICAL,
        source_file="pipeline/recommender/services/adaptive_difficulty.py",
        source_function="AdaptiveDifficultyController._avg_mastery (DifficultyPlan.avg_mastery)",
        reused=True, expected_range="[0, 1]",
        reason_for_inclusion="Primary skill-level signal, already the basis of the live level/mix classification.",
    ))
    r.register(FeatureSpec(
        name="mastery_variance", group=FeatureGroup.USER, dtype=DataType.NUMERICAL,
        source_file="pipeline/recommender/models/user_graph.py", source_function="new: population variance over concept_edges[*].mastery_score",
        reused=False,
        engineering_justification=(
            "average_mastery alone can't distinguish a user who is uniformly "
            "mid-level everywhere from one who is expert in 2 topics and "
            "near-zero in 20 others -- these two users warrant very "
            "different candidate difficulty targeting even at the same "
            "average. Computed purely from concept_edges mastery scores "
            "already loaded on the graph; no new query."
        ),
        expected_range="[0, 0.25] (variance of values bounded in [0,1])",
        reason_for_inclusion="Captures distribution shape average_mastery discards.",
    ))
    r.register(FeatureSpec(
        name="mastered_topic_count", group=FeatureGroup.USER, dtype=DataType.NUMERICAL,
        source_file="pipeline/recommender/models/user_graph.py", source_function="len(UserGraph.mastered_concepts())",
        reused=True, expected_range="[0, inf)",
        reason_for_inclusion="Breadth of mastered material.",
    ))
    r.register(FeatureSpec(
        name="weak_topic_count", group=FeatureGroup.USER, dtype=DataType.NUMERICAL,
        source_file="pipeline/recommender/services/adaptive_difficulty.py", source_function="DifficultyPlan.n_weak",
        reused=True, expected_range="[0, inf)",
        reason_for_inclusion="Already boosts WeaknessPool's live weight; same signal is relevant to the model.",
    ))
    r.register(FeatureSpec(
        name="urgent_topic_count", group=FeatureGroup.USER, dtype=DataType.NUMERICAL,
        source_file="pipeline/recommender/services/adaptive_difficulty.py", source_function="DifficultyPlan.n_urgent",
        reused=True, expected_range="[0, inf)",
        reason_for_inclusion="HLR-forgetting-curve pressure at the user level.",
    ))
    r.register(FeatureSpec(
        name="overdue_review_count", group=FeatureGroup.USER, dtype=DataType.NUMERICAL,
        source_file="pipeline/recommender/services/adaptive_difficulty.py", source_function="DifficultyPlan.n_overdue",
        reused=True, expected_range="[0, inf)",
        reason_for_inclusion="SM-2 review pressure at the user level.",
    ))
    r.register(FeatureSpec(
        name="prerequisite_completion_ratio", group=FeatureGroup.GRAPH, dtype=DataType.NUMERICAL,
        source_file="pipeline/recommender/models/user_graph.py",
        source_function="new: mastered prereq edges / total prereq edges, via the already-cached _prereq_index()",
        reused=False,
        engineering_justification=(
            "Distinct from mastered_topic_count -- measures structural "
            "progress through the prerequisite DAG rather than raw topic "
            "count. Reuses the already-cached _prereq_index() (no new "
            "traversal, no new query); purely a ratio over data the pool "
            "layer's own locking logic already computes."
        ),
        expected_range="[0, 1]",
        reason_for_inclusion="Graph-structural progress signal not captured by any existing scalar.",
    ))

    # ================================================================
    # Problem features
    # ================================================================
    r.register(FeatureSpec(
        name="difficulty_score", group=FeatureGroup.PROBLEM, dtype=DataType.NUMERICAL,
        source_file="pipeline/recommender/pools/base_pool.py", source_function="Qdrant payload difficulty_score (via _problems_by_concept / _ann)",
        reused=True, expected_range="[0, 1]",
        reason_for_inclusion="Core difficulty signal already read by every pool.",
    ))
    r.register(FeatureSpec(
        name="topic_tag_count", group=FeatureGroup.PROBLEM, dtype=DataType.NUMERICAL,
        source_file="pipeline/recommender/pools/base_pool.py", source_function="new: len(Candidate.topic_tags)",
        reused=False,
        engineering_justification=(
            "The raw topic_tags list is a high-cardinality, multi-valued "
            "categorical -- a poor direct fit for LightGBM columns, and its "
            "per-tag mastery/urgency signal is already captured by the pair "
            "features avg_mastery/max_urgency below. topic_tag_count is a "
            "cheap scalar summary (how broadly this problem is tagged) that "
            "doesn't require encoding a multi-valued field."
        ),
        expected_range="[0, inf)",
        reason_for_inclusion="Cheap breadth signal without a high-cardinality categorical encoding problem.",
    ))
    r.register(FeatureSpec(
        name="company_tag_count", group=FeatureGroup.PROBLEM, dtype=DataType.NUMERICAL,
        source_file="pipeline/ingestion/schema.py", source_function="new: len(RELATIONAL_COLUMNS['companies']) via offline catalog join",
        reused=True,
        reason_for_inclusion="Real, already-ingested static problem metadata (50.4% coverage per ingestion schema); a scalar count avoids the same high-cardinality multi-valued issue as topic_tags.",
        expected_range="[0, inf)",
        inference_available=False,
        computational_cost=ComputationalCost.MEDIUM,
        missing_strategy="native_nan (49.6% of problems have no company data)",
    ))
    r.register(FeatureSpec(
        name="frequency", group=FeatureGroup.PROBLEM, dtype=DataType.NUMERICAL,
        source_file="pipeline/ingestion/schema.py", source_function="SIGNAL_COLUMNS['frequency']",
        reused=True, expected_range="[0, 1]",
        reason_for_inclusion="Interview-frequency popularity proxy, already computed offline.",
        inference_available=False, computational_cost=ComputationalCost.MEDIUM,
        missing_strategy="native_nan (50.4% coverage per ingestion schema)",
    ))
    r.register(FeatureSpec(
        name="rating", group=FeatureGroup.PROBLEM, dtype=DataType.NUMERICAL,
        source_file="pipeline/ingestion/schema.py", source_function="SIGNAL_COLUMNS['rating']",
        reused=True, expected_range="[0, 1]",
        reason_for_inclusion="Community like-ratio, a quality/engagement signal independent of difficulty.",
        inference_available=False, computational_cost=ComputationalCost.MEDIUM,
        missing_strategy="native_nan (50.4% coverage)",
    ))
    r.register(FeatureSpec(
        name="asked_by_faang", group=FeatureGroup.PROBLEM, dtype=DataType.BOOLEAN,
        source_file="pipeline/ingestion/schema.py", source_function="SIGNAL_COLUMNS['asked_by_faang']",
        reused=True,
        reason_for_inclusion="100% coverage, real interview-relevance signal.",
        inference_available=False, computational_cost=ComputationalCost.MEDIUM,
    ))

    # ================================================================
    # Pair features -- reused directly from the heuristic ranker's own
    # formulas (heuristic_ranker.py::HeuristicRanker.score_one), since
    # none of its sub-scores are persisted anywhere -- see design doc.
    # ================================================================
    r.register(FeatureSpec(
        name="predicted_success", group=FeatureGroup.PAIR, dtype=DataType.NUMERICAL,
        source_file="pipeline/recommender/services/candidate_filtering.py",
        source_function="CandidateFilteringLayer._default_success_estimator",
        reused=True, expected_range="(0, 1)",
        reason_for_inclusion="The ZPD-band success estimate every candidate in the dataset already passed.",
    ))
    r.register(FeatureSpec(
        name="avg_mastery_pair", group=FeatureGroup.PAIR, dtype=DataType.NUMERICAL,
        source_file="pipeline/recommender/services/candidate_filtering.py",
        source_function="CandidateFilteringLayer._topic_mastery_and_urgency (cached on MergedCandidate)",
        reused=True, expected_range="[0, 1]",
        missing_strategy="native_nan (None when no concept_edges exist for this candidate's topic_tags)",
        reason_for_inclusion="User's mastery specifically over THIS candidate's topics -- distinct from the user-level average_mastery.",
    ))
    r.register(FeatureSpec(
        name="max_urgency_pair", group=FeatureGroup.PAIR, dtype=DataType.NUMERICAL,
        source_file="pipeline/recommender/services/candidate_filtering.py",
        source_function="CandidateFilteringLayer._topic_mastery_and_urgency (cached on MergedCandidate)",
        reused=True, expected_range="[0, 1]",
        missing_strategy="native_nan",
        reason_for_inclusion="HLR forgetting-curve urgency specific to this candidate's topics.",
    ))
    r.register(FeatureSpec(
        name="proximity", group=FeatureGroup.PAIR, dtype=DataType.NUMERICAL,
        source_file="pipeline/recommender/services/heuristic_ranker.py",
        source_function="HeuristicRanker.score_one (re-derived: not persisted by to_ranker_input, same formula)",
        reused=True, expected_range="[0, 1]",
        reason_for_inclusion="The ranker's own primary (highest-weighted) signal.",
    ))
    r.register(FeatureSpec(
        name="pool_agreement", group=FeatureGroup.PAIR, dtype=DataType.NUMERICAL,
        source_file="pipeline/recommender/services/heuristic_ranker.py", source_function="HeuristicRanker.score_one",
        reused=True, expected_range="[0, 1]",
        reason_for_inclusion="How many independent pools proposed this candidate.",
    ))
    r.register(FeatureSpec(
        name="urgency_boost", group=FeatureGroup.PAIR, dtype=DataType.NUMERICAL,
        source_file="pipeline/recommender/services/heuristic_ranker.py", source_function="HeuristicRanker.score_one",
        reused=True, expected_range="[0, 1]",
        reason_for_inclusion="The ranker's own (clamped) urgency term.",
    ))
    r.register(FeatureSpec(
        name="similarity_score", group=FeatureGroup.PAIR, dtype=DataType.NUMERICAL,
        source_file="pipeline/recommender/services/heuristic_ranker.py", source_function="HeuristicRanker.score_one",
        reused=True, expected_range="[0, 1]",
        missing_strategy="native value 0.0 for concept-based pools by construction (not missing data -- see registry note)",
        reason_for_inclusion="ANN pool confidence; naturally near-zero/sparse for the 6 non-ANN pools -- documented, not imputed away.",
    ))
    r.register(FeatureSpec(
        name="difficulty_alignment", group=FeatureGroup.PAIR, dtype=DataType.NUMERICAL,
        source_file="pipeline/recommender/services/heuristic_ranker.py", source_function="HeuristicRanker.score_one",
        reused=True, expected_range="[0, 1]",
        reason_for_inclusion="The ranker's own difficulty-vs-mastery alignment term. Included despite documented conceptual overlap with proximity -- see REJECTED_FEATURES note on rank_score for how redundancy is handled: kept (both are real, distinct-formula signals the ranker already computes) and left for LightGBM's regularisation to weigh, per the explicit instruction not to hand-remove correlated features without clear redundancy.",
    ))
    r.register(FeatureSpec(
        name="mastery_difficulty_gap", group=FeatureGroup.PAIR, dtype=DataType.NUMERICAL,
        source_file="pipeline/recommender/services/heuristic_ranker.py",
        source_function="new: avg_mastery_pair - difficulty_score (signed)",
        reused=False,
        engineering_justification=(
            "difficulty_alignment already exists as 1 - |mastery - difficulty| "
            "(symmetric, direction-blind). The SIGNED gap tells the model "
            "WHICH direction the mismatch runs (problem harder than mastery, "
            "vs. easier) -- information the absolute-value formulation "
            "structurally discards. Cheap (reuses the same two already-"
            "computed inputs), no leakage, genuinely non-redundant with the "
            "existing symmetric feature."
        ),
        expected_range="[-1, 1]",
        reason_for_inclusion="Directional mastery-difficulty interaction the existing symmetric feature can't express.",
    ))
    r.register(FeatureSpec(
        name="best_pool_score", group=FeatureGroup.PAIR, dtype=DataType.NUMERICAL,
        source_file="pipeline/recommender/services/candidate_filtering.py", source_function="MergedCandidate.best_score",
        reused=True, expected_range="[0, 1]",
        missing_strategy="native value 0.0 for concept-based pools by construction",
        reason_for_inclusion="Raw pool-local relevance score feeding similarity_score.",
    ))

    # ================================================================
    # Pool features
    # ================================================================
    r.register(FeatureSpec(
        name="pool_count", group=FeatureGroup.POOL, dtype=DataType.NUMERICAL,
        source_file="pipeline/recommender/services/candidate_filtering.py", source_function="MergedCandidate.pool_count",
        reused=True, expected_range="[1, 7]",
        reason_for_inclusion="Independent-agreement count.",
    ))
    for _pool in ("A", "B_C", "D", "E", "F", "G", "vector"):
        r.register(FeatureSpec(
            name=f"from_pool_{_pool}", group=FeatureGroup.POOL, dtype=DataType.BOOLEAN,
            source_file="pipeline/recommender/services/candidate_filtering.py",
            source_function=f"new: '{_pool}' in MergedCandidate.pool_sources",
            reused=False,
            engineering_justification=(
                "pool_sources is a multi-valued categorical (a list of up to "
                "7 pool names) -- a poor direct fit for LightGBM. One boolean "
                "per pool is the standard, interpretable, SHAP-friendly "
                "encoding of exactly the same information already on "
                "MergedCandidate.pool_sources, not a new computation."
            ),
            reason_for_inclusion=f"Whether pool {_pool} specifically proposed this candidate -- pools have different quality/intent (e.g. F=stretch vs D=weakness), which a single pool_count can't distinguish.",
        ))
    r.register(FeatureSpec(
        name="max_pool_weight", group=FeatureGroup.POOL, dtype=DataType.NUMERICAL,
        source_file="pipeline/recommender/services/adaptive_difficulty.py",
        source_function="new: max(DifficultyPlan.weight_of(p) for p in MergedCandidate.pool_sources)",
        reused=True,
        expected_range="[0, 1]",
        reason_for_inclusion="How strongly this user's own difficulty plan favoured the strongest of the pools that proposed this candidate -- reuses DifficultyPlan.weight_of() directly, no new computation, just a per-candidate lookup against already-computed per-pool weights.",
    ))

    # ================================================================
    # Diagnostic -- NOT part of the trained model matrix. See
    # REJECTED_FEATURES' rank_score_as_training_feature entry for why.
    # ================================================================
    r.register(FeatureSpec(
        name="current_heuristic_rank_score", group=FeatureGroup.DIAGNOSTIC, dtype=DataType.NUMERICAL,
        source_file="pipeline/recommender/services/heuristic_ranker.py", source_function="HeuristicRanker.score_one",
        reused=True, expected_range="[0, 1]",
        reason_for_inclusion="Comparison baseline only -- lets a future evaluation compare the learned model's ranking against the current production heuristic on identical rows.",
        leakage_risk=LeakageRisk.MEDIUM,
        include_in_model_matrix=False,
    ))

    return r

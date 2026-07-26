"""
pipeline/recommender/services/lightgbm_ranker.py

Production inference service for the trained LightGBM LambdaRank model.
Sits exactly where HeuristicRanker currently sits in
recommend.py::get_recommendations() -- it REORDERS the candidates
PoolGenerationOrchestrator/CandidateFilteringLayer already produced. It
does not generate, filter, or diversity-mix anything itself:

    Candidate Generation -> FeatureExtractor -> LightGBM -> Final Ranking

Every feature value comes from training/feature_extractor.py::extract_features()
-- the exact same function training/dataset_generator.py calls to build the
training set -- so the inference feature vector is guaranteed to match the
training feature vector column-for-column. No feature formula is
duplicated here.

Ranker selection is config-only (RANKER env var: heuristic / lightgbm /
hybrid, default heuristic) via rank_candidates(), the one entry point
recommend.py calls. Any failure to load the model, a feature-schema
mismatch, or an inference error falls back to the heuristic ranker
automatically, with the reason logged -- the recommender must never fail
because the ML model is unavailable.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Optional

import lightgbm as lgb
import pandas as pd

from pipeline.recommender.models.user_graph import UserGraph
from pipeline.recommender.services.adaptive_difficulty import DifficultyPlan
from pipeline.recommender.services.candidate_filtering import CandidateFilteringLayer, MergedCandidate
from pipeline.recommender.services.heuristic_ranker import HeuristicRanker

from training.config import LIGHTGBM_MODEL_PATH, MODEL_METADATA_PATH
from training.feature_extractor import extract_features
from training.feature_registry import build_default_registry
from training.model_metadata import load_metadata

log = logging.getLogger(__name__)

RANKER_ENV_VAR = "RANKER"
DEFAULT_RANKER_MODE = "heuristic"
VALID_RANKER_MODES = ("heuristic", "lightgbm", "hybrid")


class LightGBMRankerError(Exception):
    """Raised for any condition the caller must fall back to the heuristic
    ranker for: model missing/unreadable, feature-schema mismatch, or an
    inference failure."""


def get_ranker_mode() -> str:
    """
    Reads RANKER from the environment on every call (not cached at import
    time) so switching modes is purely a configuration change -- no code
    change, no process restart required for tests, and a bad/unknown value
    degrades to the safe default rather than raising.
    """
    mode = os.getenv(RANKER_ENV_VAR, DEFAULT_RANKER_MODE).strip().lower()
    if mode not in VALID_RANKER_MODES:
        log.warning("Unknown RANKER=%r, falling back to %r", mode, DEFAULT_RANKER_MODE)
        return DEFAULT_RANKER_MODE
    return mode


@dataclass
class ScoredCandidate:
    candidate: MergedCandidate
    score: float


class LightGBMRanker:
    """
    Usage:
        ranker = LightGBMRanker()
        ranker.load_model()
        scored = ranker.score_candidates(candidates, graph, plan)
        reranked = ranker.rerank(candidates, graph, plan)

    Prefer get_lightgbm_ranker() below in production code -- it caches a
    single instance per process so the model is loaded from disk at most
    once, never per request.
    """

    def __init__(self, model_path=None, metadata_path=None):
        # Reads the module-level defaults at CALL time (statement in the
        # constructor body), not as a default-argument value -- default
        # arguments bind once at import time, which would silently ignore
        # any later change to LIGHTGBM_MODEL_PATH/MODEL_METADATA_PATH
        # (e.g. test monkeypatching, or a future config override). See
        # get_lightgbm_ranker() below for the same reasoning.
        self.model_path = model_path if model_path is not None else LIGHTGBM_MODEL_PATH
        self.metadata_path = metadata_path if metadata_path is not None else MODEL_METADATA_PATH
        self._booster: Optional[lgb.Booster] = None
        self._feature_names: list[str] = []
        self._categorical_names: list[str] = []
        self._metadata: Optional[dict] = None
        self._registry = build_default_registry()

    @property
    def is_loaded(self) -> bool:
        return self._booster is not None

    def load_model(self) -> None:
        """
        Loads the booster and validates its stored feature order against
        FeatureRegistry.model_matrix_features() -- the single source of
        truth for what the trainable feature vector looks like. Raises
        LightGBMRankerError (never a raw exception) on either a missing/
        unreadable model file or a feature-order mismatch; both are
        exactly the conditions callers must fall back to heuristic for.

        Also loads the sidecar model_metadata.json (training/model_metadata.py),
        best-effort: a missing/unreadable metadata file does NOT fail the
        model load (get_model_info() just returns None afterward) -- the
        model is still fully usable for inference without its provenance
        record, and a model file copied without its metadata sidecar is a
        real, expected condition, not something to fabricate values for.
        """
        try:
            booster = lgb.Booster(model_file=str(self.model_path))
        except Exception as exc:
            raise LightGBMRankerError(
                f"failed to load LightGBM model from {self.model_path}: {exc}"
            ) from exc

        feature_names = [f.name for f in self._registry.model_matrix_features()]
        model_features = list(booster.feature_name())
        if model_features != feature_names:
            raise LightGBMRankerError(
                "LightGBM model's stored feature order does not match "
                "FeatureRegistry.model_matrix_features() -- refusing to "
                f"serve predictions with a mismatched schema "
                f"(model has {len(model_features)} features, registry has {len(feature_names)})"
            )

        self._booster = booster
        self._feature_names = feature_names
        self._categorical_names = [f.name for f in self._registry.categorical_features()]
        self._metadata = load_metadata(self.metadata_path)

        if self._metadata is not None:
            log.info(
                "LightGBM ranker loaded: version=%s trained=%s features=%d "
                "best_iteration=%s dataset_version=%s",
                self._metadata.get("model_version"),
                self._metadata.get("training_timestamp"),
                len(feature_names),
                self._metadata.get("best_iteration"),
                self._metadata.get("dataset_version"),
            )
        else:
            log.warning(
                "LightGBM model loaded from %s (%d features) but no metadata "
                "found at %s -- version/provenance info unavailable",
                self.model_path, len(feature_names), self.metadata_path,
            )

    def get_model_info(self) -> Optional[dict]:
        """The loaded model_metadata.json contents (model_version,
        training_timestamp, git_commit_hash, dataset_version, training
        metrics, etc.), or None if load_model() hasn't succeeded yet or no
        metadata file was found alongside the model."""
        return self._metadata

    def score_candidates(
        self, candidates: list[MergedCandidate], graph: UserGraph, plan: DifficultyPlan,
        catalog_metadata_by_problem_id: Optional[dict] = None,
    ) -> list[ScoredCandidate]:
        """
        Extracts the training feature vector for every candidate (reusing
        training/feature_extractor.py::extract_features exactly, no
        duplicated formula) and scores it with the loaded booster. Raises
        LightGBMRankerError if the model isn't loaded or inference fails.
        """
        if not self.is_loaded:
            raise LightGBMRankerError("score_candidates() called before load_model() succeeded")
        if not candidates:
            return []

        heuristic_ranker = HeuristicRanker()   # feeds current_heuristic_rank_score/proximity/etc as INPUT features, same as at training time
        filtering_layer = CandidateFilteringLayer(graph)
        catalog_metadata_by_problem_id = catalog_metadata_by_problem_id or {}

        rows = []
        for candidate in candidates:
            catalog_row = catalog_metadata_by_problem_id.get(candidate.problem_id)
            extracted = extract_features(
                graph, plan, candidate,
                ranker=heuristic_ranker, filtering_layer=filtering_layer,
                catalog_metadata=catalog_row,
            )
            rows.append(extracted.to_flat_dict())

        try:
            df = pd.DataFrame(rows)
            missing = set(self._feature_names) - set(df.columns)
            if missing:
                raise LightGBMRankerError(f"extracted features are missing columns: {sorted(missing)}")

            X = df[self._feature_names].copy()
            for col in self._feature_names:
                if col in self._categorical_names:
                    X[col] = X[col].astype("category")
                else:
                    X[col] = X[col].astype("float64")

            scores = self._booster.predict(X)
        except LightGBMRankerError:
            raise
        except Exception as exc:
            raise LightGBMRankerError(f"LightGBM inference failed: {exc}") from exc

        if any(pd.isna(scores)):
            raise LightGBMRankerError("LightGBM produced NaN prediction(s)")

        return [ScoredCandidate(candidate=c, score=float(s)) for c, s in zip(candidates, scores)]

    def rerank(
        self, candidates: list[MergedCandidate], graph: UserGraph, plan: DifficultyPlan,
        catalog_metadata_by_problem_id: Optional[dict] = None,
    ) -> list[MergedCandidate]:
        """
        Scores every candidate and returns them sorted best-first. Every
        candidate object is returned exactly as given -- pool_sources,
        topic_tags, predicted_success, and every other piece of metadata
        are untouched; only the ORDER changes.
        """
        scored = self.score_candidates(candidates, graph, plan, catalog_metadata_by_problem_id)
        scored.sort(key=lambda sc: sc.score, reverse=True)
        return [sc.candidate for sc in scored]


# ---------------------------------------------------------------------------
# Process-wide singleton: the model is read from disk (and its schema
# validated) at most ONCE per process, whether that attempt succeeds or
# fails -- a persistently-missing/broken model does not retry disk IO on
# every request either, it just keeps re-raising the same cached error.
# ---------------------------------------------------------------------------

_singleton_lock = threading.Lock()
_singleton: Optional[LightGBMRanker] = None
_load_attempted = False
_load_error: Optional[str] = None


def get_lightgbm_ranker() -> LightGBMRanker:
    """Returns the process-wide LightGBMRanker, loading it on first call
    only. Raises LightGBMRankerError (using the cached reason) on every
    call if the one load attempt failed."""
    global _singleton, _load_attempted, _load_error
    with _singleton_lock:
        if not _load_attempted:
            _load_attempted = True
            # Re-reads the module-level LIGHTGBM_MODEL_PATH at call time
            # rather than relying on LightGBMRanker.__init__'s default
            # parameter value -- Python binds default arguments once, at
            # class-definition (import) time, so `LightGBMRanker()` alone
            # would silently ignore any later change to this module's
            # LIGHTGBM_MODEL_PATH (e.g. test monkeypatching, or a future
            # config override) and keep using whatever it resolved to at
            # import time.
            candidate = LightGBMRanker(model_path=LIGHTGBM_MODEL_PATH)
            try:
                candidate.load_model()
                _singleton = candidate
            except LightGBMRankerError as exc:
                _load_error = str(exc)
                log.warning("LightGBM ranker unavailable: %s", exc)

    if _singleton is None:
        raise LightGBMRankerError(_load_error or "LightGBM model not loaded")
    return _singleton


def reset_lightgbm_ranker_cache() -> None:
    """Test-only hook: clears the process-wide singleton/cached load
    result so a test can force a fresh load_model() attempt (e.g. against
    a different/missing model path). Production code never needs this."""
    global _singleton, _load_attempted, _load_error
    with _singleton_lock:
        _singleton = None
        _load_attempted = False
        _load_error = None


# ---------------------------------------------------------------------------
# Single entry point recommend.py calls instead of a hardcoded HeuristicRanker
# ---------------------------------------------------------------------------

def _percentile_ranks(ordered_problem_ids: list[str]) -> dict[str, float]:
    """1.0 for the best item, descending toward 0.0 for the worst -- a
    scale-free way to combine two rankers whose raw scores aren't on
    comparable scales (HeuristicRanker's bounded ~[0,1] weighted sum vs.
    LightGBM's unbounded raw lambdarank score)."""
    n = len(ordered_problem_ids)
    if n == 0:
        return {}
    if n == 1:
        return {ordered_problem_ids[0]: 1.0}
    return {pid: 1.0 - (i / (n - 1)) for i, pid in enumerate(ordered_problem_ids)}


def rank_candidates(
    ranker_rows: list[dict],
    merged_candidates: list[MergedCandidate],
    graph: UserGraph,
    plan: DifficultyPlan,
    k: int,
    catalog_metadata_by_problem_id: Optional[dict] = None,
) -> list[dict]:
    """
    Single entry point recommend.py calls in place of a hardcoded
    HeuristicRanker -- picks heuristic / lightgbm / hybrid per the RANKER
    env var (get_ranker_mode()) and ALWAYS returns the same shape
    HeuristicRanker.top_k() does (each ranker_input row plus a numeric
    "rank_score"), so nothing downstream of this call (DiversityMixer,
    the by_id/scores_by_id re-attachment in get_recommendations) needs to
    know which ranker produced it.

    Falls back to the heuristic ranker -- logging why -- on any
    model-missing / feature-schema-mismatch / inference failure. The
    recommender must never fail because the ML model is unavailable.
    """
    mode = get_ranker_mode()

    if mode == "heuristic":
        return HeuristicRanker().top_k(ranker_rows, k=k)

    try:
        lgbm = get_lightgbm_ranker()
        scored = lgbm.score_candidates(merged_candidates, graph, plan, catalog_metadata_by_problem_id)
    except LightGBMRankerError as exc:
        log.warning("RANKER=%s unavailable, falling back to heuristic: %s", mode, exc)
        return HeuristicRanker().top_k(ranker_rows, k=k)

    rows_by_id = {row["problem_id"]: row for row in ranker_rows}
    lgbm_score_by_id = {sc.candidate.problem_id: sc.score for sc in scored}

    if mode == "lightgbm":
        ranked_ids = sorted(lgbm_score_by_id, key=lambda pid: lgbm_score_by_id[pid], reverse=True)
        ranked = [
            {**rows_by_id[pid], "rank_score": lgbm_score_by_id[pid]}
            for pid in ranked_ids if pid in rows_by_id
        ]
        return ranked[:k]

    # mode == "hybrid": blend percentile ranks from both rankers, 50/50.
    heuristic_ranked = HeuristicRanker().rank(ranker_rows)   # list[RankedCandidate], best-first
    heuristic_pct = _percentile_ranks([rc.problem_id for rc in heuristic_ranked])
    lgbm_pct = _percentile_ranks(
        sorted(lgbm_score_by_id, key=lambda pid: lgbm_score_by_id[pid], reverse=True)
    )

    hybrid_scored = []
    for pid, row in rows_by_id.items():
        hybrid_score = 0.5 * heuristic_pct.get(pid, 0.0) + 0.5 * lgbm_pct.get(pid, 0.0)
        hybrid_scored.append({**row, "rank_score": hybrid_score})
    hybrid_scored.sort(key=lambda r: r["rank_score"], reverse=True)
    return hybrid_scored[:k]

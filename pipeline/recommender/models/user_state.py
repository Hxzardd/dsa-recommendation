"""
pipeline/recommender/models/user_state.py
==========================================
Projects the UserGraph into a 1920-d user state vector that lives in the
same embedding space as problems_full, enabling direct ANN queries.

PRIMARY PATH -- concept-centroid aggregation
---------------------------------------------
For each concept c the user has a ConceptEdge on:

    effective_weight(c) =
        BKT_mastery(c)              # P(L) from Shraddha's bkt_store
      x confidence(c)               # consistency: low=0.33 med=0.66 hi=1.0
      x (1 - HLR_urgency(c))        # forgotten concepts get LESS weight
                                    # so ANN pulls toward gaps naturally
      x recency_decay(c)            # e^(-lambda * days_since_last_attempt)
      x difficulty_weight(c, elo)   # difficulty_score from Qdrant payload
                                    # amplified or attenuated by ELO tier

    qs_subspace  (1792-d) = weighted_mean(concept_qs_centroids,  w)
    rgcn_subspace (128-d) = weighted_mean(concept_rgcn_centroids, w)
    user_state           = L2_norm(concat(qs_subspace, rgcn_subspace))

Why concept centroids instead of problem-history mean:
    Problem-history mean drowns out rare topics (50 array + 2 graph
    problems -> vector points at arrays). Concept centroids give each
    topic equal structural weight, then scale by the five-signal weight.

Why (1 - urgency) reduces weight:
    High urgency = user is forgetting this concept. Reducing its weight
    means the user vector moves AWAY from well-remembered concepts and
    TOWARD forgotten/weak ones. ANN then surfaces problems in those areas.

SECONDARY PATH -- problem-history blend (warm start only)
----------------------------------------------------------
When n_solved >= MIN_PROBLEMS, the final vector is:

    vector = ALPHA * concept_vector + (1 - ALPHA) * problem_history_vector

Problem history is still the edge-weighted mean but now acts as a
correction term, not the primary signal. ALPHA = 0.6 by default.

ELO -> difficulty_weight mapping:
    ELO < 1200  -> prefer difficulty_score 0.0-0.4  (easy problems)
    1200-1600   -> prefer difficulty_score 0.3-0.7  (medium)
    > 1600      -> prefer difficulty_score 0.6-1.0  (hard)
    Approximated by ELO tier via UserXP.current_level until real ELO exists.

Cold start (< MIN_PROBLEMS solved):
    Concept-centroid path only. Works well even with 0 solved problems
    if the user linked their LeetCode/CF account (UserTopicMastery seeded
    from LC GraphQL at onboarding).
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pipeline.recommender.models.user_graph import UserGraph, EdgeType

# ---------------------------------------------------------------------------
# Hyperparameters -- all env-overridable
# ---------------------------------------------------------------------------

LAMBDA_DAYS      = float(os.getenv("USER_STATE_LAMBDA",       "0.05"))
MIN_PROBLEMS     = int(os.getenv("USER_STATE_MIN_PROBLEMS",    "3"))
ALPHA_CONCEPT    = float(os.getenv("USER_STATE_ALPHA_CONCEPT", "0.6"))
QS_DIM           = int(os.getenv("USER_STATE_QS_DIM",         "1792"))
RGCN_DIM         = int(os.getenv("USER_STATE_RGCN_DIM",       "128"))
FULL_DIM         = QS_DIM + RGCN_DIM   # 1920

EDGE_TYPE_WEIGHT = {
    EdgeType.SOLVED:    1.0,
    EdgeType.ATTEMPTED: 0.4,
    EdgeType.EXPOSED:   0.1,
    EdgeType.SKIPPED:   0.0,
}

# ELO tier -> (preferred difficulty center, bandwidth)
# difficulty_score is 0-1 float from offline ingestion
_ELO_TIERS = [
    (1200, 0.25, 0.30),   # < 1200  -> center=0.25  bw=0.30
    (1600, 0.50, 0.35),   # 1200-1600 -> center=0.50
    (9999, 0.75, 0.35),   # > 1600  -> center=0.75
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class UserStateVector:
    user_id:           str
    vector:            Optional[np.ndarray]   # (1920,) L2-normed, None=no data
    qs_subspace:       Optional[np.ndarray]   # (1792,) for diagnostics
    rgcn_subspace:     Optional[np.ndarray]   # (128,)  for diagnostics

    # Signals passed to candidate pools
    mastered_concepts:  list[str]        = field(default_factory=list)
    weak_concepts:      list[str]        = field(default_factory=list)
    urgent_concepts:    list[str]        = field(default_factory=list)
    solved_ids:         set[str]         = field(default_factory=set)
    exposed_ids:        dict[str, float] = field(default_factory=dict)
    deprioritised_ids:  set[str]         = field(default_factory=set)
    n_solved:           int              = 0
    is_cold_start:      bool             = False
    built_at:           float            = field(default_factory=time.time)

    # Per-concept effective weights for debug / explainability
    concept_weights:    dict[str, float] = field(default_factory=dict)

    def is_valid(self) -> bool:
        return self.vector is not None and not np.any(np.isnan(self.vector))

    def to_query_vector(self) -> Optional[list[float]]:
        return self.vector.tolist() if self.vector is not None else None


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class UserStateBuilder:
    """
    Converts a UserGraph into a UserStateVector.

    Instantiate once and reuse -- concept centroid cache lives here.

    Usage:
        builder = UserStateBuilder(qdrant_client)
        graph   = user_graph_service.get(user_id)
        state   = builder.build(graph)
    """

    def __init__(self, qdrant_client):
        self._q = qdrant_client
        # concept centroid cache: slug -> (qs_vec 1792-d, rgcn_vec 128-d)
        self._centroid_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        # difficulty_score cache: slug -> float 0-1
        self._difficulty_cache: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def build(self, graph: UserGraph) -> UserStateVector:
        n_solved   = sum(1 for e in graph.problem_edges.values()
                         if e.edge_type == EdgeType.SOLVED)
        is_cold    = n_solved < MIN_PROBLEMS
        elo        = graph.user.elo_rating or 1200

        # Step 1: concept-centroid vector (primary)
        concept_vec, concept_weights = self._concept_vector(graph, elo)

        # Step 2: problem-history vector (secondary blend, warm start only)
        prob_vec = None
        if not is_cold:
            prob_vec = self._problem_history_vector(graph)

        # Step 3: blend
        qs_sub, rgcn_sub = self._blend(concept_vec, prob_vec, is_cold)

        # Step 4: fuse + L2-norm
        vector = self._fuse(qs_sub, rgcn_sub)

        # Store vector back on user node
        if vector is not None:
            graph.user.embedding = vector.tolist()

        return UserStateVector(
            user_id           = graph.user.user_id,
            vector            = vector,
            qs_subspace       = qs_sub,
            rgcn_subspace     = rgcn_sub,
            mastered_concepts = graph.mastered_concepts(),
            weak_concepts     = graph.weak_concepts(),
            urgent_concepts   = graph.urgent_concepts(),
            solved_ids        = graph.solved_ids,
            exposed_ids       = graph.exposed_ids,
            deprioritised_ids = graph.deprioritised_ids,
            n_solved          = n_solved,
            is_cold_start     = is_cold,
            concept_weights   = concept_weights,
        )

    # ------------------------------------------------------------------
    # Step 1: concept-centroid vector
    # ------------------------------------------------------------------

    def _concept_vector(
        self, graph: UserGraph, elo: int
    ) -> tuple[Optional[tuple], dict[str, float]]:
        """
        Build (qs_1792, rgcn_128) from concept centroids weighted by
        BKT x confidence x (1-urgency) x recency x difficulty_weight(elo).
        """
        qs_acc   = np.zeros(QS_DIM,   dtype=np.float32)
        rgcn_acc = np.zeros(RGCN_DIM, dtype=np.float32)
        total_w  = 0.0
        weights  = {}

        for slug, edge in graph.concept_edges.items():
            w = self._effective_weight(edge, slug, elo)
            if w <= 0:
                continue

            qs_c, rgcn_c = self._get_centroid(slug)
            if qs_c is None:
                continue

            qs_acc   += w * qs_c
            rgcn_acc += w * (rgcn_c if rgcn_c is not None
                             else np.zeros(RGCN_DIM, dtype=np.float32))
            total_w  += w
            weights[slug] = round(w, 4)

        if total_w == 0:
            return (None, None), {}

        return (qs_acc / total_w, rgcn_acc / total_w), weights

    def _effective_weight(self, edge, slug: str, elo: int) -> float:
        """
        Five signals multiplied together:
            BKT mastery x confidence x (1 - HLR urgency) x recency x difficulty_weight
        """
        # 1. BKT mastery (P_L from Shraddha's store, already on edge)
        mastery = float(edge.mastery_score or 0.0)
        if mastery == 0:
            return 0.0

        # 2. Confidence (consistency over last 5 sessions)
        #    low=0.33, medium=0.66, high=1.0 -- already mapped in service
        confidence = float(getattr(edge, "confidence", 0.66))

        # 3. HLR: (1 - urgency)
        #    urgency = 1 - recall_probability
        #    high urgency = forgotten = reduce weight so ANN pulls toward gap
        urgency   = float(getattr(edge, "urgency", 0.0))
        retention = 1.0 - urgency       # how much they still remember

        # 4. Recency decay: e^(-lambda * days_since_last_attempt)
        last_ts = getattr(edge, "last_attempted", None)
        if last_ts:
            days  = (time.time() - float(last_ts)) / 86400
            decay = math.exp(-LAMBDA_DAYS * days)
        else:
            decay = 0.5   # unknown recency -> half weight

        # 5. Difficulty weight: how well does concept difficulty match ELO?
        #    difficulty_score 0-1 from offline ingestion via Qdrant payload
        diff_score = self._concept_difficulty(slug)
        diff_w     = self._difficulty_weight(diff_score, elo)

        return mastery * confidence * retention * decay * diff_w

    def _difficulty_weight(self, diff_score: float, elo: int) -> float:
        """
        Gaussian-shaped weight: peaks at the ELO-appropriate difficulty center.
        A beginner (ELO<1200) gets high weight for easy concepts (diff~0.25).
        An expert (ELO>1600) gets high weight for hard concepts (diff~0.75).
        """
        center, bw = 0.5, 0.35   # default: mid difficulty
        for elo_threshold, c, b in _ELO_TIERS:
            if elo < elo_threshold:
                center, bw = c, b
                break
        # Gaussian: exp(-(x-center)^2 / (2*bw^2)), min-clipped at 0.1
        w = math.exp(-((diff_score - center) ** 2) / (2 * bw ** 2))
        return max(0.1, w)   # always some small weight so concepts aren't zeroed

    # ------------------------------------------------------------------
    # Step 2: problem-history vector (secondary)
    # ------------------------------------------------------------------

    def _problem_history_vector(
        self, graph: UserGraph
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """
        Edge-weighted mean of full/rgcn embeddings of solved+attempted problems.
        Weight = edge_type_weight x normalised_score x recency_decay.
        This is the OLD primary path, now used only as a blend correction.
        """
        edges = [e for e in graph.problem_edges.values()
                 if EDGE_TYPE_WEIGHT.get(e.edge_type, 0) > 0]
        if not edges:
            return None

        pids    = [e.problem_id for e in edges]
        weights = [self._problem_edge_weight(e) for e in edges]
        if sum(weights) == 0:
            return None

        full_vecs = self._fetch_vectors(pids, "problems_full", FULL_DIM)
        rgcn_vecs = self._fetch_vectors(pids, "problems_rgcn", RGCN_DIM)

        qs_acc   = np.zeros(QS_DIM,   dtype=np.float32)
        rgcn_acc = np.zeros(RGCN_DIM, dtype=np.float32)
        total_w  = 0.0

        for e, w in zip(edges, weights):
            if w == 0:
                continue
            fv = full_vecs.get(e.problem_id)
            rv = rgcn_vecs.get(e.problem_id)
            if fv is not None:
                qs_acc  += w * fv[:QS_DIM]
                total_w += w
            if rv is not None:
                rgcn_acc += w * rv

        if total_w == 0:
            return None

        return qs_acc / total_w, rgcn_acc / total_w

    @staticmethod
    def _problem_edge_weight(edge) -> float:
        base   = EDGE_TYPE_WEIGHT.get(edge.edge_type, 0.0)
        score  = float(edge.normalised_score) if edge.normalised_score > 0 else 0.5
        days   = (time.time() - float(edge.timestamp)) / 86400
        decay  = math.exp(-LAMBDA_DAYS * days)
        return base * score * decay

    # ------------------------------------------------------------------
    # Step 3: blend concept + problem vectors
    # ------------------------------------------------------------------

    def _blend(
        self,
        concept_vec: tuple,
        prob_vec:    Optional[tuple],
        is_cold:     bool,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        c_qs, c_rgcn = concept_vec

        if c_qs is None:
            # No concept data at all -- fall back to problem history
            if prob_vec is not None:
                return prob_vec
            return None, None

        if prob_vec is None or is_cold:
            return c_qs, c_rgcn

        p_qs, p_rgcn = prob_vec
        a = ALPHA_CONCEPT
        qs   = a * c_qs   + (1 - a) * p_qs
        rgcn = a * c_rgcn + (1 - a) * p_rgcn
        return qs, rgcn

    # ------------------------------------------------------------------
    # Step 4: fuse + L2-norm
    # ------------------------------------------------------------------

    @staticmethod
    def _fuse(
        qs: Optional[np.ndarray], rgcn: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        if qs is None or rgcn is None:
            return None
        v = np.concatenate([qs, rgcn]).astype(np.float32)
        n = np.linalg.norm(v)
        return v / (n or 1.0)

    # ------------------------------------------------------------------
    # Concept centroid fetch + cache
    # ------------------------------------------------------------------

    def _get_centroid(
        self, slug: str
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if slug in self._centroid_cache:
            return self._centroid_cache[slug]

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            hits, _ = self._q.scroll(
                collection_name="problems_full",
                scroll_filter=Filter(must=[
                    FieldCondition(key="topic_tags",
                                   match=MatchValue(value=slug))
                ]),
                limit=50, with_vectors=True, with_payload=True,
            )
            if not hits:
                self._centroid_cache[slug] = (None, None)
                return None, None

            vecs, diff_scores = [], []
            for h in hits:
                v = h.vector
                if isinstance(v, dict):
                    v = next(iter(v.values()), None)
                if v is not None:
                    arr = np.asarray(v, dtype=np.float32)
                    if arr.shape == (FULL_DIM,) and not np.any(np.isnan(arr)):
                        vecs.append(arr)
                        # collect difficulty_score from payload for cache
                        pl = h.payload or {}
                        ds = pl.get("difficulty_score")
                        if ds is not None:
                            diff_scores.append(float(ds))

            if not vecs:
                self._centroid_cache[slug] = (None, None)
                return None, None

            centroid = np.mean(vecs, axis=0)
            qs_v     = centroid[:QS_DIM]
            rgcn_v   = centroid[QS_DIM:]

            # cache difficulty score for this concept
            if diff_scores and slug not in self._difficulty_cache:
                self._difficulty_cache[slug] = float(np.mean(diff_scores))

            self._centroid_cache[slug] = (qs_v, rgcn_v)
            return qs_v, rgcn_v

        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Centroid fetch failed for slug=%s: %s", slug, exc)
            self._centroid_cache[slug] = (None, None)
            return None, None

    def _concept_difficulty(self, slug: str) -> float:
        """
        Mean difficulty_score (0-1) of problems tagged with this concept.
        Populated as a side effect of _get_centroid. Returns 0.5 if unknown.
        """
        if slug not in self._difficulty_cache:
            self._get_centroid(slug)   # triggers cache population
        return self._difficulty_cache.get(slug, 0.5)

    # ------------------------------------------------------------------
    # Problem vector fetch
    # ------------------------------------------------------------------

    def _fetch_vectors(
        self, problem_ids: list[str], collection: str, dim: int
    ) -> dict[str, np.ndarray]:
        if not problem_ids:
            return {}
        result = {}
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchAny
            hits, _ = self._q.scroll(
                collection_name=collection,
                scroll_filter=Filter(must=[
                    FieldCondition(key="problem_id",
                                   match=MatchAny(any=problem_ids))
                ]),
                limit=len(problem_ids),
                with_vectors=True,
                with_payload=True,
            )
            for h in hits:
                pid = str((h.payload or {}).get("problem_id", ""))
                if not pid:
                    continue
                v = h.vector
                if isinstance(v, dict):
                    v = next(iter(v.values()), None)
                if v is not None:
                    arr = np.asarray(v, dtype=np.float32)
                    if arr.shape == (dim,) and not np.any(np.isnan(arr)):
                        result[pid] = arr
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Vector fetch failed collection=%s: %s", collection, exc)
        return result

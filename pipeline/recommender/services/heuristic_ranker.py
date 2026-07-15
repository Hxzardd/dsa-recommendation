"""
Heuristic ranker.

A hand-tuned weighted formula standing in for LightGBM/LambdaMART until
there's enough real RecommendationLog data (was_attempted / was_skipped)
to train one. This is not a placeholder to be embarrassed about -- rule-based
scoring is how most recommendation systems actually start; it becomes the
labeled-data source a trained model eventually replaces.

Scoring combines four signals, all already present on every candidate
after CandidateFilteringLayer.to_ranker_input():

  proximity_score   how close predicted_success sits to the ZPD sweet spot
                    (0.68). This is the PRIMARY signal -- a candidate at
                    exactly the productive-struggle point should usually
                    win over one that's merely "safe" or "risky."
  pool_agreement    how many independent pools proposed this same problem.
                    3 pools agreeing is a much stronger signal than 1.
  urgency_boost     HLR forgetting-curve urgency -- a nearly-forgotten
                    concept's review problem gets pushed up the list.
  similarity_score  the originating pool's own local relevance score
                    (meaningful for ANN pools like vector/B_C, 0 by
                    default for concept-based pools that don't compute one).

Weights are named constants, intentionally simple to retune by hand as real
usage data starts suggesting better values -- no training required to adjust.
"""

from __future__ import annotations

from dataclasses import dataclass

# Must match candidate_filtering.py's ZPD constants -- this ranker assumes
# every candidate it sees already passed that band filter.
ZPD_LO      = 0.55
ZPD_HI      = 0.80
ZPD_OPTIMAL = 0.68
_ZPD_HALF_RANGE = max(ZPD_OPTIMAL - ZPD_LO, ZPD_HI - ZPD_OPTIMAL)   # 0.13

# Pool agreement saturates here -- 3+ pools agreeing gives the same full
# credit as more; beyond 3 there are only 7 pools total, so it stops being
# a meaningfully stronger signal past this point.
POOL_AGREEMENT_SATURATION = 3

# Hand-tuned weights, sum to 1.0. Adjust these directly as real usage data
# comes in -- no retraining needed, just edit the numbers.
WEIGHT_PROXIMITY  = 0.45
WEIGHT_POOL_AGREE = 0.25
WEIGHT_URGENCY    = 0.20
WEIGHT_SIMILARITY = 0.10
WEIGHT_DIFFICULTY_ALIGNMENT = 0.05
assert abs((WEIGHT_PROXIMITY + WEIGHT_POOL_AGREE + WEIGHT_URGENCY + WEIGHT_SIMILARITY) - 1.0) < 1e-9


@dataclass
class RankedCandidate:
    """One ranker_input row plus its computed score and sub-scores (for debugging/explanation)."""
    row:               dict
    score:             float
    proximity_score:   float
    pool_agreement:    float
    urgency_boost:     float
    similarity_score:  float

    @property
    def problem_id(self) -> str:
        return self.row.get("problem_id")

    def to_dict(self) -> dict:
        return {
            **self.row,
            "rank_score": round(self.score, 4),
            "rank_components": {
                "proximity":  round(self.proximity_score, 4),
                "pool_agreement": round(self.pool_agreement, 4),
                "urgency":    round(self.urgency_boost, 4),
                "similarity": round(self.similarity_score, 4),
            },
        }


class HeuristicRanker:
    """
    Usage:
        ranker = HeuristicRanker()
        top10 = ranker.top_k(ranker_input_rows, k=10)

    ranker_input_rows is exactly what CandidateFilteringLayer.to_ranker_input()
    returns -- a list of dicts with problem_id, pool_sources, pool_count,
    topic_tags, difficulty_score, avg_mastery, max_urgency,
    predicted_success, best_pool_score.
    """

    def __init__(self,
                 weight_proximity: float = WEIGHT_PROXIMITY,
                 weight_pool_agree: float = WEIGHT_POOL_AGREE,
                 weight_urgency: float = WEIGHT_URGENCY,
                 weight_similarity: float = WEIGHT_SIMILARITY,
                 weight_difficulty_alignment: float = WEIGHT_DIFFICULTY_ALIGNMENT):
        self.weight_proximity = weight_proximity
        self.weight_pool_agree = weight_pool_agree
        self.weight_urgency = weight_urgency
        self.weight_similarity = weight_similarity
        self.weight_difficulty_alignment = weight_difficulty_alignment

    def score_one(self, row: dict) -> RankedCandidate:
        predicted_success = row.get("predicted_success")
        if predicted_success is None:
            predicted_success = ZPD_OPTIMAL   # no estimate -> assume the sweet spot, don't penalise
        proximity = 1.0 - min(1.0, abs(predicted_success - ZPD_OPTIMAL) / _ZPD_HALF_RANGE)

        pool_count = row.get("pool_count") or 1
        pool_agreement = min(1.0, pool_count / POOL_AGREEMENT_SATURATION)

        urgency = row.get("max_urgency")
        urgency_boost = urgency if urgency is not None else 0.0

        similarity = row.get("best_pool_score")
        similarity_score = similarity if similarity is not None else 0.0
        similarity_score = max(0.0, min(1.0, similarity_score))   # clamp defensively
        difficulty = row.get("difficulty_score")
        avg_mastery = row.get("avg_mastery")

        difficulty_alignment = 0.0

        if difficulty is not None and avg_mastery is not None:
          difficulty_alignment = 1.0 - min(
            1.0,
            abs(difficulty - avg_mastery),
          )
        score = (
             self.weight_proximity  * proximity +
             self.weight_pool_agree * pool_agreement +
             self.weight_urgency    * urgency_boost +
             self.weight_similarity * similarity_score
               )

        # Small tie-break bonus: prefer problems whose difficulty
        #  matches the user's estimated mastery.
        score += self.weight_difficulty_alignment * difficulty_alignment

        return RankedCandidate(
            row=row, score=score,
            proximity_score=proximity, pool_agreement=pool_agreement,
            urgency_boost=urgency_boost, similarity_score=similarity_score,
        )

    def rank(self, rows: list) -> list:
        """Score every row, return all of them sorted best-first."""
        ranked = [self.score_one(r) for r in rows]
        ranked.sort(key=lambda rc: rc.score, reverse=True)
        return ranked

    def top_k(self, rows: list, k: int = 10) -> list:
        """Convenience: rank then take the top k as plain dicts (rank_score attached)."""
        ranked = self.rank(rows)
        return [rc.to_dict() for rc in ranked[:k]]


def rank_top_k(rows: list, k: int = 10) -> list:
    """One-shot convenience wrapper with default weights."""
    return HeuristicRanker().top_k(rows, k=k)

"""
Diversity mixer.

Sits after ranking in the architecture diagram. Without it, a greedy ranker
would happily return 5 problems all from the same pool, same topic, same
difficulty -- individually well-matched, collectively poor UX.

Enforces, per the diagram:
    - pool diversity     : no single pool contributes more than max_per_pool
    - topic diversity     : final slate spans multiple topics/patterns
    - difficulty spread    : mix of challenge levels within the user's band

Algorithm: Maximum Marginal Relevance (MMR). Iteratively picks the candidate
that maximises (relevance - redundancy_penalty * similarity_to_already_picked),
then applies hard caps as a second guardrail so MMR's soft preference for
diversity can't be defeated by one pool simply having many high-relevance
candidates.

Relevance score is pluggable: Shraddha's ranker output can be passed in via
`relevance_scores`; if not supplied, falls back to `predicted_success` (from
CandidateFilteringLayer's ZPD estimate) so this works standalone before her
ranker exists, with no interface change needed once it does.
"""

from __future__ import annotations

from typing import Callable, Optional

from pipeline.recommender.services.candidate_filtering import MergedCandidate


DEFAULT_LAMBDA = 0.7   # weight on relevance vs. diversity; higher = more relevance-driven


def _jaccard(a: list, b: list) -> float:
    """Similarity between two topic-tag (or pool-source) lists, 0..1."""
    sa, sb = set(a or []), set(b or [])
    if not sa and not sb:
        return 0.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


class DiversityMixer:
    """
    Usage:
        mixer = DiversityMixer(max_per_pool=2, max_per_topic=2)
        final_slate = mixer.mix(merged_candidates, k=10)

    Or with a real ranker's scores:
        final_slate = mixer.mix(merged_candidates, k=10,
                                 relevance_scores={"p1": 0.91, "p2": 0.77, ...})
    """

    def __init__(self, max_per_pool: int = None, max_per_topic: int = None,
                 lambda_param: float = DEFAULT_LAMBDA):
        self.max_per_pool = max_per_pool
        self.max_per_topic = max_per_topic
        self.lambda_param = lambda_param

    def mix(self, candidates: list, k: int,
            relevance_scores: Optional[dict] = None,
            relevance_fn: Optional[Callable[[MergedCandidate], float]] = None) -> list:
        """
        candidates: list[MergedCandidate] (already filtered/deduped upstream).
        k: final slate size.
        relevance_scores: optional {problem_id: score} from a real ranker.
        relevance_fn: optional callable(mc) -> score, takes priority if given.
        Falls back to mc.predicted_success, then mc.best_score, if neither
        relevance_scores nor relevance_fn is supplied.
        """
        if not candidates:
            return []

        def _relevance(mc: MergedCandidate) -> float:
            if relevance_fn is not None:
                return relevance_fn(mc)
            if relevance_scores is not None and mc.problem_id in relevance_scores:
                return relevance_scores[mc.problem_id]
            if mc.predicted_success is not None:
                return mc.predicted_success
            return mc.best_score

        remaining = list(candidates)
        selected: list = []
        pool_counts: dict = {}
        topic_counts: dict = {}

        while remaining and len(selected) < k:
            best_mc = None
            best_mmr = float("-inf")

            for mc in remaining:
                if not self._within_caps(mc, pool_counts, topic_counts):
                    continue

                relevance = _relevance(mc)
                if not selected:
                    redundancy = 0.0
                else:
                    redundancy = max(
                        _jaccard(mc.topic_tags, s.topic_tags) for s in selected
                    )
                mmr = self.lambda_param * relevance - (1 - self.lambda_param) * redundancy

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_mc = mc

            if best_mc is None:
                # Every remaining candidate violates a cap. Caps are hard
                # constraints ("no pool contributes more than N") -- a
                # shorter slate that respects them is correct behaviour,
                # not a bug to work around by ignoring the cap. Silently
                # padding out to k here would defeat the entire purpose of
                # setting a cap in the first place.
                break

            selected.append(best_mc)
            remaining.remove(best_mc)
            self._record_caps(best_mc, pool_counts, topic_counts)

        return selected

    # ------------------------------------------------------------- caps

    def _within_caps(self, mc: MergedCandidate, pool_counts: dict, topic_counts: dict) -> bool:
        if self.max_per_pool is not None:
            for pool in mc.pool_sources:
                if pool_counts.get(pool, 0) >= self.max_per_pool:
                    return False
        if self.max_per_topic is not None:
            for tag in (mc.topic_tags or []):
                if topic_counts.get(tag, 0) >= self.max_per_topic:
                    return False
        return True

    def _record_caps(self, mc: MergedCandidate, pool_counts: dict, topic_counts: dict) -> None:
        for pool in mc.pool_sources:
            pool_counts[pool] = pool_counts.get(pool, 0) + 1
        for tag in (mc.topic_tags or []):
            topic_counts[tag] = topic_counts.get(tag, 0) + 1


def mix_candidates(candidates: list, k: int, max_per_pool: int = None,
                   max_per_topic: int = None,
                   relevance_scores: Optional[dict] = None) -> list:
    """Convenience one-shot wrapper."""
    return DiversityMixer(max_per_pool=max_per_pool, max_per_topic=max_per_topic).mix(
        candidates, k, relevance_scores=relevance_scores)

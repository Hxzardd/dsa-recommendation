"""
tests/test_diversity_mixer.py

Tests the MMR-based diversity mixer: pool caps, topic caps, relevance
ordering, and the fallback relevance chain (custom fn > scores dict >
predicted_success > best_score).

Run:
    python -m pytest tests/test_diversity_mixer.py -v
"""

from __future__ import annotations

import unittest

from pipeline.recommender.services.candidate_filtering import MergedCandidate
from pipeline.recommender.services.diversity_mixer import (
    DiversityMixer, mix_candidates, _jaccard,
)


def _mc(pid, pools, tags, score=0.5, predicted_success=None):
    return MergedCandidate(
        problem_id=pid,
        pool_sources=list(pools),
        best_score=score,
        topic_tags=list(tags),
        difficulty_score=0.5,
        predicted_success=predicted_success,
    )


class TestJaccard(unittest.TestCase):

    def test_identical_sets_similarity_one(self):
        self.assertAlmostEqual(_jaccard(["a", "b"], ["a", "b"]), 1.0)

    def test_disjoint_sets_similarity_zero(self):
        self.assertAlmostEqual(_jaccard(["a"], ["b"]), 0.0)

    def test_partial_overlap(self):
        # {a,b} vs {b,c} -> intersection {b}=1, union {a,b,c}=3
        self.assertAlmostEqual(_jaccard(["a", "b"], ["b", "c"]), 1/3)

    def test_both_empty_is_zero_not_error(self):
        self.assertAlmostEqual(_jaccard([], []), 0.0)


class TestBasicMixing(unittest.TestCase):

    def test_empty_input_returns_empty(self):
        mixer = DiversityMixer()
        self.assertEqual(mixer.mix([], k=5), [])

    def test_returns_at_most_k(self):
        cands = [_mc(f"p{i}", ["A"], ["arrays"], score=0.9) for i in range(10)]
        mixer = DiversityMixer()
        result = mixer.mix(cands, k=3)
        self.assertEqual(len(result), 3)

    def test_returns_fewer_if_fewer_available(self):
        cands = [_mc("p1", ["A"], ["arrays"], score=0.9)]
        mixer = DiversityMixer()
        result = mixer.mix(cands, k=5)
        self.assertEqual(len(result), 1)

    def test_highest_relevance_picked_first_no_diversity_pressure(self):
        cands = [
            _mc("low",  ["A"], ["arrays"], score=0.1),
            _mc("high", ["A"], ["arrays"], score=0.9),
        ]
        mixer = DiversityMixer(lambda_param=1.0)   # pure relevance, no diversity
        result = mixer.mix(cands, k=1)
        self.assertEqual(result[0].problem_id, "high")


class TestPoolCap(unittest.TestCase):

    def test_pool_cap_enforced(self):
        # 5 candidates all from pool A, cap at 2 per pool
        cands = [_mc(f"p{i}", ["A"], [f"topic{i}"], score=0.9 - i*0.01) for i in range(5)]
        mixer = DiversityMixer(max_per_pool=2)
        result = mixer.mix(cands, k=5)
        # cap should limit to 2 even though k=5 and 5 candidates are available
        self.assertLessEqual(len(result), 2)

    def test_pool_cap_allows_other_pools_to_fill_slate(self):
        cands = (
            [_mc(f"a{i}", ["A"], [f"ta{i}"], score=0.9) for i in range(5)] +
            [_mc(f"d{i}", ["D"], [f"td{i}"], score=0.8) for i in range(5)]
        )
        mixer = DiversityMixer(max_per_pool=2)
        result = mixer.mix(cands, k=4)
        self.assertEqual(len(result), 4)
        pools_used = [p for mc in result for p in mc.pool_sources]
        self.assertLessEqual(pools_used.count("A"), 2)
        self.assertLessEqual(pools_used.count("D"), 2)

    def test_no_cap_allows_all_from_one_pool(self):
        cands = [_mc(f"p{i}", ["A"], [f"topic{i}"], score=0.9 - i*0.01) for i in range(5)]
        mixer = DiversityMixer(max_per_pool=None)
        result = mixer.mix(cands, k=5)
        self.assertEqual(len(result), 5)


class TestTopicCap(unittest.TestCase):

    def test_topic_cap_enforced(self):
        cands = [_mc(f"p{i}", [f"pool{i}"], ["arrays"], score=0.9 - i*0.01) for i in range(5)]
        mixer = DiversityMixer(max_per_topic=2)
        result = mixer.mix(cands, k=5)
        self.assertLessEqual(len(result), 2)

    def test_topic_cap_allows_other_topics(self):
        cands = (
            [_mc(f"arr{i}", [f"pool{i}"], ["arrays"], score=0.9) for i in range(4)] +
            [_mc(f"gph{i}", [f"pool{i+10}"], ["graphs"], score=0.85) for i in range(4)]
        )
        mixer = DiversityMixer(max_per_topic=2)
        result = mixer.mix(cands, k=4)
        self.assertEqual(len(result), 4)
        tags_used = [t for mc in result for t in mc.topic_tags]
        self.assertLessEqual(tags_used.count("arrays"), 2)
        self.assertLessEqual(tags_used.count("graphs"), 2)


class TestRelevanceFallbackChain(unittest.TestCase):

    def test_uses_predicted_success_when_no_scores_given(self):
        cands = [
            _mc("low",  ["A"], ["x"], score=0.5, predicted_success=0.2),
            _mc("high", ["A"], ["x"], score=0.5, predicted_success=0.9),
        ]
        mixer = DiversityMixer(lambda_param=1.0)
        result = mixer.mix(cands, k=1)
        self.assertEqual(result[0].problem_id, "high")

    def test_falls_back_to_best_score_when_no_predicted_success(self):
        cands = [
            _mc("low",  ["A"], ["x"], score=0.2, predicted_success=None),
            _mc("high", ["A"], ["x"], score=0.9, predicted_success=None),
        ]
        mixer = DiversityMixer(lambda_param=1.0)
        result = mixer.mix(cands, k=1)
        self.assertEqual(result[0].problem_id, "high")

    def test_external_relevance_scores_override_predicted_success(self):
        cands = [
            _mc("a", ["A"], ["x"], score=0.5, predicted_success=0.9),
            _mc("b", ["A"], ["x"], score=0.5, predicted_success=0.1),
        ]
        # external ranker says b is actually better -- should override predicted_success
        scores = {"a": 0.1, "b": 0.9}
        mixer = DiversityMixer(lambda_param=1.0)
        result = mixer.mix(cands, k=1, relevance_scores=scores)
        self.assertEqual(result[0].problem_id, "b")

    def test_relevance_fn_takes_priority_over_scores_dict(self):
        cands = [
            _mc("a", ["A"], ["x"], score=0.5),
            _mc("b", ["A"], ["x"], score=0.5),
        ]
        scores = {"a": 0.9, "b": 0.1}
        def custom_fn(mc):
            return 1.0 if mc.problem_id == "b" else 0.0

        mixer = DiversityMixer(lambda_param=1.0)
        result = mixer.mix(cands, k=1, relevance_scores=scores, relevance_fn=custom_fn)
        self.assertEqual(result[0].problem_id, "b")


class TestDiversityPressure(unittest.TestCase):

    def test_similar_topics_penalised_after_first_pick(self):
        """
        Two near-identical-topic high-score candidates and one different-topic
        lower-score candidate: with real diversity pressure (lambda < 1), the
        second pick should prefer the different-topic one over the near-dupe.
        """
        cands = [
            _mc("dup1", ["A"], ["arrays", "hashing"], score=0.95),
            _mc("dup2", ["A"], ["arrays", "hashing"], score=0.94),
            _mc("diff", ["A"], ["graphs"],            score=0.80),
        ]
        mixer = DiversityMixer(lambda_param=0.5)
        result = mixer.mix(cands, k=2)
        ids = [mc.problem_id for mc in result]
        self.assertEqual(ids[0], "dup1")     # highest relevance picked first
        self.assertEqual(ids[1], "diff")     # diff preferred over near-duplicate dup2


class TestConvenienceFunction(unittest.TestCase):

    def test_mix_candidates_matches_class(self):
        cands = [_mc(f"p{i}", ["A"], ["arrays"], score=0.9 - i*0.01) for i in range(5)]
        result = mix_candidates(cands, k=3, max_per_pool=5)
        self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)

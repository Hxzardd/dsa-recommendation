"""
tests/test_candidate_store.py

Tests the candidate storage layer: save/retrieve, TTL expiry, and the
stage_candidates() bridge from MergedCandidate objects to storage rows.

Run:
    python -m pytest tests/test_candidate_store.py -v
"""

from __future__ import annotations

import time
import unittest

from pipeline.recommender.services.candidate_store import (
    InMemoryCandidateStore, StoredCandidateSet, stage_candidates,
    CANDIDATE_SET_TTL_SECONDS,
)
from pipeline.recommender.services.candidate_filtering import MergedCandidate


def _mc(pid, pools=("A",), tags=("arrays",)):
    return MergedCandidate(
        problem_id=pid, pool_sources=list(pools), best_score=0.8,
        topic_tags=list(tags), difficulty_score=0.5, predicted_success=0.68,
    )


class TestSaveAndRetrieve(unittest.TestCase):

    def test_save_returns_stored_set(self):
        store = InMemoryCandidateStore()
        cs = store.save("u1", [{"problem_id": "p1"}], {"level": "mid"})
        self.assertIsInstance(cs, StoredCandidateSet)
        self.assertEqual(cs.user_id, "u1")

    def test_get_latest_returns_saved_set(self):
        store = InMemoryCandidateStore()
        store.save("u1", [{"problem_id": "p1"}], {"level": "mid"})
        cs = store.get_latest("u1")
        self.assertIsNotNone(cs)
        self.assertEqual(cs.candidates, [{"problem_id": "p1"}])

    def test_get_by_id(self):
        store = InMemoryCandidateStore()
        saved = store.save("u1", [{"problem_id": "p1"}], {})
        fetched = store.get(saved.set_id)
        self.assertEqual(fetched.set_id, saved.set_id)

    def test_get_latest_none_for_unknown_user(self):
        store = InMemoryCandidateStore()
        self.assertIsNone(store.get_latest("nobody"))

    def test_second_save_supersedes_first_for_same_user(self):
        store = InMemoryCandidateStore()
        store.save("u1", [{"problem_id": "p1"}], {})
        store.save("u1", [{"problem_id": "p2"}], {})
        latest = store.get_latest("u1")
        self.assertEqual(latest.candidates, [{"problem_id": "p2"}])


class TestExpiry(unittest.TestCase):

    def test_fresh_set_not_expired(self):
        store = InMemoryCandidateStore()
        cs = store.save("u1", [], {})
        self.assertFalse(cs.is_expired)

    def test_expired_set_not_returned_by_get_latest(self):
        store = InMemoryCandidateStore()
        cs = store.save("u1", [], {})
        cs.created_at = time.time() - CANDIDATE_SET_TTL_SECONDS - 10
        cs.expires_at = cs.created_at + CANDIDATE_SET_TTL_SECONDS
        self.assertIsNone(store.get_latest("u1"))

    def test_expired_set_not_returned_by_get(self):
        store = InMemoryCandidateStore()
        cs = store.save("u1", [], {})
        cs.created_at = time.time() - CANDIDATE_SET_TTL_SECONDS - 10
        cs.expires_at = cs.created_at + CANDIDATE_SET_TTL_SECONDS
        self.assertIsNone(store.get(cs.set_id))


class TestStageCandidatesBridge(unittest.TestCase):

    def test_already_flattened_dicts_pass_through(self):
        store = InMemoryCandidateStore()
        rows = [{"problem_id": "p1", "pool_sources": ["A"]}]
        cs = stage_candidates(store, "u1", rows, {"level": "mid"})
        self.assertEqual(cs.candidates, rows)

    def test_merged_candidates_require_graph(self):
        store = InMemoryCandidateStore()
        with self.assertRaises(ValueError):
            stage_candidates(store, "u1", [_mc("p1")], {}, graph=None)

    def test_merged_candidates_flattened_with_graph(self):
        from pipeline.recommender.models.user_graph import UserGraph, UserNode, ConceptEdge, EdgeType
        store = InMemoryCandidateStore()
        graph = UserGraph(user=UserNode(user_id="u1"))
        graph.add_concept_edge(ConceptEdge("arrays", EdgeType.MASTERED, mastery_score=0.7))

        cs = stage_candidates(store, "u1", [_mc("p1")], {"level": "mid"}, graph=graph)
        self.assertEqual(len(cs.candidates), 1)
        self.assertEqual(cs.candidates[0]["problem_id"], "p1")
        self.assertIn("avg_mastery", cs.candidates[0])


class TestSerialization(unittest.TestCase):

    def test_to_dict_json_serialisable(self):
        import json
        store = InMemoryCandidateStore()
        cs = store.save("u1", [{"problem_id": "p1"}], {"level": "mid"})
        raw = json.dumps(cs.to_dict())
        self.assertIsInstance(raw, str)

    def test_to_dict_has_expected_keys(self):
        store = InMemoryCandidateStore()
        cs = store.save("u1", [], {})
        d = cs.to_dict()
        for key in ("set_id", "user_id", "created_at", "expires_at",
                    "candidates", "difficulty_plan"):
            self.assertIn(key, d)


if __name__ == "__main__":
    unittest.main(verbosity=2)

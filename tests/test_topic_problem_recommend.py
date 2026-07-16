"""
tests/test_topic_problem_recommend.py

Tests pipeline/recommender/services/topic_recommend.py's
recommend_problems_for_topic: topic-based problem recommendation ranked
by relevance to the user's own level (BKT mastery vs each candidate's
difficulty_score), not a fixed band.

Run:
    python -m pytest tests/test_topic_problem_recommend.py -v
"""

from __future__ import annotations

import unittest

from pipeline.recommender.models.user_graph import UserGraph, UserNode, ConceptEdge, EdgeType
from pipeline.recommender.services.topic_recommend import recommend_problems_for_topic


class _Pt:
    def __init__(self, pid, tags, difficulty):
        self.id = pid
        self.payload = {"problem_id": pid, "topic_tags": tags, "difficulty_score": difficulty}


class FakeQdrant:
    def __init__(self, problems):
        self.problems = problems

    def scroll(self, collection_name, scroll_filter=None, limit=10,
               with_payload=True, with_vectors=False):
        want_tags = None
        if scroll_filter is not None:
            for cond in scroll_filter.must:
                if getattr(cond, "key", None) == "topic_tags":
                    want_tags = set(cond.match.any)
        out = []
        for p in self.problems:
            if want_tags is not None and not (set(p.payload["topic_tags"]) & want_tags):
                continue
            out.append(p)
            if len(out) >= limit:
                break
        return out, None


def _graph(mastery=None, solved=None):
    g = UserGraph(user=UserNode(user_id="u1"))
    if mastery is not None:
        g.add_concept_edge(ConceptEdge(concept_slug="array", edge_type=EdgeType.LEARNING,
                                       mastery_score=mastery))
    for pid in (solved or []):
        from pipeline.recommender.models.user_graph import ProblemEdge
        g.add_problem_edge(ProblemEdge(problem_id=pid, edge_type=EdgeType.SOLVED, timestamp=1.0))
    return g


class TestRecommendProblemsForTopic(unittest.TestCase):

    def test_matches_only_requested_topic(self):
        problems = [
            _Pt("p_array", ["array"], 0.4),
            _Pt("p_string", ["string"], 0.4),
        ]
        g = _graph(mastery=0.4)
        out = recommend_problems_for_topic(g, FakeQdrant(problems), "test", "array", n=10)
        ids = [c["problem_id"] for c in out]
        self.assertIn("p_array", ids)
        self.assertNotIn("p_string", ids)

    def test_low_mastery_user_gets_easier_problem_ranked_first(self):
        problems = [
            _Pt("p_easy", ["array"], 0.2),
            _Pt("p_hard", ["array"], 0.9),
        ]
        g = _graph(mastery=0.15)
        out = recommend_problems_for_topic(g, FakeQdrant(problems), "test", "array", n=10)
        self.assertEqual(out[0]["problem_id"], "p_easy")

    def test_high_mastery_user_gets_harder_problem_ranked_first(self):
        problems = [
            _Pt("p_easy", ["array"], 0.2),
            _Pt("p_hard", ["array"], 0.9),
        ]
        g = _graph(mastery=0.85)
        out = recommend_problems_for_topic(g, FakeQdrant(problems), "test", "array", n=10)
        self.assertEqual(out[0]["problem_id"], "p_hard")

    def test_solved_problems_excluded(self):
        problems = [_Pt("p_solved", ["array"], 0.4)]
        g = _graph(mastery=0.4, solved=["p_solved"])
        out = recommend_problems_for_topic(g, FakeQdrant(problems), "test", "array", n=10)
        self.assertEqual(out, [])

    def test_no_qdrant_returns_empty(self):
        g = _graph(mastery=0.4)
        out = recommend_problems_for_topic(g, None, "test", "array", n=10)
        self.assertEqual(out, [])

    def test_respects_n_limit(self):
        problems = [_Pt(f"p{i}", ["array"], 0.3 + i * 0.05) for i in range(10)]
        g = _graph(mastery=0.4)
        out = recommend_problems_for_topic(g, FakeQdrant(problems), "test", "array", n=3)
        self.assertEqual(len(out), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)

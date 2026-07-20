"""
tests/test_topic_recommend.py

Tests pipeline/recommender/services/topic_recommend.py's recommend_topic:
picks a single next topic for a user, in priority order spaced_review ->
in_progress -> unlocked -> novelty -> cold_start.

Run:
    python -m pytest tests/test_topic_recommend.py -v
"""

from __future__ import annotations

import unittest

from pipeline.recommender.models.user_graph import (
    UserGraph, UserNode, ConceptEdge, ConceptConceptEdge, EdgeType,
)
from pipeline.recommender.services.topic_recommend import recommend_topic
from pipeline.recommender.pools.pools import STARTER_CONCEPTS


def _graph(concepts=None, cc=None):
    g = UserGraph(user=UserNode(user_id="u1"))
    for c in (concepts or []):
        g.add_concept_edge(c)
    for e in (cc or []):
        g.add_cc_edge(e)
    return g


def _concept(slug, mastery=0.5, urgency=0.0, edge_type=EdgeType.LEARNING):
    return ConceptEdge(concept_slug=slug, edge_type=edge_type,
                       mastery_score=mastery, urgency=urgency)


class TestRecommendTopic(unittest.TestCase):

    def test_cold_start_falls_back_to_starter_concept(self):
        g = _graph()
        topic_id, reason = recommend_topic(g)
        self.assertEqual(topic_id, STARTER_CONCEPTS[0])
        self.assertEqual(reason, "cold_start")

    def test_urgent_concept_takes_top_priority(self):
        g = _graph([
            _concept("array", mastery=0.3, urgency=0.8),
            _concept("string", mastery=0.9, edge_type=EdgeType.MASTERED),
        ])
        topic_id, reason = recommend_topic(g)
        self.assertEqual(topic_id, "array")
        self.assertEqual(reason, "spaced_review")

    def test_most_urgent_wins_among_multiple_urgent_concepts(self):
        g = _graph([
            _concept("array", mastery=0.3, urgency=0.65),
            _concept("string", mastery=0.4, urgency=0.9),
        ])
        topic_id, reason = recommend_topic(g)
        self.assertEqual(topic_id, "string")
        self.assertEqual(reason, "spaced_review")

    def test_in_progress_closest_to_mastery_wins(self):
        g = _graph([
            _concept("array", mastery=0.65),
            _concept("string", mastery=0.3),
        ])
        topic_id, reason = recommend_topic(g)
        self.assertEqual(topic_id, "array")
        self.assertEqual(reason, "in_progress")

    def test_mastered_concepts_are_not_in_progress_candidates(self):
        g = _graph([
            _concept("array", mastery=0.9, edge_type=EdgeType.MASTERED),
        ])
        topic_id, reason = recommend_topic(g)
        # array is mastered, nothing in progress -> falls through to
        # unlocked/novelty/cold_start (no cc_edges here -> cold_start)
        self.assertEqual(reason, "cold_start")

    def test_unlocked_topic_from_mastered_prereq(self):
        g = _graph(
            [_concept("array", mastery=0.8, edge_type=EdgeType.MASTERED)],
            cc=[ConceptConceptEdge("array", "two_pointers", EdgeType.PREREQ, 1.0)],
        )
        topic_id, reason = recommend_topic(g)
        self.assertEqual(topic_id, "two_pointers")
        self.assertEqual(reason, "unlocked")

    def test_novelty_topic_from_mastered_cooccurrence(self):
        g = _graph(
            [_concept("array", mastery=0.8, edge_type=EdgeType.MASTERED)],
            cc=[ConceptConceptEdge("array", "hash_map", EdgeType.COOCCURS, 0.6)],
        )
        topic_id, reason = recommend_topic(g)
        self.assertEqual(topic_id, "hash_map")
        self.assertEqual(reason, "novelty")

    def test_unlocked_takes_priority_over_novelty(self):
        g = _graph(
            [_concept("array", mastery=0.8, edge_type=EdgeType.MASTERED)],
            cc=[
                ConceptConceptEdge("array", "two_pointers", EdgeType.PREREQ, 1.0),
                ConceptConceptEdge("array", "hash_map", EdgeType.COOCCURS, 0.6),
            ],
        )
        topic_id, reason = recommend_topic(g)
        self.assertEqual(topic_id, "two_pointers")
        self.assertEqual(reason, "unlocked")


if __name__ == "__main__":
    unittest.main(verbosity=2)

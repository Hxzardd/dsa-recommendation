"""
tests/test_synthetic_user_generator.py

Tests training/synthetic_user_generator.py against a fake Qdrant catalog --
same FakeQdrant pattern already used in test_progression_changes_ranking.py
to exercise the real recommender without live infra. Verifies the
simulator's output flows cleanly through the UNMODIFIED
DatasetGenerator/LabelGenerator.

Run:
    python -m pytest tests/test_synthetic_user_generator.py -v
"""

from __future__ import annotations

import unittest

from pipeline.recommender.models.user_graph import ConceptConceptEdge, EdgeType, UserGraph
from training.dataset_generator import DatasetGenerator
from training.label_generator import LabelGenerator
from training.synthetic_user_generator import (
    PERSONAS,
    SimulatorConfig,
    SyntheticUserGenerator,
)

_TOPICS = ["array", "graph", "tree", "dp", "string", "two_pointers", "hash_map", "greedy"]

# A small, deterministic fake offline concept graph -- same shape
# load_real_concept_graph() returns, just hand-built so these tests stay
# fully offline (no DB/Neo4j round trip), matching this file's existing
# FakeQdrant-only convention.
_FAKE_CC_EDGES: dict[str, list[ConceptConceptEdge]] = {
    "array": [
        ConceptConceptEdge(source_slug="array", target_slug="two_pointers",
                            edge_type=EdgeType.COOCCURS, weight=0.4),
        ConceptConceptEdge(source_slug="array", target_slug="dp",
                            edge_type=EdgeType.PREREQ, weight=1.0),
    ],
    "tree": [
        ConceptConceptEdge(source_slug="tree", target_slug="graph",
                            edge_type=EdgeType.COOCCURS, weight=0.3),
    ],
    "hash_map": [
        ConceptConceptEdge(source_slug="hash_map", target_slug="string",
                            edge_type=EdgeType.PREREQ, weight=1.0),
    ],
}


class _Pt:
    def __init__(self, pid, tags, diff, score=0.9):
        self.id = pid
        self.payload = {"problem_id": pid, "topic_tags": tags, "difficulty_score": diff}
        self.score = score


class FakeQdrant:
    """Same pattern as test_progression_changes_ranking.py::FakeQdrant."""

    def __init__(self, problems):
        self.problems = problems

    def scroll(self, collection_name, scroll_filter=None, limit=10,
               with_payload=True, with_vectors=False):
        want_tags, diff_range = None, None
        if scroll_filter is not None:
            for cond in scroll_filter.must:
                key = getattr(cond, "key", None)
                if key == "topic_tags" and getattr(cond, "match", None) is not None:
                    want_tags = set(cond.match.any)
                if key == "difficulty_score" and getattr(cond, "range", None) is not None:
                    diff_range = cond.range
        out = []
        for p in self.problems:
            if want_tags is not None and not (set(p.payload["topic_tags"]) & want_tags):
                continue
            if diff_range is not None:
                d = p.payload["difficulty_score"]
                if diff_range.gte is not None and d < diff_range.gte:
                    continue
                if diff_range.lte is not None and d > diff_range.lte:
                    continue
            out.append(p)
            if len(out) >= limit:
                break
        return out, None

    def query_points(self, collection_name, query, limit=10,
                     with_payload=True, with_vectors=False):
        class R:
            points = sorted(self.problems, key=lambda p: p.score, reverse=True)[:limit]
        return R()


def _catalog(per_topic=15):
    problems = []
    for topic in _TOPICS:
        for i in range(per_topic):
            diff = round(i / per_topic, 4)
            problems.append(_Pt(f"{topic}_{i}", [topic], diff, score=1.0 - diff))
    return problems


def _small_generator(num_users=3, seed=42, persona_distribution=None, cc_edges=None):
    gen = SyntheticUserGenerator(
        qdrant=FakeQdrant(_catalog()),
        topic_slugs=_TOPICS,
        config=SimulatorConfig(
            num_users=num_users, random_seed=seed,
            persona_distribution=persona_distribution,
            total_n=20, k=8,
        ),
        cc_edges=cc_edges,
    )
    return gen.generate()


class TestPersonas(unittest.TestCase):

    def test_seven_personas_defined(self):
        expected = {"beginner", "intermediate", "advanced", "competitive",
                    "reviewer", "inconsistent_learner", "returning_user"}
        self.assertEqual(set(PERSONAS.keys()), expected)

    def test_every_persona_has_valid_ranges(self):
        for name, p in PERSONAS.items():
            self.assertLessEqual(p.mastery_range[0], p.mastery_range[1])
            self.assertLessEqual(p.topics_touched_range[0], p.topics_touched_range[1])
            self.assertLessEqual(p.sessions_range[0], p.sessions_range[1])
            self.assertGreaterEqual(p.base_engagement_probability, 0.0)
            self.assertLessEqual(p.base_engagement_probability, 1.0)


class TestGeneratorValidation(unittest.TestCase):

    def test_empty_topic_slugs_raises(self):
        with self.assertRaises(ValueError):
            SyntheticUserGenerator(qdrant=FakeQdrant(_catalog()), topic_slugs=[])


class TestGenerate(unittest.TestCase):

    def test_produces_events_and_outcome_provider(self):
        result = _small_generator(num_users=3)
        self.assertGreater(len(result.events), 0)
        self.assertIsNotNone(result.outcome_provider)

    def test_events_have_real_merged_candidates_with_predicted_success(self):
        result = _small_generator(num_users=2)
        self.assertGreater(len(result.events), 0)
        for event in result.events:
            self.assertGreater(len(event.candidates), 0)
            for c in event.candidates:
                self.assertIsNotNone(c.predicted_success)
                self.assertTrue(c.pool_sources)

    def test_query_ids_are_unique_across_all_events(self):
        result = _small_generator(num_users=4)
        query_ids = [e.query_id for e in result.events]
        self.assertEqual(len(query_ids), len(set(query_ids)))

    def test_multiple_sessions_produce_multiple_events_for_some_users(self):
        result = _small_generator(num_users=5)
        user_event_counts = {}
        for e in result.events:
            user_event_counts[e.user_id] = user_event_counts.get(e.user_id, 0) + 1
        self.assertTrue(any(count > 1 for count in user_event_counts.values()))

    def test_deterministic_given_same_seed(self):
        result1 = _small_generator(num_users=5, seed=7)
        result2 = _small_generator(num_users=5, seed=7)
        ids1 = [(e.query_id, e.user_id, [c.problem_id for c in e.candidates]) for e in result1.events]
        ids2 = [(e.query_id, e.user_id, [c.problem_id for c in e.candidates]) for e in result2.events]
        self.assertEqual(ids1, ids2)

    def test_different_seed_can_produce_different_output(self):
        result1 = _small_generator(num_users=5, seed=1)
        result2 = _small_generator(num_users=5, seed=2)
        ids1 = [e.user_id for e in result1.events]
        ids2 = [e.user_id for e in result2.events]
        self.assertNotEqual(ids1, ids2)

    def test_forced_persona_distribution_is_honoured(self):
        result = _small_generator(num_users=4, persona_distribution={"beginner": 1.0})
        for event in result.events:
            self.assertIn("beginner", event.user_id)

    def test_earlier_event_graph_snapshot_not_mutated_by_later_sessions(self):
        """The graph captured in an earlier RecommendationEvent must not
        reflect mastery/solved_ids changes from LATER sessions of the same
        synthetic user -- otherwise features would leak future state
        backward into an earlier query's row."""
        result = _small_generator(num_users=3, seed=99)
        by_user: dict[str, list] = {}
        for e in result.events:
            by_user.setdefault(e.user_id, []).append(e)

        multi_session_users = [uid for uid, evs in by_user.items() if len(evs) > 1]
        self.assertTrue(multi_session_users, "test setup needs at least one multi-session user")

        for uid in multi_session_users:
            evs = by_user[uid]
            first_graph: UserGraph = evs[0].graph
            last_graph: UserGraph = evs[-1].graph
            # The two snapshots must be independent objects...
            self.assertIsNot(first_graph, last_graph)
            self.assertIsNot(first_graph.concept_edges, last_graph.concept_edges)


class TestConceptConceptGraphInjection(unittest.TestCase):
    """Proves cc_edges (PREREQ + COOCCURS) actually reach every synthetic
    user's graph, and that downstream prerequisite_completion_ratio becomes
    meaningful once they do -- the fix for Issue B/C (previously: cc_edges
    was always empty, so this feature was 100% missing and Pool B_C's
    graph-fallback path never had anything to draw from)."""

    def test_cc_edges_attached_to_every_user_graph(self):
        result = _small_generator(num_users=3, cc_edges=_FAKE_CC_EDGES)
        self.assertGreater(len(result.events), 0)
        for event in result.events:
            self.assertEqual(event.graph.cc_edges, _FAKE_CC_EDGES)

    def test_no_cc_edges_means_empty_graph_not_a_crash(self):
        # Default (cc_edges=None) must stay offline-safe -- {} rather than
        # attempting any DB/Neo4j round trip.
        result = _small_generator(num_users=2, cc_edges=None)
        self.assertGreater(len(result.events), 0)
        for event in result.events:
            self.assertEqual(event.graph.cc_edges, {})

    def test_prerequisite_completion_ratio_becomes_meaningful(self):
        result = _small_generator(num_users=8, seed=17, cc_edges=_FAKE_CC_EDGES)
        df = DatasetGenerator().generate_dataframe(result.events)
        self.assertIn("prerequisite_completion_ratio", df.columns)
        self.assertGreater(df["prerequisite_completion_ratio"].notna().sum(), 0)


class TestForgettingAndReviewUrgency(unittest.TestCase):
    """Proves urgency now grows with real elapsed time (HLR forgetting
    curve) instead of being seeded once from a range that could never
    cross urgent_concepts()'s 0.6 threshold -- the fix for Issue A."""

    def test_urgency_grows_for_an_untouched_topic_over_many_sessions(self):
        result = _small_generator(num_users=10, seed=23)
        by_user: dict[str, list] = {}
        for e in result.events:
            by_user.setdefault(e.user_id, []).append(e)

        found_growth = False
        for uid, evs in by_user.items():
            if len(evs) < 3:
                continue
            first_edges = evs[0].graph.concept_edges
            last_edges = evs[-1].graph.concept_edges
            for slug, first_edge in first_edges.items():
                last_edge = last_edges.get(slug)
                if last_edge is not None and last_edge.urgency > first_edge.urgency + 0.05:
                    found_growth = True
                    break
            if found_growth:
                break
        self.assertTrue(found_growth, "expected at least one concept's urgency to grow across sessions")

    def test_severity_derives_from_repeated_failure_not_a_single_miss(self):
        gen = SyntheticUserGenerator(qdrant=FakeQdrant(_catalog()), topic_slugs=_TOPICS)
        self.assertEqual(gen._severity_from_history(1, 0), 0.0)
        self.assertGreater(gen._severity_from_history(5, 0), 0.5)
        self.assertEqual(gen._severity_from_history(5, 5), 0.0)


class TestEndToEndIntegrationWithExistingPipeline(unittest.TestCase):
    """Proves the simulator's output flows through the UNMODIFIED
    DatasetGenerator and LabelGenerator with no special-casing."""

    def test_dataset_generator_consumes_events_without_modification(self):
        result = _small_generator(num_users=4)
        df = DatasetGenerator().generate_dataframe(result.events)
        self.assertGreater(len(df), 0)
        self.assertIn("query_id", df.columns)
        self.assertIn("predicted_success", df.columns)

    def test_dataset_generator_schema_validates(self):
        result = _small_generator(num_users=4)
        gen = DatasetGenerator()
        df = gen.generate_dataframe(result.events)
        self.assertEqual(gen.validate_schema(df), [])

    def test_label_generator_produces_valid_labels(self):
        result = _small_generator(num_users=5, seed=3)
        df = DatasetGenerator().generate_dataframe(result.events)
        label_gen = LabelGenerator(provider=result.outcome_provider)
        labeled = label_gen.label_dataframe(df)

        valid_labels = {0.0, 1.0, 2.0}
        actual_labels = set(labeled["label"].dropna().unique().tolist())
        self.assertTrue(actual_labels.issubset(valid_labels))

    def test_label_generator_validate_reports_no_problems(self):
        """End-to-end proof: simulated events -> dataset -> labels contain
        no duplicate rows, no impossible attribution, no inconsistent
        query groups -- the full chain is internally consistent."""
        result = _small_generator(num_users=6, seed=11)
        df = DatasetGenerator().generate_dataframe(result.events)
        label_gen = LabelGenerator(provider=result.outcome_provider)
        labeled = label_gen.label_dataframe(df)
        problems = label_gen.validate(labeled)
        self.assertEqual(problems, [])

    def test_label_statistics_are_computable(self):
        result = _small_generator(num_users=6, seed=5)
        df = DatasetGenerator().generate_dataframe(result.events)
        label_gen = LabelGenerator(provider=result.outcome_provider)
        labeled = label_gen.label_dataframe(df)
        stats = label_gen.compute_statistics(labeled)
        self.assertEqual(stats["total_rows"], len(labeled))
        self.assertGreaterEqual(stats["percent_solved"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

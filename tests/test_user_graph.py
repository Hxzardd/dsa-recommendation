"""
tests/test_user_graph.py
=========================
Tests for UserGraph construction and UserStateVector projection.

Runs entirely without a real database or Qdrant instance.
All Postgres tables and Qdrant calls are mocked inline.

What is tested:
    1.  UserGraphService._build() assembles the correct graph from
        Submission, RecommendationLog, UserTopicMastery, ConceptGapProfile
    2.  Edge types are assigned correctly
        (SOLVED/ATTEMPTED/EXPOSED/SKIPPED/MASTERED/LEARNING/WEAK)
    3.  BKT store overrides UserTopicMastery.mastery_score
    4.  HLR store populates urgency + half_life on ConceptEdge
    5.  Confidence is mapped low/medium/high -> 0.33/0.66/1.0
    6.  ConceptGapProfile merges severity onto existing ConceptEdge
    7.  Skip count >= 3 produces SKIPPED edge, < 3 produces EXPOSED
    8.  Solved problem IDs land in graph.solved_ids
    9.  Offline CC edges are attached only for concepts the user touched
    10. UserStateBuilder._effective_weight() compounds all five signals
    11. effective_weight = 0 when mastery = 0
    12. effective_weight decreases as urgency increases
    13. effective_weight decreases as recency increases (old practice)
    14. ELO < 1200 amplifies easy concepts; ELO > 1600 amplifies hard
    15. UserStateVector.vector is 1920-d and L2-normalised
    16. Cold start (0 solved) falls back to concept-centroid path
    17. Warm start blends concept vector + problem history
    18. graph.to_dict() / from_dict() round-trip is lossless
    19. UserGraph.mastered_concepts() / weak_concepts() / urgent_concepts()
    20. Cache invalidation: invalidate() removes Redis key

Run:
    python -m pytest tests/test_user_graph.py -v
    python -m pytest tests/test_user_graph.py -v -k "test_edge_types"
"""

from __future__ import annotations

import json
import sys
import time
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# ---------------------------------------------------------------------------
# Minimal stubs so we can import without qdrant_client / fastapi installed
# ---------------------------------------------------------------------------

def _stub_module(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod

for _m in ["qdrant_client", "qdrant_client.models",
           "fastapi", "sentence_transformers"]:
    if _m not in sys.modules:
        _stub_module(_m)

# stub hlr so user_graph_service can import calculate_urgency
_hlr_stub = _stub_module("pipeline.recommender.hlr")
def _fake_urgency(state, ts):
    hl = state.get("half_life", 1.0)
    lr = state.get("last_review")
    if lr is None:
        return 0.5
    import datetime
    try:
        dt = datetime.datetime.fromisoformat(lr)
        if dt.tzinfo is None:
            import datetime as _dt2
            dt = dt.replace(tzinfo=_dt2.timezone.utc)
        days = (datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc) - dt
                ).total_seconds() / 86400
        p = 2 ** (-days / max(hl, 0.01))
        return round(1.0 - p, 4)
    except Exception:
        return 0.5
_hlr_stub.calculate_urgency = _fake_urgency

# stub qdrant_client.models classes used at import time
_qm = sys.modules["qdrant_client.models"]
for _cls in ["Filter", "FieldCondition", "MatchAny", "MatchValue",
             "Range", "Distance", "VectorParams", "PointStruct",
             "OptimizersConfigDiff"]:
    setattr(_qm, _cls, MagicMock)

# ---------------------------------------------------------------------------
# Now import our code
# ---------------------------------------------------------------------------

from pipeline.recommender.models.user_graph import (
    UserGraph, UserNode, ProblemEdge, ConceptEdge,
    ConceptConceptEdge, EdgeType,
)
from pipeline.recommender.models.user_state import (
    UserStateBuilder, QS_DIM, RGCN_DIM, FULL_DIM,
)
from pipeline.recommender.services.user_graph_service import (
    UserGraphService, load_offline_concept_graph,
)


# ===========================================================================
# Helpers
# ===========================================================================

USER_ID = "test_user_001"

def _ts(days_ago: float = 0.0) -> float:
    return time.time() - days_ago * 86400

def _dt(days_ago: float = 0.0):
    """Return a datetime-like object with .timestamp() method."""
    ts = _ts(days_ago)
    m = MagicMock()
    m.timestamp.return_value = ts
    m.tzinfo = None
    return m

def _nrd():
    """next_review_date mock."""
    return "2026-07-01"


def _make_db(
    submissions=None,
    rec_logs=None,
    mastery_rows=None,
    gap_rows=None,
    user_row=None,
):
    """
    Returns a mock DB session whose .execute().fetchall() / .fetchone()
    return the supplied data in the order the service queries them.

    Query order in UserGraphService._build():
        1. fetch_user       → fetchone()
        2. load_submissions → fetchall()
        3. load_rec_log     → fetchall()
        4. load_mastery     → fetchall()
        5. load_gaps        → fetchall()
    """
    db = MagicMock()

    _user = user_row or (
        USER_ID, "testuser", True,   # user_id, username, onboarding_complete
        500, 3,                       # total_xp, current_level
    )

    _subs    = submissions  or []
    _rec     = rec_logs     or []
    _mastery = mastery_rows or []
    _gaps    = gap_rows     or []

    call_count = {"n": 0}

    def _execute(sql, params=None):
        n = call_count["n"]
        call_count["n"] += 1
        result = MagicMock()
        if n == 0:   result.fetchone.return_value  = _user
        elif n == 1: result.fetchall.return_value  = _subs
        elif n == 2: result.fetchall.return_value  = _rec
        elif n == 3: result.fetchall.return_value  = _mastery
        elif n == 4: result.fetchall.return_value  = _gaps
        return result

    db.execute.side_effect = _execute
    return db


def _mastery_row(
    topic_id,
    mastery_score=0.5,   # 0-1 in the real schema, not 0-100 -- see user_graph_service.py fix
    confidence="medium",
    attempt_count=5,
    problems_solved=3,
    days_ago=2.0,
    sm2_ef=2.5,
    sm2_interval=3,
    next_review_date=None,
):
    # Must match exactly the 9 columns in _load_topic_mastery SELECT:
    # topic_id, mastery_score, confidence, attempt_count, problems_solved,
    # last_attempted, sm2_ef, sm2_interval, next_review_date
    return (
        topic_id, mastery_score, confidence,
        attempt_count, problems_solved,
        _dt(days_ago),
        sm2_ef, sm2_interval,
        next_review_date or _nrd(),
    )


def _submission_row(
    problem_id,
    verdict="Accepted",
    score=80.0,
    hints=0,
    attempts=1,
    days_ago=1.0,
):
    return (
        problem_id, verdict, score,
        hints, attempts,
        _dt(days_ago),
    )


def _rec_row(problem_id, days_ago=1.0, skipped=False, skip_count=0):
    return (problem_id, _dt(days_ago), skipped, skip_count)


def _gap_row(gap_name, severity=0.7):
    return (gap_name, severity)


# ===========================================================================
# 1. Graph assembly from DB
# ===========================================================================

class TestGraphAssembly(unittest.TestCase):

    def setUp(self):
        # _load_topic_mastery translates user_topic_mastery.topic_id (a real
        # opaque FK in production) to an ML slug via a live topic-table
        # lookup (database.postgres.db._topic_id_to_ml_slug) -- this file's
        # module docstring guarantees "runs entirely without a real
        # database", so that lookup is patched to identity here: these
        # fixtures' topic_id values (e.g. "arrays") pass through unchanged,
        # keeping every other assertion (confidence mapping, urgency,
        # mastered/learning edges, etc.) exercising real logic without a
        # live Postgres dependency.
        patcher_get_conn = patch(
            "pipeline.recommender.services.user_graph_service.get_connection",
            return_value=None)
        patcher_release_conn = patch(
            "pipeline.recommender.services.user_graph_service.release_connection")
        patcher_translate = patch(
            "pipeline.recommender.services.user_graph_service._topic_id_to_ml_slug",
            side_effect=lambda conn, topic_id: topic_id)
        patcher_get_conn.start()
        patcher_release_conn.start()
        patcher_translate.start()
        self.addCleanup(patcher_get_conn.stop)
        self.addCleanup(patcher_release_conn.stop)
        self.addCleanup(patcher_translate.stop)

    def _build(self, **kwargs):
        db = _make_db(**kwargs)
        svc = UserGraphService(db=db, redis=None, bkt={}, hlr={})
        return svc.get(USER_ID)

    # ------------------------------------------------------------------
    # User node
    # ------------------------------------------------------------------

    def test_user_node_populated(self):
        g = self._build()
        self.assertEqual(g.user.user_id,  USER_ID)
        self.assertEqual(g.user.username, "testuser")
        self.assertTrue(g.user.onboarding_complete)
        self.assertEqual(g.user.total_xp,      500)
        self.assertEqual(g.user.current_level, 3)
        self.assertEqual(g.user.elo_rating,    1200)   # default

    def test_user_node_is_only_node_of_type(self):
        # UserGraph has no second user node — single user context
        g = self._build()
        self.assertIsInstance(g.user, UserNode)
        self.assertFalse(hasattr(g, "users"))

    # ------------------------------------------------------------------
    # Problem edges
    # ------------------------------------------------------------------

    def test_solved_edge_created(self):
        g = self._build(submissions=[
            _submission_row("prob_001", verdict="Accepted", score=90.0),
        ])
        self.assertIn("prob_001", g.problem_edges)
        self.assertEqual(g.problem_edges["prob_001"].edge_type, EdgeType.SOLVED)

    def test_attempted_edge_for_wrong_answer(self):
        g = self._build(submissions=[
            _submission_row("prob_002", verdict="WA", score=20.0),
        ])
        self.assertEqual(g.problem_edges["prob_002"].edge_type, EdgeType.ATTEMPTED)

    def test_attempted_edge_for_tle(self):
        g = self._build(submissions=[
            _submission_row("prob_003", verdict="TLE", score=10.0),
        ])
        self.assertEqual(g.problem_edges["prob_003"].edge_type, EdgeType.ATTEMPTED)

    def test_solved_id_in_set(self):
        g = self._build(submissions=[
            _submission_row("prob_001", verdict="Accepted"),
        ])
        self.assertIn("prob_001", g.solved_ids)

    def test_failed_not_in_solved_ids(self):
        g = self._build(submissions=[
            _submission_row("prob_002", verdict="WA"),
        ])
        self.assertNotIn("prob_002", g.solved_ids)

    def test_normalised_score_stored_as_fraction(self):
        # DB stores 0-100, service divides by 100
        g = self._build(submissions=[
            _submission_row("prob_001", verdict="Accepted", score=80.0),
        ])
        self.assertAlmostEqual(g.problem_edges["prob_001"].normalised_score, 0.80)

    def test_recency_weight_recent_higher(self):
        g = self._build(submissions=[
            _submission_row("prob_A", verdict="Accepted", days_ago=1.0),
            _submission_row("prob_B", verdict="Accepted", days_ago=30.0),
        ])
        w_recent = g.problem_edges["prob_A"].recency_weight
        w_old    = g.problem_edges["prob_B"].recency_weight
        self.assertGreater(w_recent, w_old)

    # ------------------------------------------------------------------
    # Recommendation log edges
    # ------------------------------------------------------------------

    def test_exposed_edge_low_skip_count(self):
        g = self._build(rec_logs=[
            _rec_row("prob_exp", skipped=True, skip_count=2),
        ])
        self.assertIn("prob_exp", g.problem_edges)
        self.assertEqual(g.problem_edges["prob_exp"].edge_type, EdgeType.EXPOSED)

    def test_skipped_edge_at_threshold(self):
        g = self._build(rec_logs=[
            _rec_row("prob_skip", skipped=True, skip_count=3),
        ])
        self.assertEqual(g.problem_edges["prob_skip"].edge_type, EdgeType.SKIPPED)
        self.assertIn("prob_skip", g.deprioritised_ids)

    def test_solved_not_overwritten_by_exposed(self):
        # If problem is solved, EXPOSED edge should not overwrite it
        g = self._build(
            submissions=[_submission_row("prob_both", verdict="Accepted")],
            rec_logs=[_rec_row("prob_both", skip_count=0)],
        )
        self.assertEqual(g.problem_edges["prob_both"].edge_type, EdgeType.SOLVED)

    # ------------------------------------------------------------------
    # Concept edges
    # ------------------------------------------------------------------

    def test_mastered_edge_high_mastery(self):
        g = self._build(mastery_rows=[
            _mastery_row("arrays", mastery_score=0.8),
        ])
        self.assertIn("arrays", g.concept_edges)
        self.assertEqual(g.concept_edges["arrays"].edge_type, EdgeType.MASTERED)

    def test_learning_edge_mid_mastery(self):
        g = self._build(mastery_rows=[
            _mastery_row("graphs", mastery_score=0.5),
        ])
        self.assertEqual(g.concept_edges["graphs"].edge_type, EdgeType.LEARNING)

    def test_zero_mastery_excluded(self):
        g = self._build(mastery_rows=[
            _mastery_row("dp", mastery_score=0.0),
        ])
        self.assertNotIn("dp", g.concept_edges)

    def test_confidence_mapped_low(self):
        g = self._build(mastery_rows=[
            _mastery_row("arrays", mastery_score=0.75, confidence="low"),
        ])
        self.assertAlmostEqual(g.concept_edges["arrays"].confidence, 0.33)

    def test_confidence_mapped_medium(self):
        g = self._build(mastery_rows=[
            _mastery_row("arrays", mastery_score=0.75, confidence="medium"),
        ])
        self.assertAlmostEqual(g.concept_edges["arrays"].confidence, 0.66)

    def test_confidence_mapped_high(self):
        g = self._build(mastery_rows=[
            _mastery_row("arrays", mastery_score=0.75, confidence="high"),
        ])
        self.assertAlmostEqual(g.concept_edges["arrays"].confidence, 1.0)

    def test_bkt_store_overrides_db_mastery(self):
        bkt = {USER_ID: {"arrays": 0.9}}
        db  = _make_db(mastery_rows=[
            _mastery_row("arrays", mastery_score=0.5),
        ])
        svc = UserGraphService(db=db, redis=None, bkt=bkt, hlr={})
        g   = svc.get(USER_ID)
        self.assertAlmostEqual(g.concept_edges["arrays"].mastery_score, 0.9)

    def test_hlr_store_populates_urgency_and_half_life(self):
        hlr_state = {"half_life": 7.0, "last_review": "2025-01-01T00:00:00+00:00"}
        hlr = {USER_ID: {"arrays": hlr_state}}
        db  = _make_db(mastery_rows=[
            _mastery_row("arrays", mastery_score=0.75),
        ])
        svc = UserGraphService(db=db, redis=None, bkt={}, hlr=hlr)
        g   = svc.get(USER_ID)
        edge = g.concept_edges["arrays"]
        self.assertEqual(edge.half_life, 7.0)
        self.assertGreater(edge.urgency, 0.0)

    # ------------------------------------------------------------------
    # Concept gap profile
    # ------------------------------------------------------------------

    def test_gap_severity_merged_onto_existing_edge(self):
        g = self._build(
            mastery_rows=[_mastery_row("arrays", mastery_score=0.75)],
            gap_rows=[_gap_row("arrays", severity=0.8)],
        )
        self.assertAlmostEqual(g.concept_edges["arrays"].severity, 0.8)

    def test_gap_creates_new_weak_edge(self):
        # gap_name not in mastery_rows → new WEAK edge created
        g = self._build(
            mastery_rows=[],
            gap_rows=[_gap_row("off_by_one", severity=0.75)],
        )
        self.assertIn("off_by_one", g.concept_edges)
        self.assertEqual(g.concept_edges["off_by_one"].edge_type, EdgeType.WEAK)

    def test_weak_edge_on_high_severity(self):
        g = self._build(
            mastery_rows=[_mastery_row("arrays", mastery_score=0.75)],
            gap_rows=[_gap_row("arrays", severity=0.7)],
        )
        self.assertEqual(g.concept_edges["arrays"].edge_type, EdgeType.WEAK)

    # ------------------------------------------------------------------
    # Concept-concept edges
    # ------------------------------------------------------------------

    def test_cc_edges_loaded_for_user_concepts(self):
        # Manually inject a CC edge into _PREREQ_CACHE
        import pipeline.recommender.services.user_graph_service as svc_mod
        svc_mod._PREREQ_CACHE["arrays"] = [
            ConceptConceptEdge("arrays", "sorting", EdgeType.COOCCURS, 0.5)
        ]
        g = self._build(mastery_rows=[
            _mastery_row("arrays", mastery_score=0.75),
        ])
        self.assertIn("arrays", g.cc_edges)
        self.assertEqual(g.cc_edges["arrays"][0].target_slug, "sorting")
        # cleanup
        svc_mod._PREREQ_CACHE.clear()

    def test_cc_edges_loaded_for_untouched_concepts_too(self):
        """
        Regression test for a real bug (flagged by Greptile): _load_cc_edges
        used to only attach edges whose SOURCE was already a concept the
        user had touched. That meant a prerequisite like arrays->dp was
        invisible to is_locked() whenever the user hadn't touched "arrays"
        yet -- exactly the case prerequisite gating exists to catch, since
        "locked" by definition means a concept the user hasn't engaged with.
        Fixed: the full offline prerequisite graph is loaded every time,
        regardless of what the user has touched.
        """
        import pipeline.recommender.services.user_graph_service as svc_mod
        svc_mod._PREREQ_CACHE["graphs"] = [
            ConceptConceptEdge("graphs", "trees", EdgeType.PREREQ, 1.0)
        ]
        # user has no mastery on "graphs" at all
        g = self._build(mastery_rows=[
            _mastery_row("arrays", mastery_score=0.75),
        ])
        self.assertIn("graphs", g.cc_edges)
        # and the lock check must actually see it
        self.assertTrue(g.is_locked(["trees"]))
        svc_mod._PREREQ_CACHE.clear()

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def test_mastered_concepts(self):
        g = self._build(mastery_rows=[
            _mastery_row("arrays",  mastery_score=0.8),
            _mastery_row("sorting", mastery_score=0.5),
        ])
        mastered = g.mastered_concepts()
        self.assertIn("arrays",  mastered)
        self.assertNotIn("sorting", mastered)

    def test_weak_concepts(self):
        g = self._build(
            mastery_rows=[_mastery_row("arrays", mastery_score=0.75)],
            gap_rows=[_gap_row("arrays", severity=0.7)],
        )
        self.assertIn("arrays", g.weak_concepts())

    def test_urgent_concepts(self):
        hlr = {USER_ID: {"arrays": {
            "half_life": 1.0,
            "last_review": "2020-01-01T00:00:00+00:00",  # very old
        }}}
        db = _make_db(mastery_rows=[_mastery_row("arrays", mastery_score=0.75)])
        svc = UserGraphService(db=db, redis=None, bkt={}, hlr=hlr)
        g = svc.get(USER_ID)
        self.assertIn("arrays", g.urgent_concepts())


# ===========================================================================
# 2. Serialisation round-trip
# ===========================================================================

class TestSerialization(unittest.TestCase):

    def _sample_graph(self):
        g = UserGraph(user=UserNode(user_id=USER_ID, username="tester"))
        g.add_problem_edge(ProblemEdge(
            problem_id="p1", edge_type=EdgeType.SOLVED,
            normalised_score=0.8, timestamp=_ts(1),
        ))
        g.add_concept_edge(ConceptEdge(
            concept_slug="arrays", edge_type=EdgeType.MASTERED,
            mastery_score=0.8, confidence=0.66, half_life=7.0,
        ))
        g.add_cc_edge(ConceptConceptEdge(
            source_slug="arrays", target_slug="dp",
            edge_type=EdgeType.PREREQ, weight=1.0,
        ))
        return g

    def test_roundtrip_preserves_user(self):
        g = self._sample_graph()
        g2 = UserGraph.from_dict(g.to_dict())
        self.assertEqual(g2.user.user_id,  g.user.user_id)
        self.assertEqual(g2.user.username, g.user.username)

    def test_roundtrip_preserves_problem_edges(self):
        g = self._sample_graph()
        g2 = UserGraph.from_dict(g.to_dict())
        self.assertIn("p1", g2.problem_edges)
        self.assertEqual(g2.problem_edges["p1"].edge_type, EdgeType.SOLVED)
        self.assertAlmostEqual(g2.problem_edges["p1"].normalised_score, 0.8)

    def test_roundtrip_preserves_concept_edges(self):
        g = self._sample_graph()
        g2 = UserGraph.from_dict(g.to_dict())
        self.assertIn("arrays", g2.concept_edges)
        self.assertAlmostEqual(g2.concept_edges["arrays"].mastery_score, 0.8)
        self.assertAlmostEqual(g2.concept_edges["arrays"].confidence,    0.66)
        self.assertAlmostEqual(g2.concept_edges["arrays"].half_life,     7.0)

    def test_roundtrip_preserves_cc_edges(self):
        """
        Regression test for a real Greptile-flagged bug: to_dict() never
        included cc_edges at all, so any graph restored from the Redis
        cache silently lost every prerequisite/cooccurrence edge -- after
        the FIRST cache write, is_locked() had nothing to check against
        and every candidate looked unlocked. Fixed: cc_edges is now
        serialized and restored like every other field.
        """
        g = self._sample_graph()
        g2 = UserGraph.from_dict(g.to_dict())
        self.assertIn("arrays", g2.cc_edges)
        self.assertEqual(g2.cc_edges["arrays"][0].target_slug, "dp")
        self.assertEqual(g2.cc_edges["arrays"][0].edge_type, EdgeType.PREREQ)

    def test_roundtrip_cc_edges_actually_drive_lock_check(self):
        """
        Same fix, verified through is_locked() specifically (not just that
        the data survived serialization) -- uses a prerequisite that is
        genuinely unmastered, unlike _sample_graph()'s "arrays" (mastered
        at 0.8, used by the concept-edge round-trip test above).
        """
        g = UserGraph(user=UserNode(user_id=USER_ID))
        g.add_concept_edge(ConceptEdge(
            concept_slug="arrays", edge_type=EdgeType.LEARNING,
            mastery_score=0.3,   # below the 0.7 mastered threshold
        ))
        g.add_cc_edge(ConceptConceptEdge(
            source_slug="arrays", target_slug="dp",
            edge_type=EdgeType.PREREQ, weight=1.0,
        ))
        g2 = UserGraph.from_dict(g.to_dict())
        self.assertTrue(g2.is_locked(["dp"]))

    def test_roundtrip_missing_cc_edges_key_defaults_empty(self):
        """Graphs cached BEFORE this fix have no cc_edges key at all -- must
        restore gracefully with empty cc_edges, not raise KeyError."""
        g = self._sample_graph()
        d = g.to_dict()
        del d["cc_edges"]   # simulate a pre-fix cached graph
        g2 = UserGraph.from_dict(d)
        self.assertEqual(g2.cc_edges, {})

    def test_roundtrip_preserves_solved_ids(self):
        g = self._sample_graph()
        g2 = UserGraph.from_dict(g.to_dict())
        self.assertIn("p1", g2.solved_ids)

    def test_json_serialisable(self):
        g = self._sample_graph()
        # should not raise
        raw = json.dumps(g.to_dict())
        self.assertIsInstance(raw, str)


# ===========================================================================
# 3. UserStateBuilder — effective_weight
# ===========================================================================

class TestEffectiveWeight(unittest.TestCase):

    def _builder(self):
        q = MagicMock()
        q.scroll.return_value = ([], None)
        return UserStateBuilder(q)

    def _edge(self, mastery=0.8, confidence=0.66, urgency=0.3,
               half_life=7.0, days_ago=2.0, severity=0.0):
        return ConceptEdge(
            concept_slug="arrays",
            edge_type=EdgeType.MASTERED,
            mastery_score=mastery,
            confidence=confidence,
            urgency=urgency,
            half_life=half_life,
            severity=severity,
            last_attempted=_ts(days_ago),
        )

    def test_zero_mastery_gives_zero_weight(self):
        b = self._builder()
        w = b._effective_weight(self._edge(mastery=0.0), "arrays", 1200)
        self.assertEqual(w, 0.0)

    def test_high_mastery_gives_positive_weight(self):
        b = self._builder()
        w = b._effective_weight(self._edge(mastery=0.8), "arrays", 1200)
        self.assertGreater(w, 0.0)

    def test_high_urgency_reduces_weight(self):
        b = self._builder()
        w_low_urgency  = b._effective_weight(self._edge(urgency=0.1), "arrays", 1200)
        w_high_urgency = b._effective_weight(self._edge(urgency=0.9), "arrays", 1200)
        self.assertGreater(w_low_urgency, w_high_urgency)

    def test_low_confidence_reduces_weight(self):
        b = self._builder()
        w_low  = b._effective_weight(self._edge(confidence=0.33), "arrays", 1200)
        w_high = b._effective_weight(self._edge(confidence=1.0),  "arrays", 1200)
        self.assertGreater(w_high, w_low)

    def test_old_practice_reduces_weight(self):
        b = self._builder()
        w_recent = b._effective_weight(self._edge(days_ago=1.0),  "arrays", 1200)
        w_old    = b._effective_weight(self._edge(days_ago=60.0), "arrays", 1200)
        self.assertGreater(w_recent, w_old)

    def test_all_five_signals_compound(self):
        # perfect signal: mastery=1, confidence=1, urgency=0, recent, correct difficulty
        b = self._builder()
        w_perfect = b._effective_weight(
            self._edge(mastery=1.0, confidence=1.0, urgency=0.0, days_ago=0.1),
            "arrays", 1200
        )
        # degraded: all signals halved
        w_degraded = b._effective_weight(
            self._edge(mastery=0.5, confidence=0.5, urgency=0.5, days_ago=30.0),
            "arrays", 1200
        )
        self.assertGreater(w_perfect, w_degraded * 2)   # should be much larger


# ===========================================================================
# 4. ELO difficulty weighting
# ===========================================================================

class TestDifficultyWeight(unittest.TestCase):

    def _builder(self):
        q = MagicMock()
        q.scroll.return_value = ([], None)
        return UserStateBuilder(q)

    def test_beginner_elo_prefers_easy(self):
        b = self._builder()
        w_easy = b._difficulty_weight(0.2, elo=1000)   # easy
        w_hard = b._difficulty_weight(0.8, elo=1000)   # hard
        self.assertGreater(w_easy, w_hard)

    def test_expert_elo_prefers_hard(self):
        b = self._builder()
        w_easy = b._difficulty_weight(0.2, elo=1800)
        w_hard = b._difficulty_weight(0.8, elo=1800)
        self.assertGreater(w_hard, w_easy)

    def test_mid_elo_prefers_medium(self):
        b = self._builder()
        w_easy   = b._difficulty_weight(0.1, elo=1400)
        w_medium = b._difficulty_weight(0.5, elo=1400)
        w_hard   = b._difficulty_weight(0.9, elo=1400)
        self.assertGreater(w_medium, w_easy)
        self.assertGreater(w_medium, w_hard)

    def test_minimum_weight_floor(self):
        # even very mismatched difficulty should not be zero
        b = self._builder()
        w = b._difficulty_weight(1.0, elo=800)
        self.assertGreaterEqual(w, 0.1)


# ===========================================================================
# 5. UserStateVector construction
# ===========================================================================

class TestUserStateVector(unittest.TestCase):

    def _qdrant_mock(self, n_hits=5):
        """Mock Qdrant that returns n_hits fake vectors for any query."""
        q = MagicMock()
        rng = np.random.default_rng(42)

        def _scroll(**kwargs):
            hits = []
            for i in range(n_hits):
                h = MagicMock()
                v = rng.random(FULL_DIM).astype(np.float32)
                v /= np.linalg.norm(v)
                h.vector = v.tolist()
                h.payload = {
                    "problem_id":     f"prob_{i}",
                    "topic_tags":     ["arrays"],
                    "difficulty_score": 0.3,
                }
                hits.append(h)
            return hits, None

        q.scroll.side_effect = _scroll
        return q

    def _graph_with_concepts(self, n_concepts=2, mastery=0.8):
        g = UserGraph(user=UserNode(user_id=USER_ID, elo_rating=1200))
        for i in range(n_concepts):
            slug = f"concept_{i}"
            g.add_concept_edge(ConceptEdge(
                concept_slug=slug,
                edge_type=EdgeType.MASTERED,
                mastery_score=mastery,
                confidence=0.66,
                urgency=0.2,
                half_life=7.0,
                last_attempted=_ts(2.0),
            ))
        return g

    def _graph_with_problems(self, n=5):
        g = self._graph_with_concepts()
        for i in range(n):
            g.add_problem_edge(ProblemEdge(
                problem_id=f"prob_{i}",
                edge_type=EdgeType.SOLVED,
                normalised_score=0.8,
                timestamp=_ts(float(i)),
            ))
        return g

    # ------------------------------------------------------------------

    def test_vector_is_1920d(self):
        b = UserStateBuilder(self._qdrant_mock())
        g = self._graph_with_concepts()
        s = b.build(g)
        self.assertIsNotNone(s.vector)
        self.assertEqual(s.vector.shape, (FULL_DIM,))

    def test_vector_is_unit_norm(self):
        b = UserStateBuilder(self._qdrant_mock())
        g = self._graph_with_concepts()
        s = b.build(g)
        norm = float(np.linalg.norm(s.vector))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_no_nan_in_vector(self):
        b = UserStateBuilder(self._qdrant_mock())
        g = self._graph_with_concepts()
        s = b.build(g)
        self.assertFalse(np.any(np.isnan(s.vector)))

    def test_cold_start_flag(self):
        b = UserStateBuilder(self._qdrant_mock())
        g = self._graph_with_concepts()   # 0 solved problems
        s = b.build(g)
        self.assertTrue(s.is_cold_start)

    def test_warm_start_flag(self):
        b = UserStateBuilder(self._qdrant_mock())
        g = self._graph_with_problems(n=5)
        s = b.build(g)
        self.assertFalse(s.is_cold_start)

    def test_cold_start_still_produces_vector(self):
        # Even with 0 solved problems, concept centroids give a vector
        b = UserStateBuilder(self._qdrant_mock())
        g = self._graph_with_concepts(n_concepts=3)
        s = b.build(g)
        self.assertTrue(s.is_valid())

    def test_no_concept_no_vector(self):
        q = MagicMock()
        q.scroll.return_value = ([], None)
        b = UserStateBuilder(q)
        g = UserGraph(user=UserNode(user_id=USER_ID))
        s = b.build(g)
        self.assertFalse(s.is_valid())
        self.assertIsNone(s.to_query_vector())

    def test_embedding_stored_on_user_node(self):
        b = UserStateBuilder(self._qdrant_mock())
        g = self._graph_with_concepts()
        b.build(g)
        self.assertIsNotNone(g.user.embedding)
        self.assertEqual(len(g.user.embedding), FULL_DIM)

    def test_concept_weights_populated(self):
        b = UserStateBuilder(self._qdrant_mock())
        g = self._graph_with_concepts(n_concepts=3)
        s = b.build(g)
        self.assertGreater(len(s.concept_weights), 0)

    def test_qs_and_rgcn_subspaces(self):
        b = UserStateBuilder(self._qdrant_mock())
        g = self._graph_with_concepts()
        s = b.build(g)
        self.assertEqual(s.qs_subspace.shape,   (QS_DIM,))
        self.assertEqual(s.rgcn_subspace.shape, (RGCN_DIM,))

    def test_mastered_concepts_in_state(self):
        b = UserStateBuilder(self._qdrant_mock())
        g = self._graph_with_concepts(n_concepts=2, mastery=0.8)
        s = b.build(g)
        self.assertGreater(len(s.mastered_concepts), 0)

    def test_solved_ids_in_state(self):
        b = UserStateBuilder(self._qdrant_mock())
        g = self._graph_with_problems(n=3)
        s = b.build(g)
        self.assertEqual(len(s.solved_ids), 3)

    def test_n_solved_uses_solved_ids_not_current_edge_type(self):
        """
        Regression test: n_solved was computed by counting problem_edges
        currently marked SOLVED, but add_problem_edge() overwrites the
        stored edge with the MOST RECENT submission's type. If a user
        solves p1, then LATER fails a resubmission on p1, the stored edge
        flips SOLVED -> ATTEMPTED even though the user genuinely solved it
        once. solved_ids is a monotonic add that never removes an entry
        once solved -- counting via problem_edges silently undercounted,
        and after enough such cases a warm user could be misclassified as
        cold-start, skipping the problem-history vector blend entirely.
        """
        g = UserGraph(user=UserNode(user_id=USER_ID, elo_rating=1200))
        g.add_problem_edge(ProblemEdge("p1", EdgeType.SOLVED, normalised_score=0.9, timestamp=100.0))
        g.add_problem_edge(ProblemEdge("p2", EdgeType.SOLVED, normalised_score=0.9, timestamp=101.0))
        g.add_problem_edge(ProblemEdge("p3", EdgeType.SOLVED, normalised_score=0.9, timestamp=102.0))

        # later failed resubmission on p1 -- overwrites the stored edge,
        # but solved_ids must still remember p1 was solved
        g.add_problem_edge(ProblemEdge("p1", EdgeType.ATTEMPTED, normalised_score=0.1, timestamp=200.0))

        self.assertEqual(g.problem_edges["p1"].edge_type, EdgeType.ATTEMPTED)
        self.assertIn("p1", g.solved_ids)

        b = UserStateBuilder(self._qdrant_mock())
        s = b.build(g)
        self.assertEqual(s.n_solved, 3)
        self.assertFalse(s.is_cold_start)

    def test_high_mastery_zero_urgency_gets_higher_weight_than_low_mastery(self):
        q = self._qdrant_mock()
        b = UserStateBuilder(q)
        g = UserGraph(user=UserNode(user_id=USER_ID, elo_rating=1200))
        g.add_concept_edge(ConceptEdge(
            "strong_topic", EdgeType.MASTERED,
            mastery_score=0.9, confidence=1.0, urgency=0.05,
            half_life=30.0, last_attempted=_ts(1.0),
        ))
        g.add_concept_edge(ConceptEdge(
            "weak_topic", EdgeType.LEARNING,
            mastery_score=0.3, confidence=0.33, urgency=0.8,
            half_life=2.0, last_attempted=_ts(20.0),
        ))
        s = b.build(g)
        w_strong = s.concept_weights.get("strong_topic", 0)
        w_weak   = s.concept_weights.get("weak_topic", 0)
        self.assertGreater(w_strong, w_weak)


# ===========================================================================
# 6. Redis cache
# ===========================================================================

class TestRedisCache(unittest.TestCase):

    def test_invalidate_deletes_key(self):
        redis = MagicMock()
        db    = _make_db()
        svc   = UserGraphService(db=db, redis=redis, bkt={}, hlr={})
        svc.invalidate(USER_ID)
        redis.delete.assert_called_once_with(f"user_graph:{USER_ID}")

    def test_cache_hit_skips_db(self):
        g_stored = UserGraph(user=UserNode(user_id=USER_ID))
        cached_json = json.dumps(g_stored.to_dict())

        redis = MagicMock()
        redis.get.return_value = cached_json.encode()

        db  = _make_db()
        svc = UserGraphService(db=db, redis=redis, bkt={}, hlr={})
        g   = svc.get(USER_ID)

        # DB should NOT have been called since Redis returned data
        db.execute.assert_not_called()
        self.assertEqual(g.user.user_id, USER_ID)

    def test_cache_miss_calls_db(self):
        redis = MagicMock()
        redis.get.return_value = None   # cache miss

        db  = _make_db()
        svc = UserGraphService(db=db, redis=redis, bkt={}, hlr={})
        svc.get(USER_ID)

        self.assertTrue(db.execute.called)

    def test_graph_written_to_cache_after_build(self):
        redis = MagicMock()
        redis.get.return_value = None

        db  = _make_db()
        svc = UserGraphService(db=db, redis=redis, bkt={}, hlr={})
        svc.get(USER_ID)

        redis.setex.assert_called_once()
        key = redis.setex.call_args[0][0]
        self.assertEqual(key, f"user_graph:{USER_ID}")


# ===========================================================================
# 7. Edge merging
# ===========================================================================

class TestEdgeMerge(unittest.TestCase):

    def test_add_concept_edge_merges_mastery(self):
        g = UserGraph(user=UserNode(user_id="u1"))
        g.add_concept_edge(ConceptEdge("arrays", EdgeType.MASTERED, mastery_score=0.7))
        g.add_concept_edge(ConceptEdge("arrays", EdgeType.MASTERED, mastery_score=0.9))
        # should keep the higher mastery
        self.assertAlmostEqual(g.concept_edges["arrays"].mastery_score, 0.9)

    def test_add_concept_edge_merges_severity(self):
        g = UserGraph(user=UserNode(user_id="u1"))
        g.add_concept_edge(ConceptEdge("arrays", EdgeType.MASTERED, severity=0.4))
        g.add_concept_edge(ConceptEdge("arrays", EdgeType.WEAK,     severity=0.8))
        self.assertAlmostEqual(g.concept_edges["arrays"].severity, 0.8)

    def test_add_problem_edge_keeps_most_recent(self):
        g = UserGraph(user=UserNode(user_id="u1"))
        old = ProblemEdge("p1", EdgeType.ATTEMPTED, timestamp=_ts(10.0))
        new = ProblemEdge("p1", EdgeType.SOLVED,    timestamp=_ts(1.0))
        g.add_problem_edge(old)
        g.add_problem_edge(new)
        self.assertEqual(g.problem_edges["p1"].edge_type, EdgeType.SOLVED)


# ===========================================================================
# 8. Offline CC edge loader
# ===========================================================================

class TestOfflineCCEdges(unittest.TestCase):

    def setUp(self):
        # load_offline_concept_graph() tries Neo4j FIRST (see
        # pipeline/graphs/neo4j_offline_writer.py), falling back to the
        # local JSON file this test class specifically exercises. Without
        # this patch, a real reachable+populated Neo4j instance (as this
        # repo now has) would return non-empty results and the JSON
        # fallback path these tests are actually testing would never run
        # -- same isolation pattern as TestGraphAssembly's setUp above.
        # load_offline_concept_graph() does the import locally inside the
        # function body (`from pipeline.graphs.neo4j_offline_writer import
        # load_cooccurs_edges, load_prereq_edges`), re-resolved from that
        # module on every call -- so the patch target is the ORIGINAL
        # module's attribute, not a name inside user_graph_service.
        patcher_cooccurs = patch(
            "pipeline.graphs.neo4j_offline_writer.load_cooccurs_edges",
            return_value=[])
        patcher_prereq = patch(
            "pipeline.graphs.neo4j_offline_writer.load_prereq_edges",
            return_value=[])
        patcher_cooccurs.start()
        patcher_prereq.start()
        self.addCleanup(patcher_cooccurs.stop)
        self.addCleanup(patcher_prereq.stop)

    def test_load_from_json(self, tmp_path=None):
        import tempfile, os
        data = [
            {"source": "arrays", "target": "sorting",
             "edgeType": "CO_OCCURS_WITH",
             "shared_problem_count": 5, "jaccard": 0.3},
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            tmp = f.name

        import pipeline.recommender.services.user_graph_service as svc_mod
        from pathlib import Path
        orig = svc_mod._TOPIC_TOPIC_JSON
        svc_mod._TOPIC_TOPIC_JSON = Path(tmp)
        try:
            cc = load_offline_concept_graph(db=None)
        finally:
            svc_mod._TOPIC_TOPIC_JSON = orig
            os.unlink(tmp)

        self.assertIn("arrays", cc)
        self.assertEqual(cc["arrays"][0].target_slug, "sorting")
        self.assertAlmostEqual(cc["arrays"][0].weight, 0.3)

    def test_low_shared_count_filtered(self):
        import tempfile, os
        data = [
            {"source": "a", "target": "b",
             "edgeType": "CO_OCCURS_WITH",
             "shared_problem_count": 1, "jaccard": 0.1},
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f); tmp = f.name

        import pipeline.recommender.services.user_graph_service as svc_mod
        from pathlib import Path
        orig = svc_mod._TOPIC_TOPIC_JSON
        svc_mod._TOPIC_TOPIC_JSON = Path(tmp)
        try:
            cc = load_offline_concept_graph(db=None)
        finally:
            svc_mod._TOPIC_TOPIC_JSON = orig
            os.unlink(tmp)

        self.assertNotIn("a", cc)


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
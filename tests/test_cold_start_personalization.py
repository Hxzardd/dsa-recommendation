"""
tests/test_cold_start_personalization.py

BEFORE / AFTER demonstration for the cold-start "common path" fix, driven
through the REAL pipeline with simulated backend requests.

The bug
-------
Several users import a LeetCode/CF history during onboarding. All of them end
up seeded with every topic at the same flat mastery floor (~0.25) because the
import never produced differentiated mastery. The recommender ranks by
mastery, so with everything tied it drew from one shared pool and handed EVERY
user the same problems -- a generic "common path" that ignored what each user
had actually been practicing.

The one signal that genuinely DIFFERS between those users -- their per-topic
activity (`problems_solved` / `attempt_count` in user_topic_mastery) -- was
SELECTed by the graph builder and then thrown away, so every user built an
identical-looking graph.

The fix
-------
`ConceptEdge` now carries that activity as `engagement`, and the cold-start
paths (`CoursePathPool`, `recommend_topic`) break the mastery tie with it.

What this file exercises
------------------------
1. Unit level      -- CoursePathPool / recommend_topic directly.
2. Full pipeline   -- get_recommendations(): graph build from a fake Postgres
                      -> 7 pools -> filtering -> ranking -> diversity mixer
                      -> title resolution. Four user personas, realistic
                      140-problem catalog.
3. HTTP level      -- a real GET /recommend/{user_id} and
                      GET /topic/recommend/{user_id} through FastAPI, the auth
                      middleware, and the controller, exactly as the backend
                      would call it.

BEFORE is simulated faithfully: the same rows with the activity columns zeroed
(which is precisely what the old code produced, since it discarded them).

Zero real I/O: fake Qdrant, fake Postgres, Neo4j disabled.

Run:
    python -m pytest tests/test_cold_start_personalization.py -v
    python tests/test_cold_start_personalization.py     # prints the report
"""

from __future__ import annotations

import contextlib
import os
import sys
import unittest
from unittest.mock import patch

# Allow `python tests/test_cold_start_personalization.py` to run directly
# (not just under pytest, which puts the project root on sys.path for us).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.recommender.models.user_graph import (
    UserGraph, UserNode, ConceptEdge, EdgeType,
)
from pipeline.recommender.pools.pools import CoursePathPool
from pipeline.recommender.services.topic_recommend import recommend_topic
from pipeline.recommender.services.recommend import get_recommendations
from pipeline.recommender.services.neo4j_graph_store import Neo4jGraphStore

# The HTTP-level test needs the FastAPI app, which imports database.postgres.db
# -- that module raises at import time when DATABASE_URL isn't configured.
# Skip just those tests rather than failing the whole file on such a machine.
try:
    import main as app_main
    import controllers.recommendation_controller as rec_ctl
    import middlewares.auth as auth_mod
    from fastapi.testclient import TestClient
    _HTTP_AVAILABLE = True
    _HTTP_SKIP_REASON = ""
except Exception as _exc:            # pragma: no cover - environment dependent
    _HTTP_AVAILABLE = False
    _HTTP_SKIP_REASON = f"FastAPI app unavailable ({_exc.__class__.__name__}: {_exc})"


# ==========================================================================
# Realistic problem catalog
# ==========================================================================

TOPICS = [
    "array", "string", "hash_map", "two_pointers", "sliding_window",
    "binary_search", "linked_list", "trees", "graphs", "dp",
    "greedy", "backtracking", "math", "stack",
]

# A few natural co-occurrences so problems carry 1-2 tags like the real
# catalog, without exploding tag counts (the cold-start path prefers <=2 tags).
_CO_TAG = {
    "two_pointers": "array", "sliding_window": "array", "hash_map": "array",
    "binary_search": "array", "graphs": "trees", "backtracking": "dp",
    "stack": "string", "greedy": "math",
}

FLOOR_MASTERY = 0.25

# Realistic difficulty density: a real catalog has MANY problems bunched in
# the easy/low-mid band and comparatively few hard ones. A flat 0.20->0.95
# ramp would leave only ONE ZPD-eligible problem per topic for a 0.25-mastery
# cold user, so the diversity mixer would be forced to take "the single
# easiest problem from each topic" and every user's slate would converge
# regardless of personalization -- an artefact of the fixture, not the code.
_DIFFICULTIES = [0.20, 0.23, 0.26, 0.29, 0.32, 0.36,
                 0.41, 0.47, 0.54, 0.63, 0.75, 0.90]
PER_TOPIC = len(_DIFFICULTIES)


class _Pt:
    def __init__(self, pid, tags, diff, score=0.9):
        self.id = pid
        self.payload = {
            "problem_id": pid,
            "topic_tags": tags,
            "difficulty_score": diff,
            "title": pid.replace("_", " ").title(),
            "title_slug": pid.replace("_", "-"),
        }
        self.score = score
        # The fake carries no embeddings, so the ANN-based pools (vector /
        # transfer) stay inert. That's deliberate: this file measures the
        # graph+engagement-driven personalization the fix actually changes,
        # rather than a fabricated similarity ranking that would be identical
        # for every user anyway.
        self.vector = None


def _catalog():
    """~168 problems: every topic gets the full difficulty spread above, and
    most carry a realistic second tag."""
    out = []
    for t in TOPICS:
        for j, diff in enumerate(_DIFFICULTIES):
            tags = [t]
            co = _CO_TAG.get(t)
            if co and j % 3 == 0:
                tags.append(co)
            out.append(_Pt(f"{t}_{j}", tags, diff, score=0.95 - j * 0.02))
    return out


CATALOG = _catalog()


def _match_set(match):
    """Qdrant filters arrive as MatchAny (.any, a list) or MatchValue
    (.value, a scalar) depending on the caller -- the pools use MatchAny,
    UserStateBuilder's centroid lookup uses MatchValue. Normalise both."""
    if match is None:
        return None
    if hasattr(match, "any"):
        return set(match.any)
    if hasattr(match, "value"):
        return {match.value}
    return None


class FakeQdrant:
    """Stable catalog order -- exactly the property that made every
    flat-mastery user receive the same slate before the fix."""

    def __init__(self, problems):
        self.problems = problems

    def scroll(self, collection_name, scroll_filter=None, limit=10,
               with_payload=True, with_vectors=False):
        want_tags = want_pids = diff_range = None
        if scroll_filter is not None:
            for cond in scroll_filter.must:
                key = getattr(cond, "key", None)
                match = getattr(cond, "match", None)
                if key == "topic_tags" and match is not None:
                    want_tags = _match_set(match)
                if key == "problem_id" and match is not None:
                    want_pids = _match_set(match)
                if key == "difficulty_score" and getattr(cond, "range", None) is not None:
                    diff_range = cond.range
        out = []
        for p in self.problems:
            if want_pids is not None and p.id not in want_pids:
                continue
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


# ==========================================================================
# User personas -- different import histories, identical flat mastery
# ==========================================================================
# {topic: (problems_solved, attempt_count)} -- what each user actually ground
# through on LeetCode/CF before signing up.

PERSONAS = {
    "a": {"graphs": (24, 7), "trees": (16, 5), "dp": (4, 2)},
    "b": {"dp": (22, 8), "math": (14, 4), "greedy": (6, 3)},
    "c": {"string": (26, 6), "hash_map": (18, 5), "stack": (7, 2)},
    "d": {"binary_search": (20, 6), "two_pointers": (15, 5),
          "sliding_window": (9, 3)},
}


def _mastery_rows(activity: dict, with_activity: bool = True):
    """
    Rows shaped exactly like user_topic_mastery, in the column order
    UserGraphService._load_topic_mastery unpacks:
        (topic_id, mastery_score, confidence, attempt_count, problems_solved,
         last_attempted, sm2_ef, sm2_interval, next_review_date)

    Every topic sits at the same flat floor -- only the activity columns
    differ between users. `with_activity=False` zeroes those columns, which
    reproduces the BEFORE state (the old code read the row but discarded
    them).
    """
    rows = []
    for t in TOPICS:
        solved, attempts = activity.get(t, (0, 0)) if with_activity else (0, 0)
        rows.append((t, FLOOR_MASTERY, "medium", attempts, solved,
                     None, 2.5, 1, None))
    return rows


class FakeDB:
    """
    Stand-in for the backend's Postgres, dispatching on SQL text rather than
    call order so it stays correct across repeated graph rebuilds.
    Mirrors UserGraphService._build's five queries.
    """

    def __init__(self, user_id, mastery_rows):
        self.user_id = user_id
        self.mastery_rows = mastery_rows
        self.conn = object()          # handle_recommend's finally releases this

    def execute(self, sql, params=None):
        s = " ".join(str(sql).split()).lower()
        db = self

        class _Result:
            def fetchone(self):
                if 'from "user"' in s:
                    # (id, name, onboarding_completed, total_xp, current_level)
                    return (db.user_id, db.user_id, True, 0, 1)
                return None

            def fetchall(self):
                if "from user_topic_mastery" in s:
                    return db.mastery_rows
                # No native submissions (import-only user), no rec log,
                # no concept gaps -- this is what makes them cold-start.
                return []

        return _Result()


@contextlib.contextmanager
def _patched_pipeline():
    """
    Neutralise the two real-Postgres touchpoints inside the graph builder:
    the topic-id -> ML-slug translation (we already use ML slugs) and its
    connection handling.
    """
    ugs = "pipeline.recommender.services.user_graph_service"
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch(f"{ugs}._topic_id_to_ml_slug",
                                  side_effect=lambda conn, tid: tid))
        stack.enter_context(patch(f"{ugs}.get_connection", return_value=object()))
        stack.enter_context(patch(f"{ugs}.release_connection"))
        yield


def _recommend_for(user_id, activity, with_activity=True, k=10):
    """One full pipeline run, as a backend request would trigger."""
    with _patched_pipeline():
        result = get_recommendations(
            user_id=user_id,
            db=FakeDB(user_id, _mastery_rows(activity, with_activity)),
            redis=None,
            neo4j=Neo4jGraphStore(driver=None),      # disabled
            qdrant=FakeQdrant(CATALOG),
            bkt_store={}, hlr_store={},
            k=k, total_n=60,
        )
    return result.recommendations


def _ids(recs):
    return [r["problem_id"] for r in recs]


def _topics_of(recs):
    out = []
    for r in recs:
        out.extend(r.get("topic_tags") or [])
    return out


def _overlap(a, b):
    """Jaccard overlap of two slates: 1.0 = identical, 0.0 = disjoint."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


# ==========================================================================
# 1. Unit level
# ==========================================================================

def _cold_user_graph(user_id: str, activity: dict | None = None) -> UserGraph:
    activity = activity or {}
    g = UserGraph(user=UserNode(user_id=user_id))
    for t in TOPICS:
        solved, attempts = activity.get(t, (0, 0))
        g.add_concept_edge(ConceptEdge(
            concept_slug=t, edge_type=EdgeType.LEARNING,
            mastery_score=FLOOR_MASTERY,
            problems_solved=solved, attempt_count=attempts,
        ))
    return g


class _StubState:
    is_cold_start = True
    def to_query_vector(self):
        return None


class TestUnitLevelBeforeAfter(unittest.TestCase):
    """CoursePathPool / recommend_topic in isolation."""

    def setUp(self):
        self.pool = CoursePathPool(qdrant=FakeQdrant(CATALOG))

    def test_before_all_users_get_the_identical_slate(self):
        slates = {
            name: [c.problem_id for c in
                   self.pool.generate(_cold_user_graph(name), _StubState(), n=10)]
            for name in PERSONAS
        }
        first = next(iter(slates.values()))
        self.assertTrue(first)
        for name, s in slates.items():
            self.assertEqual(s, first,
                             f"{name} should match everyone else BEFORE the fix")

    def test_after_every_user_gets_a_distinct_slate(self):
        slates = {
            name: [c.problem_id for c in
                   self.pool.generate(_cold_user_graph(name, act), _StubState(), n=10)]
            for name, act in PERSONAS.items()
        }
        seen = set()
        for name, s in slates.items():
            key = tuple(s)
            self.assertNotIn(key, seen, f"{name}'s slate duplicates another user's")
            seen.add(key)

    def test_after_slate_leads_with_the_users_own_top_topics(self):
        for name, act in PERSONAS.items():
            top_two = sorted(act, key=lambda t: -(act[t][0] + 0.5 * act[t][1]))[:2]
            cands = self.pool.generate(_cold_user_graph(name, act), _StubState(), n=10)
            lead_tags = set()
            for c in cands[:4]:
                lead_tags.update(c.topic_tags or [])
            self.assertTrue(lead_tags & set(top_two),
                            f"{name}: lead picks {lead_tags} miss top topics {top_two}")

    def test_topic_recommend_picks_each_users_most_engaged_topic(self):
        for name, act in PERSONAS.items():
            expected = max(act, key=lambda t: act[t][0] + 0.5 * act[t][1])
            topic, reason = recommend_topic(_cold_user_graph(name, act))
            self.assertEqual(topic, expected, f"{name} -> {topic}, want {expected}")
            self.assertEqual(reason, "in_progress")

    def test_no_history_user_still_gets_a_slate(self):
        g = _cold_user_graph("nobody")
        self.assertFalse(g.has_engagement_signal())
        self.assertTrue(self.pool.generate(g, _StubState(), n=10))


# ==========================================================================
# 2. Full pipeline (simulated backend request)
# ==========================================================================

class TestFullPipelineBeforeAfter(unittest.TestCase):
    """
    Drives get_recommendations() end to end -- graph built from the fake
    Postgres rows, through all 7 pools, filtering, ranking and the diversity
    mixer -- for four users with different import histories.
    """

    def test_before_pipeline_returns_the_same_problems_for_everyone(self):
        slates = {n: _ids(_recommend_for(n, a, with_activity=False))
                  for n, a in PERSONAS.items()}
        first = next(iter(slates.values()))
        self.assertTrue(first, "expected a non-empty slate from the pipeline")
        for name, s in slates.items():
            self.assertEqual(
                _overlap(s, first), 1.0,
                f"BEFORE: {name} should be identical to every other user")

    def test_after_pipeline_returns_different_problems_per_user(self):
        slates = {n: _ids(_recommend_for(n, a)) for n, a in PERSONAS.items()}
        for s in slates.values():
            self.assertTrue(s)
        names = list(slates)
        overlaps = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                ov = _overlap(slates[names[i]], slates[names[j]])
                overlaps.append(ov)
                self.assertLess(
                    ov, 0.75,
                    f"AFTER: {names[i]} vs {names[j]} overlap {ov:.0%} -- "
                    f"too similar to call personalized")
        avg = sum(overlaps) / len(overlaps)
        self.assertLess(avg, 0.60,
                        f"AFTER: average pairwise overlap {avg:.0%} is too high")

    def test_after_each_user_gets_a_pick_nobody_else_gets(self):
        """The strongest statement of personalization: every user's slate
        contains at least one problem unique to them across the cohort."""
        slates = {n: set(_ids(_recommend_for(n, a))) for n, a in PERSONAS.items()}
        for name, mine in slates.items():
            others = set().union(*(s for n, s in slates.items() if n != name))
            unique = mine - others
            self.assertTrue(
                unique,
                f"{name} received nothing the other users didn't also get "
                f"-- slate is not personalized")

    def test_after_each_slate_reflects_that_users_own_history(self):
        for name, act in PERSONAS.items():
            recs = _recommend_for(name, act)
            tags = _topics_of(recs)
            hits = sum(t in act for t in tags)
            self.assertGreater(
                hits, 0,
                f"{name}: slate {sorted(set(tags))} contains none of their "
                f"practiced topics {sorted(act)}")

    def test_pipeline_output_still_matches_the_backend_schema(self):
        recs = _recommend_for("a", PERSONAS["a"])
        expected = {"problem_id", "title", "title_slug", "difficulty_score",
                    "topic_tags", "source", "recommended_at"}
        valid_sources = {"graph_walk", "vector_similarity", "revision",
                         "sm2_review", "course_path"}
        self.assertTrue(recs)
        for r in recs:
            self.assertEqual(set(r.keys()), expected)
            self.assertIn(r["source"], valid_sources)
            self.assertIsNotNone(r["title"])

    def test_pipeline_respects_k(self):
        self.assertLessEqual(len(_recommend_for("d", PERSONAS["d"], k=5)), 5)

    def test_no_history_user_does_not_crash_the_pipeline(self):
        self.assertIsInstance(_ids(_recommend_for("newbie", {})), list)


# ==========================================================================
# 3. HTTP level -- a real backend request through FastAPI
# ==========================================================================

@unittest.skipUnless(_HTTP_AVAILABLE, _HTTP_SKIP_REASON)
class TestBackendHttpRequest(unittest.TestCase):
    """
    Issues genuine HTTP requests against the FastAPI app the backend talks to:
    routes -> auth middleware -> controller -> ML pipeline. Only the external
    systems (Postgres, Qdrant, Neo4j) are faked.
    """

    @contextlib.contextmanager
    def _serving(self, user_id, activity, with_activity=True):
        fake_db = FakeDB(user_id, _mastery_rows(activity, with_activity))
        with contextlib.ExitStack() as stack:
            ec = stack.enter_context
            # Auth: treat the bearer token as the user id.
            ec(patch.object(auth_mod, "verify_session_token",
                            side_effect=lambda token: token))
            # Controller's external dependencies.
            ec(patch.object(rec_ctl, "get_user_mastery", return_value={}))
            ec(patch.object(rec_ctl, "get_user_hlr", return_value={}))
            ec(patch.object(rec_ctl, "_get_db_wrapper", return_value=fake_db))
            ec(patch.object(rec_ctl, "_get_qdrant", return_value=FakeQdrant(CATALOG)))
            ec(patch.object(rec_ctl, "_get_neo4j_store",
                            return_value=Neo4jGraphStore(driver=None)))
            ec(patch.object(rec_ctl, "save_recommendation_log"))
            ec(patch.object(rec_ctl, "release_connection"))
            ec(_patched_pipeline())
            yield TestClient(app_main.app, raise_server_exceptions=False)

    def _get(self, user_id, activity, path, with_activity=True):
        with self._serving(user_id, activity, with_activity) as client:
            return client.get(path, headers={"Authorization": f"Bearer {user_id}"})

    def test_recommend_endpoint_returns_200_and_schema(self):
        r = self._get("a", PERSONAS["a"], "/recommend/a?limit=10")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["user_id"], "a")
        self.assertTrue(body["recommendations"])
        for rec in body["recommendations"]:
            self.assertIn("problem_id", rec)
            self.assertIn("title", rec)

    def test_recommend_endpoint_personalizes_across_users(self):
        got = {}
        for name, act in PERSONAS.items():
            r = self._get(name, act, f"/recommend/{name}?limit=10")
            self.assertEqual(r.status_code, 200)
            got[name] = _ids(r.json()["recommendations"])
        names = list(got)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                ov = _overlap(got[names[i]], got[names[j]])
                self.assertLess(ov, 0.75,
                                f"HTTP: {names[i]} vs {names[j]} overlap {ov:.0%}")

    def test_recommend_endpoint_before_fix_is_identical_for_everyone(self):
        got = {}
        for name, act in PERSONAS.items():
            r = self._get(name, act, f"/recommend/{name}?limit=10",
                          with_activity=False)
            self.assertEqual(r.status_code, 200)
            got[name] = _ids(r.json()["recommendations"])
        first = next(iter(got.values()))
        for name, s in got.items():
            self.assertEqual(_overlap(s, first), 1.0,
                             f"BEFORE over HTTP: {name} should match everyone")

    def test_topic_recommend_endpoint_differs_per_user(self):
        picks = {}
        for name, act in PERSONAS.items():
            r = self._get(name, act, f"/topic/recommend/{name}")
            self.assertEqual(r.status_code, 200)
            picks[name] = r.json()["topicId"]
        self.assertGreater(len(set(picks.values())), 1,
                           f"topic picks were not personalized: {picks}")

    def test_recommend_endpoint_rejects_mismatched_user(self):
        with self._serving("a", PERSONAS["a"]) as client:
            r = client.get("/recommend/someone_else",
                           headers={"Authorization": "Bearer a"})
        self.assertEqual(r.status_code, 403)


# ==========================================================================
# Human-readable report (python tests/test_cold_start_personalization.py)
# ==========================================================================

def _fmt(recs, width=3):
    return ", ".join(f"{r['problem_id']}" for r in recs[:width]) + (
        f" ... (+{len(recs) - width})" if len(recs) > width else "")


def _topic_mix(recs, top=4):
    counts = {}
    for t in _topics_of(recs):
        counts[t] = counts.get(t, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:top]
    return ", ".join(f"{t}x{c}" for t, c in ranked)


def _report():
    line = "=" * 78
    print("\n" + line)
    print("COLD-START RECOMMENDATIONS -- BEFORE vs AFTER (full pipeline)")
    print(line)
    print(f"\nCatalog: {len(CATALOG)} problems across {len(TOPICS)} topics.")
    print(f"All {len(PERSONAS)} users sit at the SAME flat mastery floor "
          f"({FLOOR_MASTERY}) on every topic.")
    print("They differ only in what they actually practiced during import:\n")
    for name, act in PERSONAS.items():
        pretty = ", ".join(f"{t}({s} solved)" for t, (s, _) in act.items())
        print(f"  {name:<3} {pretty}")

    for label, use_activity in (("BEFORE (activity signal discarded)", False),
                                ("AFTER  (activity signal used)", True)):
        print(f"\n--- {label} " + "-" * (78 - 5 - len(label)))
        slates = {}
        for name, act in PERSONAS.items():
            recs = _recommend_for(name, act, with_activity=use_activity)
            slates[name] = _ids(recs)
            print(f"  {name:<3} next: {_fmt(recs)}")
            print(f"  {'':<3} mix : {_topic_mix(recs)}")
            topic, _ = recommend_topic(
                _cold_user_graph(name, act if use_activity else {}))
            print(f"  {'':<3} topic pick: {topic}")

        names = list(slates)
        pairs = [(names[i], names[j])
                 for i in range(len(names)) for j in range(i + 1, len(names))]
        avg = sum(_overlap(slates[a], slates[b]) for a, b in pairs) / len(pairs)
        verdict = ("every user gets the SAME questions  <-- the bug"
                   if avg > 0.95 else
                   "users get genuinely different questions  <-- fixed")
        print(f"\n  average pairwise slate overlap: {avg:.0%}   {verdict}")

    print(line + "\n")


if __name__ == "__main__":
    _report()
    unittest.main(verbosity=2)

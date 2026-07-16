"""
Recommendation controller.

Stateless ML service -- backend sends everything needed in the request,
ML computes and returns results. No Postgres writes here, read-only
access to mastery/HLR tables.
"""

import logging
import os

import psycopg2
from fastapi import HTTPException

import db_env
from database.postgres.db import get_connection, release_connection, get_user_mastery, get_user_hlr
from pipeline.recommender.services.neo4j_graph_store import Neo4jGraphStore
from pipeline.recommender.services.recommend import get_recommendations, _get_graph, _resolve_titles
from pipeline.recommender.services.topic_recommend import recommend_topic, recommend_problems_for_topic

log = logging.getLogger(__name__)

# FIX (Greptile P1 "Collection Override Is Removed"): this was hardcoded
# to "problems_full", silently dropping the QDRANT_COLLECTION env
# override. Any deployment using a versioned/environment-specific
# collection name would query the wrong one. Restored the override,
# same default value so nothing changes for the current setup.
COLLECTION = os.environ.get("QDRANT_COLLECTION", "problems_full")

# ---------------------------------------------------------------------------
# DBWrapper -- normalizes any psycopg2 cursor row type (plain tuple OR
# RealDictRow) into positional tuples so UserGraphService's row[0]/row[1]
# unpacking always works correctly.
# ---------------------------------------------------------------------------

class DBWrapper:
    def __init__(self, conn):
        self.conn = conn

    def execute(self, query, params=None):
        if params:
            for key in params.keys():
                query = query.replace(f":{key}", f"%({key})s")
        cur = self.conn.cursor()
        cur.execute(query, params or {})
        return _NormalizingCursor(cur)


class _NormalizingCursor:
    def __init__(self, cur):
        self._cur = cur

    def _to_tuple(self, row):
        if row is None:
            return None
        if isinstance(row, dict):
            cols = [d[0] for d in self._cur.description]
            return tuple(row.get(c) for c in cols)
        return tuple(row)

    def fetchone(self):
        return self._to_tuple(self._cur.fetchone())

    def fetchall(self):
        return [self._to_tuple(r) for r in self._cur.fetchall()]


# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

_qdrant_client = None
_neo4j_store = None


def _get_qdrant():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = db_env.qdrant_client(timeout=10)
    return _qdrant_client


def _get_neo4j_store():
    global _neo4j_store
    if _neo4j_store is None:
        driver = db_env.neo4j_driver()
        _neo4j_store = Neo4jGraphStore(
            driver, database=db_env.NEO4J_DATABASE
        )
    return _neo4j_store


def _get_db_wrapper():
    """
    Fresh DBWrapper(psycopg2 connection) per request for UserGraphService's
    own internal bootstrap read (Submission/RecommendationLog/
    ConceptGapProfile history, on top of the separate mastery/hlr fetch
    above) -- DBWrapper already existed in this file but was never wired
    in (db=None was hardcoded), so this whole read path was silently dead:
    a user with real Postgres history always built a cold-start graph
    regardless. None if DATABASE_URL isn't set or the connection fails --
    UserGraphService already degrades gracefully to a cold-start graph
    either way, matching the get_user_mastery/get_user_hlr fallback above.
    Caller must call release_connection() (NOT .close()) on the returned
    wrapper's .conn when done -- get_connection() now draws from a pool.
    """
    try:
        return DBWrapper(get_connection())
    except (RuntimeError, psycopg2.Error) as exc:
        log.warning("Postgres unavailable for UserGraphService bootstrap read (%s: %s)",
                   exc.__class__.__name__, exc)
        return None


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handle_recommend(user_id: str, limit: int = 10) -> dict:
    """
    Full ML pipeline recommendation.

    Reads mastery/HLR from Postgres (backend's tables, read-only from ML
    side), builds a UserGraph, runs the 7-pool recommendation pipeline,
    returns a ranked list shaped to the backend's RecommendationLog schema.
    """
    try:
        mastery = get_user_mastery(user_id) or {}
        hlr = get_user_hlr(user_id) or {}
    except (RuntimeError, psycopg2.Error) as exc:
        # FIX (Greptile P1 "Database Fallback Misses Outages"): only
        # RuntimeError (unset DATABASE_URL) was caught before. A real
        # outage -- connection refused, timeout, auth failure -- raises
        # psycopg2.OperationalError/InterfaceError/etc, all subclasses of
        # psycopg2.Error, NOT RuntimeError. Those escaped this branch
        # entirely and surfaced as an unhandled 500. Now both the
        # "not configured" case and genuine outages degrade to the same
        # graceful cold-start path instead of crashing.
        log.warning("Postgres unavailable (%s: %s) -- cold-start for user %s",
                   exc.__class__.__name__, exc, user_id)
        mastery, hlr = {}, {}

    bkt_store = {user_id: mastery}
    hlr_store = {user_id: hlr}

    # ML never WRITES to Postgres. db_wrapper is a read-only bootstrap
    # fallback for UserGraphService: if a user has no Redis/Neo4j state
    # yet, it seeds the graph from their real Submission/RecommendationLog/
    # ConceptGapProfile rows instead of silently building an empty
    # cold-start graph for a user who actually has history. None if
    # DATABASE_URL isn't set or the connection fails -- same graceful
    # degrade as the mastery/hlr fetch above.
    db_wrapper = _get_db_wrapper()
    try:
        result = get_recommendations(
            user_id=user_id,
            db=db_wrapper,
            redis=None,
            neo4j=_get_neo4j_store(),
            qdrant=_get_qdrant(),
            bkt_store=bkt_store,
            hlr_store=hlr_store,
            collection=COLLECTION,
            total_n=max(limit * 3, 30),
            k=limit,
        )
        return result.to_dict()

    except Exception as exc:
        log.error(
            "Recommendation pipeline failed for user %s: %s",
            user_id, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Recommendation engine encountered an error",
        )
    finally:
        # get_connection() now draws from a ThreadedConnectionPool
        # (maxconn=10, see database/postgres/db.py) -- .close() closes the
        # TCP connection but does NOT free the pool's tracked slot, so this
        # would leak one pool slot on EVERY /recommend call and exhaust the
        # pool after 10 requests. release_connection() (putconn()) is the
        # only way to actually return the slot.
        if db_wrapper is not None:
            release_connection(db_wrapper.conn)


def handle_topic_problem_recommend(user_id: str, topic_id: str, limit: int = 10) -> dict:
    """
    Topic-based problem recommendation: caller (backend) already knows
    which topic it wants ("backend's demand" -- e.g. a topic-picker UI),
    this returns problems for THAT topic ranked by relevance to the
    user's own level (BKT mastery on this topic vs each candidate's real
    Qdrant difficulty_score), not a fixed difficulty band. See
    topic_recommend.recommend_problems_for_topic's docstring for the
    relevance formula.
    """
    try:
        mastery = get_user_mastery(user_id) or {}
        hlr = get_user_hlr(user_id) or {}
    except (RuntimeError, psycopg2.Error) as exc:
        log.warning("Postgres unavailable (%s: %s) -- cold-start for user %s",
                   exc.__class__.__name__, exc, user_id)
        mastery, hlr = {}, {}

    bkt_store = {user_id: mastery}
    hlr_store = {user_id: hlr}

    db_wrapper = _get_db_wrapper()
    try:
        graph = _get_graph(user_id, db_wrapper, None, _get_neo4j_store(), bkt_store, hlr_store)
        qdrant = _get_qdrant()
        candidates = recommend_problems_for_topic(graph, qdrant, COLLECTION, topic_id, n=limit)

        payloads = _resolve_titles([c["problem_id"] for c in candidates], qdrant, COLLECTION)
        recommendations = [
            {
                "problem_id": c["problem_id"],
                "title": payloads.get(c["problem_id"], {}).get("title"),
                "title_slug": payloads.get(c["problem_id"], {}).get("title_slug"),
                "difficulty_score": c["difficulty_score"],
                "topic_tags": c["topic_tags"],
                "predicted_success": c["predicted_success"],
            }
            for c in candidates
        ]
        return {
            "userId": user_id,
            "topicId": topic_id,
            "recommendations": recommendations,
        }
    except Exception as exc:
        log.error(
            "Topic-based problem recommendation failed for user %s topic %s: %s",
            user_id, topic_id, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Topic recommendation engine encountered an error",
        )
    finally:
        if db_wrapper is not None:
            release_connection(db_wrapper.conn)


def handle_topic_recommend(user_id: str) -> dict:
    """
    Single best-next-topic recommendation -- "what ONE topic should this
    user work on next", not a problem list. Builds the same UserGraph
    handle_recommend does (same mastery/HLR read, same db_wrapper bootstrap
    fallback), then picks one topic via topic_recommend.recommend_topic
    instead of running the full 7-pool pipeline -- no Qdrant needed.
    """
    try:
        mastery = get_user_mastery(user_id) or {}
        hlr = get_user_hlr(user_id) or {}
    except (RuntimeError, psycopg2.Error) as exc:
        log.warning("Postgres unavailable (%s: %s) -- cold-start for user %s",
                   exc.__class__.__name__, exc, user_id)
        mastery, hlr = {}, {}

    bkt_store = {user_id: mastery}
    hlr_store = {user_id: hlr}

    db_wrapper = _get_db_wrapper()
    try:
        graph = _get_graph(user_id, db_wrapper, None, _get_neo4j_store(), bkt_store, hlr_store)
        topic_id, reason = recommend_topic(graph)
        return {
            "userId": user_id,
            "topicId": topic_id,
            "reason": reason,
        }
    except Exception as exc:
        log.error(
            "Topic recommendation failed for user %s: %s",
            user_id, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Topic recommendation engine encountered an error",
        )
    finally:
        if db_wrapper is not None:
            release_connection(db_wrapper.conn)
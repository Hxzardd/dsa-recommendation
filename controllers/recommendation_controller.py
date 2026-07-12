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
from database.postgres.db import get_user_mastery, get_user_hlr
from pipeline.recommender.services.neo4j_graph_store import Neo4jGraphStore
from pipeline.recommender.services.recommend import get_recommendations

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

    # db=None -- ML never writes to Postgres. UserGraphService falls back
    # to new_user_graph() for cold-start users when db is None.
    try:
        result = get_recommendations(
            user_id=user_id,
            db=None,
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
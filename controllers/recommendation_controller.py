"""
Recommendation controller.

Bridges the FastAPI routes to the full ML recommendation pipeline
(pipeline.recommender.services.recommend.get_recommendations).

Design decisions:
    - db=None: the pipeline gracefully falls back to new_user_graph() for
      cold-start users. BKT/HLR data flows in via the store dicts,
      populated from our existing Postgres functions in database/postgres/db.py.
    - redis=None, neo4j=None: disables caching tiers. The pipeline handles
      this gracefully — graphs are rebuilt per-request from the store dicts.
      Phase 2 will wire these in for performance.
    - Qdrant client is created lazily on first request, not at import time,
      so the server boots fast and doesn't crash if Qdrant isn't up yet.
"""

import os
import logging

from fastapi import HTTPException
from qdrant_client import QdrantClient

from database.postgres.db import get_user_mastery, get_user_hlr
from pipeline.recommender.services.recommend import get_recommendations

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — all overridable via environment variables
# ---------------------------------------------------------------------------

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "problems_full")

# ---------------------------------------------------------------------------
# Lazy Qdrant singleton — created on first request, not at import time
# ---------------------------------------------------------------------------

_qdrant_client: QdrantClient | None = None


def _get_qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL, timeout=10)
    return _qdrant_client


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def handle_recommend(user_id: str, limit: int = 10) -> dict:
    """
    Full ML pipeline recommendation.

    1. Fetch BKT mastery and HLR state from Postgres.
    2. Package them as {user_id: data} dicts (the format the pipeline expects).
    3. Call get_recommendations() — which internally runs:
       UserGraphService → UserStateBuilder → 7 Pools → CandidateFiltering
       → HeuristicRanker → DiversityMixer → title resolution from Qdrant.
    4. Return the result shaped to the backend's RecommendationLog schema.
    """

    # --- Step 1: fetch user data from Postgres ---
    try:
        mastery = get_user_mastery(user_id)
        hlr = get_user_hlr(user_id)
    except Exception as exc:
        log.error("Postgres fetch failed for user %s: %s", user_id, exc)
        raise HTTPException(status_code=503, detail="Database unavailable")

    # --- Step 2: package for the pipeline ---
    # Pipeline expects: {user_id: {topic: value}}
    bkt_store = {user_id: mastery} if mastery else {}
    hlr_store = {user_id: hlr} if hlr else {}

    # --- Step 3: call the full ML pipeline ---
    try:
        qdrant = _get_qdrant()
        result = get_recommendations(
            user_id=user_id,
            db=None,            # skip SQLAlchemy rebuild — data flows via stores
            redis=None,         # no Redis cache (Phase 2)
            neo4j=None,         # no Neo4j durable store (Phase 2)
            qdrant=qdrant,
            bkt_store=bkt_store,
            hlr_store=hlr_store,
            collection=COLLECTION,
            total_n=max(limit * 3, 30),   # over-fetch for filtering headroom
            k=limit,
        )
        return result.to_dict()

    except Exception as exc:
        log.error(
            "Recommendation pipeline failed for user %s: %s", user_id, exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Recommendation engine encountered an error",
        )
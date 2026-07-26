import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from middlewares.auth import auth_middleware
from routes.submission import router as submission_router
from routes.mastery import router as mastery_router
from routes.recommendation import router as recommendation_router
from routes.seeding import router as seeding_router
from routes.health import router as health_router

load_dotenv()

log = logging.getLogger(__name__)

app = FastAPI()

app.middleware("http")(auth_middleware)

app.include_router(submission_router)
app.include_router(mastery_router)
app.include_router(recommendation_router)
app.include_router(seeding_router)
app.include_router(health_router)


@app.on_event("startup")
def _load_offline_concept_graph():
    """
    FIX: pipeline/recommender/services/user_graph_service.py::
    load_offline_concept_graph() populates the module-level _PREREQ_CACHE
    every UserGraph's cc_edges are copied from (see _load_cc_edges) --
    but nothing ever called it. _PREREQ_CACHE stayed permanently empty,
    so UserGraph.is_locked() (prerequisite gating) and NoveltyPool
    (co-occurrence-based topic exploration) were silent no-ops in
    production regardless of what offline graph data existed, Neo4j or
    local JSON. Called once here so it's populated before the first
    request. db=None -- the Postgres TopicPrerequisite fallback degrades
    gracefully to "no PREREQ edges from that source" exactly like every
    other db=None path in this codebase; Neo4j (tried first) doesn't need
    a db argument at all.
    """
    from pipeline.recommender.services.user_graph_service import load_offline_concept_graph
    try:
        cc = load_offline_concept_graph(db=None)
        log.info("Offline concept graph loaded: %d source topics.", len(cc))
    except Exception as exc:
        log.warning("Offline concept graph failed to load at startup (%s: %s) -- "
                   "prerequisite gating and novelty exploration will see no "
                   "offline edges until this succeeds.", exc.__class__.__name__, exc)


@app.get("/")
def root():
    """
    Readiness check, NOT a static 200. Render (and any other platform) is
    commonly pointed at the root path for its health check; returning a
    hardcoded 200 here meant the service was reported healthy even when
    Postgres/Qdrant were unreachable and every real request was failing.

    Deliberately `def`, not `async def`: handle_health() performs BLOCKING
    dependency I/O (psycopg2, Qdrant HTTP, Neo4j bolt). An async handler runs
    directly on the event loop, so a slow dependency would stall every other
    in-flight request AND make this probe time out on itself. A sync handler
    is dispatched to FastAPI's thread pool, keeping the loop free. Same
    reasoning as routes/health.py's /health.

    This now runs the same dependency probe as /health and returns 503 when a
    CRITICAL dependency (Postgres, Qdrant) is down, so the platform stops
    reporting the instance as healthy while it can't actually serve. An
    optional dependency (Neo4j) being down reports "degraded" but stays 200 --
    the service still works via the Redis/Postgres fallback. Use /live for a
    dependency-free liveness/restart probe.
    """
    from controllers.health_controller import handle_health
    body, ready = handle_health()
    return JSONResponse(status_code=200 if ready else 503, content=body)
import logging

from dotenv import load_dotenv
from fastapi import FastAPI

from middlewares.auth import auth_middleware
from routes.submission import router as submission_router
from routes.mastery import router as mastery_router
from routes.recommendation import router as recommendation_router
from routes.seeding import router as seeding_router

load_dotenv()

log = logging.getLogger(__name__)

app = FastAPI()

app.middleware("http")(auth_middleware)

app.include_router(submission_router)
app.include_router(mastery_router)
app.include_router(recommendation_router)
app.include_router(seeding_router)


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
async def root():
    return {
        "message": "Welcome to Recommendation Service"
    }
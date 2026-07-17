"""
State update service.

This is the "STATE UPDATE LAYER" from the architecture diagram: the thing
that runs immediately after a user solves/attempts a question, updating
everything downstream needs to reflect it before the next recommendation
is served.

Flow (matches the diagram's "user solves a question, telemetry and offline
db inputs -> USER GRAPH -> new candidate with base values" path):

    submission event (problem_id, verdict, hints, etc.)
        --> BKT update (Shraddha's bkt.py)      -> new mastery_score per topic
        --> HLR update (Shraddha's hlr.py)      -> new urgency/half_life per topic
        --> UserGraph mutation:
              - a new ProblemEdge is added for this problem_id
                ("new nodes added w.r.t questions")
              - each affected ConceptEdge's mastery/urgency/half_life/
                confidence/sm2 fields are updated from the BKT/HLR output
        --> Redis cache invalidated (UserGraphService.invalidate)
        --> UserStateVector rebuilt (UserStateBuilder.build)

Persistence note: BKT/HLR stores are still Shraddha's in-memory dicts
(user_mastery_store, user_hlr_store) -- this service mutates those dicts in
place, matching how her submission_controller.py already works. Postgres
persistence for UserTopicMastery/UserProblemReview is a separate backend
task, not something this service does itself.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from pipeline.recommender.models.user_graph import (
    UserGraph, ProblemEdge, EdgeType,
)
from pipeline.recommender.models.user_state import UserStateBuilder, UserStateVector
from pipeline.recommender.services.user_graph_service import UserGraphService
from pipeline.recommender.bkt import process_submission as bkt_process_submission
from pipeline.recommender.hlr import process_hlr


# Confidence mapping consistent with UserGraphService's own DB-sourced
# confidence handling -- low/medium/high mapped to 0.33/0.66/1.0.
_CONFIDENCE_MAP = {"low": 0.33, "medium": 0.66, "high": 1.0}

# A topic gets MASTERED once BKT crosses this threshold (matches bkt.py's
# own MASTERY_THRESHOLD so the graph and Shraddha's mastery store never
# disagree about what counts as mastered).
MASTERY_THRESHOLD = 0.75


@dataclass
class StateUpdateResult:
    """What changed as a result of processing one submission."""
    user_id:            str
    problem_id:          str
    updated_topics:      list            # topic slugs touched by this submission
    newly_mastered:      list            # topics that just crossed MASTERY_THRESHOLD
    graph:               UserGraph
    state:               Optional[UserStateVector]
    processed_at:        float

    def to_dict(self) -> dict:
        return {
            "user_id":       self.user_id,
            "problem_id":    self.problem_id,
            "updated_topics": self.updated_topics,
            "newly_mastered": self.newly_mastered,
            "is_cold_start":  self.state.is_cold_start if self.state else None,
            "vector_dim":     int(self.state.vector.shape[0]) if self.state and self.state.vector is not None else None,
            "processed_at":   self.processed_at,
        }


class StateUpdateService:
    """
    Usage (called by the backend right after a submission is graded):

        svc = StateUpdateService(graph_service, qdrant, bkt_store, hlr_store)
        result = svc.process_submission(user_id, submission_dict)

    submission_dict matches what bkt.py/hlr.py already expect:
        {problemId, verdict, hintsUsed, submissionCount, normalisedScore,
         testCasesPassed, totalTestCases, timestamp}
    """

    def __init__(self, graph_service: UserGraphService, qdrant=None,
                 bkt_store: dict = None, hlr_store: dict = None):
        self.graph_service = graph_service
        self.state_builder = UserStateBuilder(qdrant_client=qdrant)
        # Shraddha's in-memory stores -- mutated in place, same objects
        # UserGraphService was constructed with (bkt=, hlr=) so both this
        # service and graph reads see the same live state.
        self.bkt_store = bkt_store if bkt_store is not None else {}
        self.hlr_store = hlr_store if hlr_store is not None else {}

    def process_submission(self, user_id: str, submission: dict,
                           rebuild_vector: bool = True) -> StateUpdateResult:
        """
        Full state update pipeline for one submission. Returns the updated
        graph (and, unless rebuild_vector=False, the updated 1920-d vector)
        so the caller can serve a recommendation immediately without a
        second round trip.
        """
        now = submission.get("timestamp", time.time())

        # 1. BKT update -- Shraddha's own function, operates on her mastery dict
        user_mastery = self.bkt_store.get(user_id, {})
        updated_mastery, newly_mastered, bkt_results = bkt_process_submission(
            submission, user_mastery)
        self.bkt_store[user_id] = updated_mastery

        # 2. HLR update -- same pattern with her HLR dict
        user_hlr = self.hlr_store.get(user_id, {})
        updated_hlr, hlr_results = process_hlr(submission, user_hlr)
        self.hlr_store[user_id] = updated_hlr

        # 3. Graph mutation: fetch (or cold-start create) the graph, add the
        # new ProblemEdge, update every affected ConceptEdge from BKT+HLR output
        graph = self._get_or_create_graph(user_id)
        self._apply_problem_edge(graph, submission, now)
        updated_topics = self._apply_concept_updates(
            graph, bkt_results, updated_mastery, updated_hlr)

        # 4. Write-through the mutated graph to BOTH tiers. invalidate()
        # first clears any stale Redis entry (defensive: if persist() then
        # only partially succeeds, we haven't left old data behind);
        # persist() writes the fresh graph to Redis AND Neo4j.
        #
        # This used to call _to_cache() directly (Redis only) -- with
        # Neo4j enabled, that silently dropped every post-submission
        # update from the durable tier: once the 5-minute Redis entry
        # expired, the next get() would fall through to Neo4j and find
        # the STALE pre-submission graph, losing the just-solved problem
        # and mastery changes until some other path forced a full rebuild.
        self.graph_service.invalidate(user_id)
        self.graph_service.persist(user_id, graph)

        # 5. Recompute the 1920-d vector so the caller can serve a
        # recommendation immediately with the fresh state.
        state = None
        if rebuild_vector:
            state = self.state_builder.build(graph)

        return StateUpdateResult(
            user_id=user_id,
            problem_id=str(submission.get("problemId", "")),
            updated_topics=updated_topics,
            newly_mastered=newly_mastered,
            graph=graph,
            state=state,
            processed_at=now,
        )

    # ------------------------------------------------------------- helpers

    def _get_or_create_graph(self, user_id: str) -> UserGraph:
        """
        Fetch the existing graph, or create a fresh new-user graph if this
        is the very first submission for a user with no prior state --
        matches the diagram's "no pre-req -> candidate generation (new
        user)" branch. get() already handles this gracefully for a user
        with a User row but no telemetry yet; this only falls back to an
        explicit fresh graph if even the User row lookup fails (e.g. the
        submission arrives before the User row write has propagated).
        """
        try:
            return self.graph_service.get(user_id)
        except ValueError:
            return self.graph_service.new_user_graph(user_id)

    def _apply_problem_edge(self, graph: UserGraph, submission: dict, now: float) -> None:
        """Add a new ProblemEdge node for this submission -- 'new nodes added w.r.t questions'."""
        problem_id = str(submission.get("problemId", ""))
        if not problem_id:
            return
        verdict = submission.get("verdict", "")
        edge_type = EdgeType.SOLVED if verdict == "OK" else EdgeType.ATTEMPTED
        normalised = submission.get("normalisedScore", 0.0)
        graph.add_problem_edge(ProblemEdge(
            problem_id=problem_id,
            edge_type=edge_type,
            normalised_score=float(normalised),
            timestamp=now,
        ))

    def _apply_concept_updates(self, graph: UserGraph, bkt_results: list,
                               updated_mastery: dict, updated_hlr: dict) -> list:
        """
        Update every ConceptEdge touched by this submission with fresh
        mastery (BKT) and urgency/half_life (HLR) values. Returns the list
        of topic slugs that were touched.
        """
        touched = set()

        for r in bkt_results:
            topic = r["topic"]
            touched.add(topic)
            mastery = r["new_p_l"]

            hlr_state = updated_hlr.get(topic, {})
            urgency = hlr_state.get("p_recall")
            urgency = (1.0 - urgency) if urgency is not None else 0.0
            half_life = hlr_state.get("half_life", 1.0)

            edge_type = (EdgeType.MASTERED if mastery >= MASTERY_THRESHOLD
                        else EdgeType.LEARNING)

            existing = graph.concept_edges.get(topic)
            confidence = existing.confidence if existing else 0.66

            # Authoritative overwrite -- the BKT/HLR output computed just now
            # IS the new truth for this topic. add_concept_edge's max-merge
            # would incorrectly refuse to let mastery decrease after a poor
            # submission; see update_concept_state's docstring for why.
            graph.update_concept_state(
                topic,
                edge_type=edge_type,
                mastery_score=mastery,
                confidence=confidence,
                urgency=urgency,
                half_life=half_life,
                last_attempted=time.time(),
            )

        return list(touched)


def process_submission_and_get_vector(
    user_id: str, submission: dict,
    graph_service: UserGraphService, qdrant=None,
    bkt_store: dict = None, hlr_store: dict = None,
) -> StateUpdateResult:
    """Convenience one-shot wrapper."""
    return StateUpdateService(
        graph_service, qdrant=qdrant, bkt_store=bkt_store, hlr_store=hlr_store
    ).process_submission(user_id, submission)
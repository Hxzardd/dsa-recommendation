"""
State update service.

This is the "STATE UPDATE LAYER" from the architecture diagram: the thing
that runs immediately after a user solves/attempts a question, updating
everything downstream needs to reflect it before the next recommendation
is served.

Flow (matches the diagram's "user solves a question, telemetry and offline
db inputs -> USER GRAPH -> new candidate with base values" path):

    /update's computed BKT/HLR result
        --> UserGraph mutation:
              - a new ProblemEdge is added for this problem_id
                ("new nodes added w.r.t questions")
              - each affected ConceptEdge's mastery/urgency/half_life/
                confidence/sm2 fields are updated from the BKT/HLR output
        --> Redis cache invalidated (UserGraphService.invalidate)
        --> Neo4j graph persisted

This service deliberately does not calculate or persist BKT/HLR. /update is
the authoritative owner of those operations and passes its already-computed
results here solely to project them into the recommendation graph.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pipeline.recommender.models.user_graph import (
    UserGraph, ProblemEdge, EdgeType,
)
from pipeline.recommender.services.user_graph_service import UserGraphService
from database.postgres.db import (
    get_connection, release_connection, _topic_id_to_ml_slug,
)
from database.postgres.topic_taxonomy import canonical_for_ml_slug


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
    processed_at:        float

    def to_dict(self) -> dict:
        return {
            "user_id":       self.user_id,
            "problem_id":    self.problem_id,
            "updated_topics": self.updated_topics,
            "newly_mastered": self.newly_mastered,
            "processed_at":   self.processed_at,
        }


class StateUpdateService:
    """
    Usage (called by /update after BKT and HLR have been computed):

        svc = StateUpdateService(graph_service)
        result = svc.apply_update(
            submission_dict, updated_mastery, mastered_topics,
            bkt_results, updated_hlr, hlr_results,
        )
    """

    def __init__(self, graph_service: UserGraphService):
        self.graph_service = graph_service

    def apply_update(
        self,
        submission: dict,
        updated_mastery: dict,
        mastered_topics: list,
        bkt_results: list,
        updated_hlr: dict,
        hlr_results: list,
    ) -> StateUpdateResult:
        """
        Project /update's already-computed BKT/HLR results into the user
        graph. This method must never recalculate BKT or HLR.
        """
        user_id = str(submission.get("userId", ""))
        now = submission.get("timestamp", time.time())

        # Graph mutation: fetch (or cold-start create) the graph, add the
        # new ProblemEdge, update every affected ConceptEdge from BKT+HLR output
        graph = self._get_or_create_graph(user_id)
        self._apply_problem_edge(graph, submission, now)
        updated_topics = self._apply_concept_updates(
            graph, bkt_results, updated_mastery, updated_hlr)

        # Write-through the mutated graph to BOTH tiers. invalidate()
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

        return StateUpdateResult(
            user_id=user_id,
            problem_id=str(submission.get("problemId", "")),
            updated_topics=updated_topics,
            newly_mastered=mastered_topics,
            graph=graph,
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

    @staticmethod
    def _resolve_ml_slug(conn, raw_topic: str):
        """
        raw_topic is either a backend topic.id (translate via
        _topic_id_to_ml_slug, same as UserGraphService._build()) or already
        a genuine ML slug/legacy tag (the static-mapping fallback caller
        convention -- canonical_for_ml_slug passes those through/collapses
        legacy tags unchanged). None if it's neither -- callers must skip
        the write, not guess.
        """
        slug = _topic_id_to_ml_slug(conn, raw_topic)
        if slug is not None:
            return slug
        return canonical_for_ml_slug(raw_topic)

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

        bkt_results/updated_hlr's topic keys have two possible conventions
        depending on how bkt.py::process_submission/hlr.py::process_hlr
        were called: when the backend sends problemTopics, the key is the
        backend's own topic.id (verbatim passthrough -- see
        submission_controller.py::_split_current_state); when problemTopics
        is omitted (the "static mapping fallback for legacy callers" path,
        e.g. problem_to_topics-driven tests/direct ML-internal callers),
        the key is already a genuine ML slug. Every OTHER ConceptEdge in
        this graph -- built by UserGraphService._build() from
        user_topic_mastery -- is keyed by the ML slug (see its docstring:
        an untranslated topic.id "never matches any real Qdrant topic_tags
        value", so every downstream pool silently finds zero candidates for
        that topic). _resolve_ml_slug tries the topic.id->slug translation
        first, falling back to treating the key as an already-valid ML slug,
        so both calling conventions land on the same ML-slug-keyed
        ConceptEdge that _build() would also produce for that topic.
        """
        touched = set()
        conn = get_connection()
        try:
            for r in bkt_results:
                raw_topic = r["topic"]
                topic = self._resolve_ml_slug(conn, str(raw_topic))
                if topic is None:
                    # Neither a backend topic.id with a known ML-slug
                    # mapping, nor a recognised ML slug itself -- skip the
                    # graph write rather than guess a key, same convention
                    # _build() uses for the cold-start path.
                    continue
                touched.add(topic)
                mastery = r["new_p_l"]

                hlr_state = updated_hlr.get(raw_topic, {})
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
        finally:
            release_connection(conn)

        return list(touched)

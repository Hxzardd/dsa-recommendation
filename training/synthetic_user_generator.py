"""
training/synthetic_user_generator.py

Bootstraps training/dataset_generator.py + training/label_generator.py with
realistic RecommendationEvents when no real recommendation_log/submission
history exists yet (confirmed empty in production as of this module's
creation -- see the database audit this component follows from).

What this module does NOT do:
  - does not generate feature vectors (feature_extractor.py still owns
    every feature computation, untouched)
  - does not fabricate dataset rows or labels directly
  - does not bypass the recommender: every RecommendationEvent's
    `candidates` list comes from actually calling
    AdaptiveDifficultyController, PoolGenerationOrchestrator,
    CandidateFilteringLayer, HeuristicRanker, and DiversityMixer -- the
    exact same real classes pipeline/recommender/services/recommend.py's
    get_recommendations() calls, in the same order, against a real (or
    real-shaped) Qdrant catalog. Only the USERS are synthetic; the
    recommender, the candidate catalog, and the labeling strategy are real.

What IS synthetic: a persona-driven UserGraph per fake user, and a
persona-conditioned interaction model deciding whether each real candidate
the real pipeline proposed gets ignored / attempted / solved -- driven by
the candidate's own real `predicted_success` (from
CandidateFilteringLayer's ZPD estimator), not an arbitrary random label.

Once real recommendation_log/submission history exists, this module is no
longer needed for training -- DatasetGenerator/LabelGenerator consume real
RecommendationEvents (built from real recommendation_log rows) exactly the
same way they consume the synthetic ones here.

_finalize_candidates() below mirrors get_recommendations()'s own
post-generation orchestration (recommend.py, the cold-start tag
simplification + MergedCandidate re-attachment + diversity-mixing
sequence) exactly, because that sequence isn't factored into an
importable function there and modifying recommend.py is out of scope for
this component. No scoring/filtering/ranking/diversity FORMULA is
reimplemented -- every decision is delegated to the same real classes
get_recommendations() itself calls. If recommend.py ever exposes this
sequence as a reusable function, this method should call that instead.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from pipeline.recommender.hlr import (
    MAX_HALF_LIFE,
    MIN_HALF_LIFE,
    calculate_urgency,
    update_half_life,
)
from pipeline.recommender.models.user_graph import (
    ConceptConceptEdge, ConceptEdge, EdgeType, UserGraph, UserNode,
)
from pipeline.recommender.services.adaptive_difficulty import AdaptiveDifficultyController
from pipeline.recommender.services.candidate_filtering import CandidateFilteringLayer, MergedCandidate
from pipeline.recommender.services.diversity_mixer import DiversityMixer
from pipeline.recommender.services.heuristic_ranker import HeuristicRanker
from pipeline.recommender.services.pool_generation import PoolGenerationOrchestrator
from pipeline.recommender.services.recommend import _build_state
from pipeline.recommender.services.user_graph_service import load_offline_concept_graph

from training.config import RANDOM_SEED
from training.dataset_generator import RecommendationEvent
from training.label_generator import NEVER_ATTEMPTED, InMemoryOutcomeProvider, RecommendationOutcome

log = logging.getLogger(__name__)

SECONDS_PER_DAY = 86400.0
DEFAULT_ATTEMPT_LATENCY_SECONDS = 3600.0   # persona interacts ~1hr after being shown the slate

# Simulation clock start -- a real-looking Unix epoch (2023-11-14), not 0.0.
# LabelGenerator.validate() flags any recommended_at <= 0 as malformed, and a
# genuine recommended_at is always a large epoch value in production anyway.
SIMULATION_EPOCH_START = 1_700_000_000.0


def _iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def load_real_concept_graph() -> dict[str, list[ConceptConceptEdge]]:
    """
    The REAL offline concept-concept graph (PREREQ + COOCCURS), for
    SyntheticUserGenerator's `cc_edges=` parameter -- not a separate
    representation, this returns exactly the same
    dict[str, list[ConceptConceptEdge]] shape
    user_graph_service.py::_load_cc_edges attaches to every real user's
    graph, built from the exact same two real sources:

      - COOCCURS: load_offline_concept_graph()'s own Neo4j-or-local-JSON
        fallback (question-graph/data/topic_topic_edges.json), unchanged.
      - PREREQ: the real `topic_prerequisite` Postgres table, same rows/
        columns/edge-direction/ML-slug-translation
        (database.postgres.db._topic_id_to_ml_slug) load_offline_concept_
        graph()'s own Postgres branch already uses -- but that branch is
        called from main.py as load_offline_concept_graph(db=None) (see
        its call site), and even when a db IS passed, that function's
        `db.execute(sql).fetchall()` call assumes a SQLAlchemy-style
        connection, which this repo's actual DB layer
        (database.postgres.db.get_connection(), a raw psycopg
        connection with .cursor()) doesn't provide -- the branch would
        silently except/log and produce zero PREREQ edges either way, in
        production exactly as here. Loading PREREQ edges directly with the
        connection API that actually works closes that gap rather than
        working around it.
    """
    cc = load_offline_concept_graph(db=None)

    try:
        from database.postgres.db import get_connection, release_connection, _topic_id_to_ml_slug

        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT topic_id, prerequisite_id FROM topic_prerequisite")
            rows = cur.fetchall()
            n_loaded = 0
            for topic_id, prereq_id in rows:
                prereq_slug = _topic_id_to_ml_slug(conn, str(prereq_id))
                topic_slug = _topic_id_to_ml_slug(conn, str(topic_id))
                if prereq_slug is None or topic_slug is None:
                    continue   # unmapped topic id -- skipped, not guessed (matches _load_topic_mastery's own convention)
                cc.setdefault(prereq_slug, []).append(ConceptConceptEdge(
                    source_slug=prereq_slug, target_slug=topic_slug,
                    edge_type=EdgeType.PREREQ, weight=1.0,
                ))
                n_loaded += 1
            log.info("Loaded %d real PREREQ edges from topic_prerequisite", n_loaded)
        finally:
            release_connection(conn)
    except Exception as exc:
        log.warning("Failed to load topic_prerequisite PREREQ edges: %s", exc)

    return cc


# ============================================================================
# Personas
# ============================================================================

@dataclass(frozen=True)
class PersonaConfig:
    """
    Every field here is a configurable probability/parameter, not a fixed
    outcome -- SyntheticUserGenerator conditions actual attempt/solve
    decisions on each candidate's real predicted_success and pool
    provenance, using these as the persona-specific modifiers, not as
    labels themselves.
    """
    name: str
    mastery_range: tuple[float, float]        # initial per-topic mastery sampling range
    topics_touched_range: tuple[int, int]     # how many topics this persona starts with any history in
    base_engagement_probability: float        # baseline chance of attempting a shown candidate at all
    review_diligence: float                   # engagement boost for spaced-review-pool (E) candidates
    explore_willingness: float                # engagement boost for novelty-pool (G) candidates
    solve_skill_multiplier: float              # scales the candidate's real predicted_success into a solve probability
    hard_problem_persistence: float           # probability of multiple attempts (vs. one-and-done) on a failed candidate
    sessions_range: tuple[int, int]           # how many recommendation events this persona generates
    session_gap_days_range: tuple[float, float]   # elapsed real time between consecutive sessions -- this is
                                               # what drives forgetting (HLR half-life decay -> urgency growth) and
                                               # inactivity periods; short/frequent for engaged personas, long/erratic
                                               # for inconsistent or returning ones


PERSONAS: dict[str, PersonaConfig] = {
    "beginner": PersonaConfig(
        name="beginner", mastery_range=(0.05, 0.30), topics_touched_range=(1, 3),
        base_engagement_probability=0.55, review_diligence=0.10, explore_willingness=0.05,
        solve_skill_multiplier=0.85, hard_problem_persistence=0.35, sessions_range=(3, 6),
        session_gap_days_range=(1.0, 4.0),
    ),
    "intermediate": PersonaConfig(
        name="intermediate", mastery_range=(0.30, 0.60), topics_touched_range=(4, 10),
        base_engagement_probability=0.65, review_diligence=0.20, explore_willingness=0.15,
        solve_skill_multiplier=1.0, hard_problem_persistence=0.45, sessions_range=(5, 10),
        session_gap_days_range=(1.0, 5.0),
    ),
    "advanced": PersonaConfig(
        name="advanced", mastery_range=(0.55, 0.85), topics_touched_range=(10, 20),
        base_engagement_probability=0.70, review_diligence=0.15, explore_willingness=0.30,
        solve_skill_multiplier=1.1, hard_problem_persistence=0.55, sessions_range=(6, 12),
        session_gap_days_range=(1.0, 4.0),
    ),
    "competitive": PersonaConfig(
        # High activity, high willingness to attempt hard problems, low
        # engagement dependence on review pools (they push forward, not back).
        # Short gaps -- competitive users check in almost daily.
        name="competitive", mastery_range=(0.45, 0.80), topics_touched_range=(8, 18),
        base_engagement_probability=0.80, review_diligence=0.05, explore_willingness=0.35,
        solve_skill_multiplier=1.15, hard_problem_persistence=0.70, sessions_range=(10, 18),
        session_gap_days_range=(0.5, 2.0),
    ),
    "reviewer": PersonaConfig(
        # Strongly prioritises review-pool (E) engagement over novelty --
        # the persona this recommender's spaced-review pool exists for.
        # Deliberately spaced (not too frequent, not too sparse) so review
        # urgency actually has time to build between sessions -- a reviewer
        # who checked in every few hours would never see anything overdue.
        name="reviewer", mastery_range=(0.35, 0.70), topics_touched_range=(6, 14),
        base_engagement_probability=0.60, review_diligence=0.35, explore_willingness=0.05,
        solve_skill_multiplier=0.95, hard_problem_persistence=0.40, sessions_range=(5, 9),
        session_gap_days_range=(2.0, 6.0),
    ),
    "inconsistent_learner": PersonaConfig(
        # Low, noisy engagement -- models a real segment: mastery exists but
        # attention is unreliable, independent of candidate quality. Wide,
        # erratic gaps between sessions -- genuine inactivity periods, not
        # just low per-candidate engagement.
        name="inconsistent_learner", mastery_range=(0.15, 0.55), topics_touched_range=(3, 9),
        base_engagement_probability=0.30, review_diligence=0.05, explore_willingness=0.05,
        solve_skill_multiplier=0.80, hard_problem_persistence=0.20, sessions_range=(2, 5),
        session_gap_days_range=(3.0, 21.0),
    ),
    "returning_user": PersonaConfig(
        # Real prior mastery (was active before), but few sessions in this
        # simulation window -- models someone coming back after a gap. Very
        # long gaps between the few sessions they do have -- this IS the
        # persona's defining behavior, not incidental.
        name="returning_user", mastery_range=(0.30, 0.65), topics_touched_range=(8, 16),
        base_engagement_probability=0.50, review_diligence=0.25, explore_willingness=0.10,
        solve_skill_multiplier=0.90, hard_problem_persistence=0.30, sessions_range=(2, 4),
        session_gap_days_range=(14.0, 60.0),
    ),
}


# ============================================================================
# Configuration and results
# ============================================================================

@dataclass(frozen=True)
class SimulatorConfig:
    num_users: int = 20
    persona_distribution: Optional[dict[str, float]] = None   # persona name -> weight; None = uniform
    total_n: int = 30
    k: int = 10
    max_per_pool: int = 3
    max_per_topic: int = 3
    collection: str = "problems_full"
    random_seed: int = RANDOM_SEED


@dataclass
class SimulationResult:
    events: list[RecommendationEvent]
    outcome_provider: InMemoryOutcomeProvider


# ============================================================================
# Generator
# ============================================================================

class SyntheticUserGenerator:
    """
    Usage:
        gen = SyntheticUserGenerator(qdrant=real_or_fake_qdrant_client,
                                     topic_slugs=real_catalog_topic_slugs)
        result = gen.generate()
        df = DatasetGenerator().generate_dataframe(result.events)
        labeled = LabelGenerator(provider=result.outcome_provider).label_dataframe(df)

    `topic_slugs` must be real catalog topic slugs (e.g. queried from the
    `topic` table, or the ML taxonomy file) -- this module never invents
    topic names, since a fabricated slug would never match anything real
    pools/Qdrant know about and every candidate for it would be empty.
    """

    def __init__(self, qdrant, topic_slugs: list[str], config: Optional[SimulatorConfig] = None,
                 cc_edges: Optional[dict[str, list[ConceptConceptEdge]]] = None):
        """
        cc_edges: the real offline concept-concept graph (PREREQ + COOCCURS),
        as returned by load_real_concept_graph() -- attached in full to every
        synthetic user's graph, mirroring user_graph_service.py::_load_cc_edges
        (which attaches the FULL offline graph to every real user too, not
        just edges for topics the user has touched, since is_locked() needs
        to look up prerequisites for candidate concepts the user hasn't
        necessarily touched yet).

        Defaults to {} (no cc_edges) rather than loading them here, so this
        class stays offline-testable with a fake Qdrant and no DB connection
        -- exactly the FakeQdrant/FakeRedis convention already used across
        this test suite. Callers generating a real bootstrap dataset should
        explicitly pass load_real_concept_graph().
        """
        if not topic_slugs:
            raise ValueError("topic_slugs must be a non-empty list of real catalog topic slugs")
        self.qdrant = qdrant
        self.topic_slugs = list(topic_slugs)
        self.config = config or SimulatorConfig()
        self.cc_edges = cc_edges or {}
        self._rng = random.Random(self.config.random_seed)

    # ------------------------------------------------------------- persona/graph setup

    def _pick_persona(self) -> PersonaConfig:
        names = list(PERSONAS.keys())
        dist = self.config.persona_distribution
        weights = [dist.get(n, 0.0) for n in names] if dist else [1.0] * len(names)
        return PERSONAS[self._rng.choices(names, weights=weights, k=1)[0]]

    @staticmethod
    def _severity_from_history(attempt_count: int, problems_solved: int) -> float:
        """
        A concept-weakness signal derived the same way ConceptGapProfile.severity
        would be in production -- from actual repeated-failure history, not
        an injected number. Needs at least 2 attempts before it says
        anything (a single failure isn't evidence of a real gap); 0.0 until
        then. Scales with how much worse than a coin-flip the fail rate is,
        capped at 1.0 -- e.g. 1/5 solved -> 0.6, 0/4 solved -> 1.0.
        """
        if attempt_count < 2:
            return 0.0
        solve_rate = problems_solved / attempt_count
        return round(min(1.0, max(0.0, (1.0 - solve_rate) * 1.5 - 0.5)), 4)

    def _build_initial_graph(self, user_id: str, persona: PersonaConfig, now: float) -> UserGraph:
        graph = UserGraph(user=UserNode(user_id=user_id))

        # Attach the FULL real offline concept-concept graph -- same
        # convention as user_graph_service.py::_load_cc_edges (real users
        # get every PREREQ/COOCCURS edge regardless of what they've
        # touched, since candidate-side prerequisite lookups need it).
        for edges in self.cc_edges.values():
            for cc_edge in edges:
                graph.add_cc_edge(cc_edge)

        n_topics = min(self._rng.randint(*persona.topics_touched_range), len(self.topic_slugs))
        touched = self._rng.sample(self.topic_slugs, n_topics) if n_topics > 0 else []
        for slug in touched:
            mastery = round(self._rng.uniform(*persona.mastery_range), 4)

            # A plausible attempt/solve history consistent with the sampled
            # mastery -- mastery had to come from SOME activity, so seeding
            # zero activity alongside non-zero mastery would itself be the
            # unrealistic choice. Not fabricating a target label: this only
            # seeds ConceptEdge.attempt_count/problems_solved, the same two
            # fields real imported CF/LC history populates.
            attempt_count = max(1, round(mastery * 10) + self._rng.randint(0, 2))
            problems_solved = round(attempt_count * mastery)

            # A cold-start seed for how long ago this topic was last
            # touched (needed to compute a real, non-zero starting urgency
            # via HLR's own forgetting curve, not a hand-picked constant) --
            # sampled independently of mastery, since "started a while ago"
            # and "how well you know it" are different axes.
            last_attempted = now - self._rng.uniform(1.0, 30.0) * SECONDS_PER_DAY

            # Initial half-life scales with solve history, same base
            # formula hlr.py::seed_half_life_from_cf uses for a real
            # imported CF/LC history (MIN_HALF_LIFE * 2**(solve_count/5),
            # capped at MAX_HALF_LIFE) -- a topic with real seeded mastery
            # implies real retention, not a one-day memory; defaulting
            # every topic to the same MIN_HALF_LIFE regardless of mastery
            # made forgetting-curve urgency saturate to ~1.0 for almost
            # every topic within days regardless of how well it was known.
            half_life = min(MAX_HALF_LIFE, MIN_HALF_LIFE * (2 ** (problems_solved / 5)))
            urgency = calculate_urgency(
                {"last_review": _iso(last_attempted), "half_life": half_life}, now,
            )

            graph.add_concept_edge(ConceptEdge(
                concept_slug=slug,
                edge_type=EdgeType.MASTERED if mastery >= 0.75 else EdgeType.LEARNING,
                mastery_score=mastery,
                urgency=urgency,
                half_life=half_life,
                last_attempted=last_attempted,
                attempt_count=attempt_count,
                problems_solved=problems_solved,
                severity=self._severity_from_history(attempt_count, problems_solved),
            ))
        return graph

    def _refresh_temporal_state(self, graph: UserGraph, now: float) -> None:
        """
        Recompute urgency for every concept the user has touched, based on
        real elapsed time since last_attempted -- HLR's own forgetting
        curve (calculate_urgency), not a hand-incremented counter. Called
        once per session, before AdaptiveDifficultyController builds its
        plan, so pool weighting/SpacedReviewPool see genuinely-aged state
        rather than whatever was true when the topic was first touched.
        """
        for slug, edge in graph.concept_edges.items():
            edge.urgency = calculate_urgency(
                {"last_review": _iso(edge.last_attempted), "half_life": edge.half_life}, now,
            )

    # ------------------------------------------------------------- real-pipeline orchestration

    def _finalize_candidates(self, graph: UserGraph, gen_result) -> list[MergedCandidate]:
        """See module docstring: mirrors get_recommendations()'s own
        post-generation sequence, delegating every actual decision to the
        real CandidateFilteringLayer/HeuristicRanker/DiversityMixer."""
        k = self.config.k
        merged_candidates = gen_result.merged_candidates

        if gen_result.difficulty_plan.is_cold_start:
            for max_tags in (2, 3):
                simple = [mc for mc in merged_candidates if len(mc.topic_tags or []) <= max_tags]
                if len(simple) >= k:
                    merged_candidates = simple
                    break

        if not merged_candidates:
            return []

        filtering = CandidateFilteringLayer(graph)

        # gen_result.merged_candidates can legitimately carry
        # predicted_success=None: PoolGenerationOrchestrator retries with
        # apply_zpd=False when the strict ZPD band empties a cold/low-mastery
        # user's pool (candidate_filtering.py::run docstring), and
        # _apply_zpd_filter -- the only place predicted_success gets set --
        # is skipped on that path. feature_extractor.py::extract_features
        # already back-fills this the same way for the real pipeline; mirror
        # it here so _simulate_interaction always sees a real estimate.
        for mc in merged_candidates:
            if mc.predicted_success is None:
                mc.predicted_success = filtering._default_success_estimator(mc, graph)

        ranker_rows = filtering.to_ranker_input(merged_candidates)
        if not ranker_rows:
            return []

        ranker = HeuristicRanker()
        ranked_rows = ranker.top_k(ranker_rows, k=max(k * 3, k))

        by_id = {mc.problem_id: mc for mc in merged_candidates}
        scores_by_id = {row["problem_id"]: row["rank_score"] for row in ranked_rows}
        ordered_candidates = [by_id[row["problem_id"]] for row in ranked_rows if row["problem_id"] in by_id]

        effective_max_per_pool = k if gen_result.difficulty_plan.is_cold_start else self.config.max_per_pool
        mixer = DiversityMixer(max_per_pool=effective_max_per_pool, max_per_topic=self.config.max_per_topic)
        return mixer.mix(ordered_candidates, k=k, relevance_scores=scores_by_id)

    # ------------------------------------------------------------- behaviour model

    def _simulate_interaction(self, persona: PersonaConfig, candidate: MergedCandidate) -> RecommendationOutcome:
        """
        Attempt/solve decisions are conditioned on the candidate's own real
        predicted_success (from CandidateFilteringLayer's ZPD estimator,
        already computed by the real pipeline) and pool provenance --
        never an arbitrary random label.
        """
        pool_sources = set(candidate.pool_sources)
        engagement = persona.base_engagement_probability
        if "E" in pool_sources:
            engagement += persona.review_diligence
        if "G" in pool_sources:
            engagement += persona.explore_willingness
        engagement = min(1.0, max(0.0, engagement))

        if self._rng.random() > engagement:
            return NEVER_ATTEMPTED   # ignored

        predicted_success = candidate.predicted_success if candidate.predicted_success is not None else 0.68
        solve_probability = min(0.97, max(0.02, predicted_success * persona.solve_skill_multiplier))
        solved = self._rng.random() < solve_probability

        if solved:
            attempt_count = self._rng.randint(1, 3)
        else:
            attempt_count = (
                self._rng.randint(2, 5) if self._rng.random() < persona.hard_problem_persistence else 1
            )

        return RecommendationOutcome(
            was_attempted=True, attempt_count=attempt_count,
            first_attempt_at=None, solved=solved, solved_at=None,
        )

    def _apply_outcome_to_graph(self, graph: UserGraph, candidate: MergedCandidate,
                                outcome: RecommendationOutcome, attempt_at: float) -> None:
        """Mutates the LIVE, continuing graph (used for subsequent sessions
        of this same synthetic user) -- never the snapshot captured in an
        already-emitted RecommendationEvent (see generate()'s use of
        UserGraph.from_dict/to_dict for that separation).

        half_life/urgency updates reuse pipeline.recommender.hlr's own
        update_half_life()/calculate_urgency() -- the exact forgetting-curve
        math a real submission would drive through HLR, not a hand-picked
        constant. Good performance lengthens half_life (slower forgetting,
        review pressure eases); poor performance shortens it (faster
        forgetting, review pressure builds sooner) -- real spaced-repetition
        behavior, driven by the real outcome we already simulated."""
        if not outcome.was_attempted:
            return
        if outcome.solved:
            graph.solved_ids.add(candidate.problem_id)

        delta = 0.12 if outcome.solved else -0.02
        # performance is HLR's own [0,1] scale (0.5 = neutral); a solve is
        # clearly-good performance, a failed attempt clearly-poor -- not a
        # graded score we have no real basis to invent more precisely.
        performance = 0.9 if outcome.solved else 0.25

        for slug in (candidate.topic_tags or []):
            existing = graph.concept_edges.get(slug)
            current_mastery = existing.mastery_score if existing else 0.3
            new_mastery = round(min(1.0, max(0.0, current_mastery + delta)), 4)

            prev_half_life = existing.half_life if existing else MIN_HALF_LIFE
            prev_last_attempted = existing.last_attempted if existing else None
            days_since = (
                (attempt_at - prev_last_attempted) / SECONDS_PER_DAY
                if prev_last_attempted is not None else 0.0
            )
            new_half_life = update_half_life(prev_half_life, performance, days_since)
            new_urgency = calculate_urgency(
                {"last_review": _iso(attempt_at), "half_life": new_half_life}, attempt_at,
            )

            prev_attempts = existing.attempt_count if existing else 0
            prev_solved = existing.problems_solved if existing else 0
            new_attempts = prev_attempts + outcome.attempt_count
            new_solved = prev_solved + (1 if outcome.solved else 0)

            graph.update_concept_state(
                slug, mastery_score=new_mastery, urgency=new_urgency,
                half_life=new_half_life, last_attempted=attempt_at,
                attempt_count=new_attempts, problems_solved=new_solved,
                severity=self._severity_from_history(new_attempts, new_solved),
                edge_type=EdgeType.MASTERED if new_mastery >= 0.75 else EdgeType.LEARNING,
            )

    # ------------------------------------------------------------- top-level entry point

    def generate(self) -> SimulationResult:
        events: list[RecommendationEvent] = []
        outcomes: dict[tuple[str, str], RecommendationOutcome] = {}
        event_counter = 0

        for user_index in range(self.config.num_users):
            persona = self._pick_persona()
            user_id = f"synthetic_{persona.name}_{user_index}"

            # Each user's own timeline starts at an independently-sampled
            # anchor, not a single clock shared/accumulated across every
            # user in the run. A shared, never-reset `now` would keep
            # advancing by every user's session gaps for the ENTIRE run --
            # at production scale (thousands of users) that drifts `now`
            # centuries past real wall-clock time. Real UserStateBuilder
            # code (pipeline/recommender/models/user_state.py's
            # _concept_vector_weight/_problem_edge_weight) computes recency
            # decay from `time.time() - last_attempted` -- with `last_attempted`
            # centuries in the "future" relative to real wall-clock time,
            # that decay exponent overflows (`math.exp` raises OverflowError:
            # math range error), silently degrading every later user's state
            # vector (caught by recommend.py's own try/except, but a real,
            # avoidable quality loss for a "highest-quality dataset" goal).
            # Bounding each user's own anchor within a realistic recent
            # window keeps per-user drift small (session count x gap-days
            # is inherently bounded by sessions_range/session_gap_days_range)
            # regardless of how many users are generated, and also makes
            # user cohorts feel organic -- not everyone "signing up" at the
            # exact same instant.
            now = SIMULATION_EPOCH_START + self._rng.uniform(0.0, 2 * 365.0) * SECONDS_PER_DAY

            graph = self._build_initial_graph(user_id, persona, now)
            n_sessions = self._rng.randint(*persona.sessions_range)

            for session_index in range(n_sessions):
                if session_index > 0:
                    gap_days = self._rng.uniform(*persona.session_gap_days_range)
                    now += gap_days * SECONDS_PER_DAY

                # Ages every touched concept's urgency by real elapsed time
                # since it was last reviewed -- must happen before the plan
                # is built, so pool weighting and SpacedReviewPool see
                # genuinely-current forgetting state, not stale seed values.
                self._refresh_temporal_state(graph, now)

                plan = AdaptiveDifficultyController(now=now).build_plan(graph)
                state = _build_state(graph, self.qdrant)
                orchestrator = PoolGenerationOrchestrator(qdrant=self.qdrant, collection=self.config.collection)
                gen_result = orchestrator.generate(graph, state, total_n=self.config.total_n)
                final_candidates = self._finalize_candidates(graph, gen_result)

                if not final_candidates:
                    continue

                event_counter += 1
                recommended_at = now

                # Snapshot the graph as it existed AT this event -- graph
                # keeps mutating for later sessions of this same user, and
                # a live reference here would leak later sessions' evolved
                # mastery backward into this (earlier) event's features.
                graph_snapshot = UserGraph.from_dict(graph.to_dict())

                events.append(RecommendationEvent(
                    query_id=f"synthetic_q_{event_counter}",
                    user_id=user_id, recommended_at=recommended_at,
                    graph=graph_snapshot, plan=plan, candidates=final_candidates,
                ))

                for candidate in final_candidates:
                    outcome = self._simulate_interaction(persona, candidate)
                    attempt_at = recommended_at
                    if outcome.was_attempted:
                        attempt_at = recommended_at + DEFAULT_ATTEMPT_LATENCY_SECONDS
                        outcome = RecommendationOutcome(
                            was_attempted=True, attempt_count=outcome.attempt_count,
                            first_attempt_at=attempt_at,
                            solved=outcome.solved,
                            solved_at=attempt_at if outcome.solved else None,
                        )
                    outcomes[(user_id, candidate.problem_id)] = outcome
                    self._apply_outcome_to_graph(graph, candidate, outcome, attempt_at)
                    # InMemoryOutcomeProvider (label_generator.py, not
                    # modified here) is keyed only by (user_id, problem_id),
                    # with no recommended_at dimension: if the same problem
                    # were ever shown again to this same synthetic user in a
                    # later session, this dict could only hold one outcome
                    # for it, and whichever session's row looked it up second
                    # would get an outcome timestamped relative to the OTHER
                    # session's recommended_at -- tripping LabelGenerator's
                    # "impossible attribution" check. Rather than work around
                    # that lookup limitation, prevent the collision from
                    # arising at all: once a candidate has been shown to a
                    # user (whatever the outcome), add it to deprioritised_ids
                    # so the real per-pool filter (_filter_pool, already
                    # checking this set) naturally excludes it from every
                    # later session's pools for this user. "Retry later" is
                    # still modelled -- just as multiple submission attempts
                    # (attempt_count) against the one event it was shown in,
                    # which mirrors how real submission rows accumulate
                    # against a single recommendation_log row over time.
                    graph.deprioritised_ids.add(candidate.problem_id)

        return SimulationResult(events=events, outcome_provider=InMemoryOutcomeProvider(outcomes))

"""
Adaptive difficulty controller.

Sits between pool generation and candidate filtering. It does NOT generate
questions. It reads the user's current state (BKT mastery, HLR urgency and
half-life, confidence, SM-2 review timing, concept-gap severity — all already
carried on the UserGraph built from Shraddha's stores and Postgres) and
produces, for every pool:

    - a weight  : how much this pool should contribute to the final slate
    - a mix     : this pool's own easy / medium / hard percentage split

The pool generation layer (built separately) uses the weight to decide how
many candidates to draw from each pool, and the mix to decide the difficulty
spread within that pool's draw.

No ELO is used — difficulty preference is derived entirely from the user's
average BKT mastery and how many concepts are weak / urgent / overdue.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from pipeline.recommender.models.user_graph import UserGraph


# Pools this controller assigns weights to. Matches the pool generation layer.
POOLS = ["A", "B_C", "D", "E", "F", "G", "vector"]

# A user is treated as "beginner-leaning" below this average mastery,
# "advanced-leaning" above the upper bound. Between them is the mid band.
LOW_MASTERY  = 0.35
HIGH_MASTERY = 0.65

# When the user has many overdue / urgent concepts, review-type pools
# (E spaced-repetition, D weakness) get boosted relative to growth pools.
URGENCY_BOOST_THRESHOLD = 0.6
SEVERITY_WEAK_THRESHOLD = 0.6

# Difficulty mix presets by user level. Each is (easy, medium, hard) and sums
# to 1.0. The controller interpolates and adjusts these per pool.
MIX_ADVANCED = (0.15, 0.40, 0.45)
MIX_BEGINNER = (0.60, 0.30, 0.10)

# A user's very first-ever recommendations (avg_mastery == 0.0 -- either a
# genuinely cold-start user, or a topic whose mastery hasn't started
# accumulating yet) get an EXTRA gentle mix, not just MIX_BEGINNER.
# MIX_BEGINNER's 30% medium / 10% hard is too much to lead with when there
# is zero evidence the user can handle anything past "easy" yet. See
# _base_mix()'s three-segment interpolation for how this eases into
# MIX_BEGINNER as avg_mastery climbs, rather than a hard cutover.
MIX_COLD_START = (0.85, 0.15, 0.0)


@dataclass
class PoolDirective:
    """The controller's instruction for a single pool."""
    pool:   str
    weight: float                       # normalised, all pools sum to 1.0
    mix:    dict                         # {"easy": .., "medium": .., "hard": ..}


@dataclass
class DifficultyPlan:
    """Full output: one directive per pool, plus the signals it was built from."""
    directives:   dict = field(default_factory=dict)   # pool -> PoolDirective
    avg_mastery:  float = 0.0
    level:        str   = "mid"                         # beginner / mid / advanced
    n_weak:       int   = 0
    n_urgent:     int   = 0
    n_overdue:    int   = 0
    is_cold_start: bool = False

    def weight_of(self, pool: str) -> float:
        d = self.directives.get(pool)
        return d.weight if d else 0.0

    def mix_of(self, pool: str) -> dict:
        d = self.directives.get(pool)
        return d.mix if d else {"easy": 0.34, "medium": 0.33, "hard": 0.33}

    def to_dict(self) -> dict:
        return {
            "level":         self.level,
            "avg_mastery":   round(self.avg_mastery, 4),
            "n_weak":        self.n_weak,
            "n_urgent":      self.n_urgent,
            "n_overdue":     self.n_overdue,
            "is_cold_start": self.is_cold_start,
            "pools": {
                p: {"weight": round(d.weight, 4), "mix": d.mix}
                for p, d in self.directives.items()
            },
        }


class AdaptiveDifficultyController:
    """Turns a UserGraph into a per-pool difficulty plan."""

    def __init__(self, now: float | None = None):
        self._now = now if now is not None else time.time()

    # ------------------------------------------------------------------ public

    def build_plan(self, graph: UserGraph) -> DifficultyPlan:
        avg_mastery = self._avg_mastery(graph)
        n_weak      = len(graph.weak_concepts(SEVERITY_WEAK_THRESHOLD))
        n_urgent    = len(graph.urgent_concepts(URGENCY_BOOST_THRESHOLD))
        n_overdue   = self._count_overdue(graph)
        is_cold     = len(graph.concept_edges) == 0 or not graph.solved_ids

        level = self._level(avg_mastery)

        weights = self._pool_weights(
            level, n_weak, n_urgent, n_overdue, is_cold
        )
        base_mix = self._base_mix(avg_mastery)

        directives = {}
        for pool in POOLS:
            mix = self._pool_mix(pool, base_mix, level)
            directives[pool] = PoolDirective(
                pool=pool,
                weight=weights[pool],
                mix=mix,
            )

        return DifficultyPlan(
            directives=directives,
            avg_mastery=avg_mastery,
            level=level,
            n_weak=n_weak,
            n_urgent=n_urgent,
            n_overdue=n_overdue,
            is_cold_start=is_cold,
        )

    # ------------------------------------------------------------- signals

    def _avg_mastery(self, graph: UserGraph) -> float:
        edges = list(graph.concept_edges.values())
        if not edges:
            return 0.0
        return sum(e.mastery_score for e in edges) / len(edges)

    def _count_overdue(self, graph: UserGraph) -> int:
        """Concepts whose SM-2 next_review_date is in the past."""
        overdue = 0
        for e in graph.concept_edges.values():
            if not e.next_review_date:
                continue
            try:
                due = datetime.fromisoformat(e.next_review_date)
                if due.tzinfo is None:
                    due = due.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue
            if due.timestamp() <= self._now:
                overdue += 1
        return overdue

    def _level(self, avg_mastery: float) -> str:
        if avg_mastery < LOW_MASTERY:
            return "beginner"
        if avg_mastery > HIGH_MASTERY:
            return "advanced"
        return "mid"

    # ------------------------------------------------------------- weights

    def _pool_weights(self, level, n_weak, n_urgent, n_overdue, is_cold) -> dict:
        """
        Raw pool scores, then normalised to sum to 1.0.

        Cold start: lean on course path (A) and novelty (G) since there is
        little behavioural signal to target weakness or review.
        Otherwise: base weights by level, then boost review/weakness pools
        when the user has overdue or weak concepts, and boost growth pools
        when the user is strong and has little to review.
        """
        if is_cold:
            raw = {
                "A": 0.35, "B_C": 0.15, "D": 0.05, "E": 0.05,
                "F": 0.10, "G": 0.20, "vector": 0.05,
            }
            return self._normalise(raw)

        # base weights per level
        if level == "beginner":
            raw = {"A": 0.25, "B_C": 0.20, "D": 0.20, "E": 0.10,
                   "F": 0.05, "G": 0.10, "vector": 0.10}
        elif level == "advanced":
            raw = {"A": 0.05, "B_C": 0.15, "D": 0.10, "E": 0.10,
                   "F": 0.25, "G": 0.15, "vector": 0.20}
        else:  # mid
            raw = {"A": 0.15, "B_C": 0.20, "D": 0.15, "E": 0.10,
                   "F": 0.15, "G": 0.10, "vector": 0.15}

        # review pressure: overdue reviews and urgent (forgetting) concepts
        # push E (spaced review) up
        if n_overdue > 0 or n_urgent > 0:
            pressure = min(0.20, 0.03 * (n_overdue + n_urgent))
            raw["E"] += pressure

        # weakness pressure: many weak concepts push D (weakness recovery) up
        if n_weak > 0:
            raw["D"] += min(0.15, 0.03 * n_weak)

        return self._normalise(raw)

    def _normalise(self, raw: dict) -> dict:
        total = sum(raw.values())
        if total <= 0:
            n = len(raw)
            return {k: 1.0 / n for k in raw}
        return {k: v / total for k, v in raw.items()}

    # ------------------------------------------------------------- mixes

    def _base_mix(self, avg_mastery: float) -> tuple:
        """
        Interpolate a global easy/med/hard mix from average mastery, in
        THREE continuous segments: MIX_COLD_START (avg_mastery=0) ->
        MIX_BEGINNER (avg_mastery=LOW_MASTERY) -> MIX_ADVANCED
        (avg_mastery=HIGH_MASTERY).

        A flat plateau at MIX_BEGINNER across the ENTIRE 0..LOW_MASTERY
        range (the old behavior) meant a user's 2nd-3rd recommendation --
        right after cold start ends, avg_mastery still very low (e.g.
        ~0.2 after one easy solve) -- got the IDENTICAL 30% medium / 10%
        hard exposure as an almost-established beginner at avg_mastery
        0.34. That reads as a sudden jump to medium, not a gradual ramp.
        Interpolating continuously from MIX_COLD_START removes that cliff:
        the easy share now eases down smoothly question by question as
        avg_mastery climbs, instead of jumping the moment is_cold_start
        flips False.
        """
        if avg_mastery <= 0.0:
            return MIX_COLD_START
        if avg_mastery < LOW_MASTERY:
            t = avg_mastery / LOW_MASTERY
            lo, hi = MIX_COLD_START, MIX_BEGINNER
            return tuple(lo[i] + (hi[i] - lo[i]) * t for i in range(3))
        if avg_mastery > HIGH_MASTERY:
            return MIX_ADVANCED
        # linear interpolate between beginner/advanced endpoints across the mid band
        span = HIGH_MASTERY - LOW_MASTERY
        t = (avg_mastery - LOW_MASTERY) / span if span > 0 else 0.5
        lo, hi = MIX_BEGINNER, MIX_ADVANCED
        return tuple(lo[i] + (hi[i] - lo[i]) * t for i in range(3))

    def _pool_mix(self, pool: str, base_mix: tuple, level: str) -> dict:
        """
        Adjust the base mix per pool. Each pool has a natural difficulty lean:
          A  course path  -> follows base
          B_C near/far    -> slightly easier (reinforcement)
          D  weakness     -> easier (rebuild confidence)
          E  spaced review-> follows base (review at learned level)
          F  stretch      -> harder (growth)
          G  novelty      -> easier (new concepts introduced gently)
          vector          -> follows base
        """
        easy, med, hard = base_mix

        if pool in ("D", "G"):
            # shift toward easy
            easy, med, hard = self._shift(easy, med, hard, toward="easy")
        elif pool == "B_C":
            easy, med, hard = self._shift(easy, med, hard, toward="easy", amount=0.05)
        elif pool == "F":
            # shift toward hard
            easy, med, hard = self._shift(easy, med, hard, toward="hard")

        total = easy + med + hard
        return {
            "easy":   round(easy / total, 4),
            "medium": round(med  / total, 4),
            "hard":   round(hard / total, 4),
        }

    def _shift(self, easy, med, hard, toward, amount=0.10):
        """Move probability mass toward easy or hard, clamped at 0."""
        if toward == "easy":
            hard = max(0.0, hard - amount)
            easy = easy + amount
        elif toward == "hard":
            easy = max(0.0, easy - amount)
            hard = hard + amount
        return easy, med, hard


def build_difficulty_plan(graph: UserGraph, now: float | None = None) -> DifficultyPlan:
    """Convenience wrapper."""
    return AdaptiveDifficultyController(now=now).build_plan(graph)

"""
Candidate storage layer.

The last piece of your message: "the questions will be stored somewhere
(may have to request a table from backend) where some appropriate ranker
will be applied."

This module defines:
  1. CandidateStore -- an abstract interface the ranker reads from and the
     pool generation orchestrator writes to, so swapping the backing store
     (Postgres table, Redis, in-memory) never touches calling code.
  2. InMemoryCandidateStore -- a working reference implementation, usable
     immediately without waiting on any backend schema work.
  3. PROPOSED_TABLE_SCHEMA -- the Postgres table to request from the backend
     team, matching the existing KNode schema's naming conventions
     (RecommendationLog already exists for POST-recommendation tracking;
     this is the PRE-recommendation staging table candidates sit in between
     filtering and ranking).

Why a staging table at all, not just an in-memory hand-off:
  - the ranker may run as a separate process/service, or on a delay
  - lets you inspect/debug exactly what candidates a user was shown before
    ranking touched them, independent of RecommendationLog (which only
    records what was ACTUALLY recommended after ranking + diversity mixing)
  - gives a natural point to expire stale candidate sets (a user's state
    changes after their next submission, so a candidate set older than a
    few minutes should not be ranked against fresh state)
"""

from __future__ import annotations

import time
import json
import uuid
from dataclasses import dataclass, field
from typing import Optional

from pipeline.recommender.services.candidate_filtering import MergedCandidate


# How long a staged candidate set remains valid before it should be
# regenerated rather than ranked. Matches the Redis user_graph cache TTL
# (300s) -- if the graph itself might be stale after this long, a candidate
# set built from it shouldn't be trusted for ranking after this long either.
CANDIDATE_SET_TTL_SECONDS = 300


@dataclass
class StoredCandidateSet:
    """One staged batch of filtered candidates, ready for ranking."""
    set_id:        str
    user_id:       str
    created_at:    float
    candidates:    list           # list[dict] -- CandidateFilteringLayer.to_ranker_input() output
    difficulty_plan: dict         # DifficultyPlan.to_dict() snapshot, for ranker context
    expires_at:    float = field(init=False)

    def __post_init__(self):
        self.expires_at = self.created_at + CANDIDATE_SET_TTL_SECONDS

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    def to_dict(self) -> dict:
        return {
            "set_id":          self.set_id,
            "user_id":         self.user_id,
            "created_at":      self.created_at,
            "expires_at":      self.expires_at,
            "candidates":      self.candidates,
            "difficulty_plan": self.difficulty_plan,
        }


class CandidateStore:
    """
    Abstract interface. Implement this against whatever the backend team
    provisions (Postgres table below, Redis, etc.) -- the orchestrator and
    ranker only ever call these three methods, never touch the storage
    medium directly.
    """

    def save(self, user_id: str, candidates: list, difficulty_plan: dict) -> StoredCandidateSet:
        raise NotImplementedError

    def get_latest(self, user_id: str) -> Optional[StoredCandidateSet]:
        raise NotImplementedError

    def get(self, set_id: str) -> Optional[StoredCandidateSet]:
        raise NotImplementedError


class InMemoryCandidateStore(CandidateStore):
    """
    Working reference implementation -- usable today without any backend
    schema work. Swap for a Postgres-backed store later; callers don't change.
    Keeps only the latest set per user (older sets are superseded, not kept).
    """

    def __init__(self):
        self._by_user: dict = {}     # user_id -> StoredCandidateSet
        self._by_id: dict = {}       # set_id -> StoredCandidateSet

    def save(self, user_id: str, candidates: list, difficulty_plan: dict) -> StoredCandidateSet:
        cs = StoredCandidateSet(
            set_id=str(uuid.uuid4()),
            user_id=user_id,
            created_at=time.time(),
            candidates=candidates,
            difficulty_plan=difficulty_plan,
        )
        self._by_user[user_id] = cs
        self._by_id[cs.set_id] = cs
        return cs

    def get_latest(self, user_id: str) -> Optional[StoredCandidateSet]:
        cs = self._by_user.get(user_id)
        if cs is None or cs.is_expired:
            return None
        return cs

    def get(self, set_id: str) -> Optional[StoredCandidateSet]:
        cs = self._by_id.get(set_id)
        if cs is None or cs.is_expired:
            return None
        return cs


def stage_candidates(store: CandidateStore, user_id: str,
                     merged_candidates: list, difficulty_plan_dict: dict,
                     graph=None) -> StoredCandidateSet:
    """
    Convenience wrapper: flatten MergedCandidate objects via
    CandidateFilteringLayer.to_ranker_input()-shaped dicts (if not already
    flattened) and stage them.
    """
    if merged_candidates and isinstance(merged_candidates[0], MergedCandidate):
        if graph is None:
            raise ValueError(
                "graph is required to flatten MergedCandidate objects via "
                "to_ranker_input(); pass already-flattened dicts instead if "
                "graph is unavailable.")
        from pipeline.recommender.services.candidate_filtering import CandidateFilteringLayer
        rows = CandidateFilteringLayer(graph).to_ranker_input(merged_candidates)
    else:
        rows = merged_candidates
    return store.save(user_id, rows, difficulty_plan_dict)


# ---------------------------------------------------------------------------
# Proposed backend table -- request this from the backend team.
# Matches KNode's existing naming convention (RecommendationLog already
# exists for post-recommendation tracking; this is the pre-ranking staging
# table candidates sit in between filtering and ranking).
# ---------------------------------------------------------------------------

PROPOSED_TABLE_SCHEMA = """
-- CandidateStaging: filtered candidates awaiting ranking.
-- One row per (user, candidate set) batch -- NOT one row per candidate,
-- since the whole batch is ranked together and candidates within it are
-- meaningless without the sibling candidates they were diversified against.

CREATE TABLE "CandidateStaging" (
    set_id            TEXT PRIMARY KEY,             -- CUID, matches other KNode PKs
    user_id           TEXT NOT NULL REFERENCES "User"(user_id),
    candidates        JSONB NOT NULL,                -- list of ranker-input rows
                                                       -- (problem_id, pool_sources,
                                                       --  pool_count, topic_tags,
                                                       --  difficulty_score, avg_mastery,
                                                       --  max_urgency, predicted_success,
                                                       --  best_pool_score)
    difficulty_plan   JSONB NOT NULL,                 -- DifficultyPlan.to_dict() snapshot
    created_at        TIMESTAMP NOT NULL DEFAULT now(),
    expires_at        TIMESTAMP NOT NULL,             -- created_at + 300s, matches Redis TTL
    ranked_at         TIMESTAMP,                      -- NULL until the ranker processes it
    ranker_version    TEXT                            -- which ranker/model version scored it
);

CREATE INDEX idx_candidate_staging_user_latest
    ON "CandidateStaging" (user_id, created_at DESC);

-- Cleanup: a nightly job (or ON CONFLICT upsert keeping only the latest per
-- user) should delete expired rows -- these are meaningless once the user's
-- state has moved on, same as the Redis user_graph cache.
"""

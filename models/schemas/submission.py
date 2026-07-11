"""
Submission schema.

Shraddha's architecture is stateless: the backend sends the user's
current mastery/HLR state IN the request body (problemTopics), ML
computes the updates, returns the new state. ML never reads from or
writes to Postgres -- backend owns all persistence.

Field names match exactly what submission_controller.py reads:
    submission.problemTopics  -> list of TopicState
    submission.userId         -> str
    submission.problemId      -> str
    submission.verdict        -> str -- must be exactly "OK" for a
                                  full-credit solve (confirmed from bkt.py)
"""

from __future__ import annotations

import time
from typing import Optional, List

from pydantic import BaseModel, Field


class TopicState(BaseModel):
    """Current per-topic state sent by backend for this problem's topics."""
    topicId: str
    currentMastery: Optional[float] = None   # current BKT P(L), None if no prior data
    currentHlr: Optional[dict] = None        # current HLR state dict, None if no prior data


class Submission(BaseModel):
    userId: str
    problemId: str
    verdict: str          # must be exactly "OK" for a solved submission
                          # -- bkt.py checks `verdict == "OK"` literally

    hintsUsed: int = 0
    testCasesPassed: int = 0
    totalTestCases: int = 1
    submissionCount: int = 1
    normalisedScore: float = 0.0
    timestamp: float = Field(default_factory=lambda: time.time())

    # Current mastery/HLR state per topic -- sent by backend, not fetched
    # by ML. Backend knows which topics this problem covers and sends the
    # user's current state for each. ML returns updatedTopics with new values.
    problemTopics: List[TopicState] = []

    class Config:
        json_schema_extra = {
            "example": {
                "userId": "user_abc123",
                "problemId": "two-sum",
                "verdict": "OK",
                "hintsUsed": 0,
                "testCasesPassed": 10,
                "totalTestCases": 10,
                "submissionCount": 1,
                "normalisedScore": 0.95,
                "timestamp": 1752345600.0,
                "problemTopics": [
                    {
                        "topicId": "array",
                        "currentMastery": 0.45,
                        "currentHlr": {
                            "half_life": 5.0,
                            "last_review": "2026-07-01T00:00:00+00:00",
                            "p_recall": 0.8,
                            "next_review_days": 3.5
                        }
                    },
                    {
                        "topicId": "hash_map",
                        "currentMastery": None,
                        "currentHlr": None
                    }
                ]
            }
        }
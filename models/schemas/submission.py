from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional


class TopicState(BaseModel):
    topicId: str
    currentMastery: Optional[float] = None
    currentHlr: Optional[Dict[str, Any]] = None


class Submission(BaseModel):
    userId: str
    problemId: str
    verdict: str
    testCasesPassed: int
    totalTestCases: int
    hintsUsed: int
    submissionCount: int
    normalisedScore: float
    timestamp: float

    # Backend-supplied current state, one entry per topic on this problem.
    # ML is stateless -- it never reads mastery/HLR from the database.
    problemTopics: List[TopicState] = Field(default_factory=list)
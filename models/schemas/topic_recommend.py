"""
Topic-based recommendation request schema.

Backend already knows which topic it wants problems for (e.g. user clicked
"practice arrays" in a topic-picker UI) -- this is the "backend's demand"
input the /topic/recommend/problems endpoint reads, separate from the
mixed-pool /recommend/{user_id} endpoint and the single-topic-suggestion
/topic/recommend/{user_id} endpoint.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TopicRecommendRequest(BaseModel):
    userId: str
    topicId: str
    limit: int = Field(10, ge=1, le=50)

    class Config:
        json_schema_extra = {
            "example": {
                "userId": "user_abc123",
                "topicId": "array",
                "limit": 10,
            }
        }

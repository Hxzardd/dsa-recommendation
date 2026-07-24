from fastapi import APIRouter, HTTPException, Request

from controllers.recommendation_controller import (
    handle_recommend, handle_topic_recommend, handle_topic_problem_recommend,
)
from models.schemas.topic_recommend import TopicRecommendRequest

MAX_RECOMMENDATIONS = 50
MIN_RECOMMENDATIONS = 1

router = APIRouter()

# NOTE: every handler below is deliberately `def`, NOT `async def`.
# handle_* run the recommendation pipeline synchronously -- pooled psycopg2
# queries plus blocking Qdrant HTTP calls. An `async def` handler executes
# directly on the event loop, so ONE /recommend call would stall every other
# in-flight request (including the health probe) for the whole duration of
# that pipeline run. Declaring them sync makes FastAPI dispatch each call to
# its thread pool instead, so concurrent requests actually proceed in
# parallel. Do not add `async` here without first making the pipeline
# genuinely non-blocking.


@router.get("/recommend/{user_id}")
def recommend(
    user_id: str,
    request: Request,
    limit: int = 10,
):
    auth_user_id = request.state.user_id

    if auth_user_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="Forbidden",
        )

    limit = max(MIN_RECOMMENDATIONS, min(limit, MAX_RECOMMENDATIONS))
    return handle_recommend(user_id, limit)


@router.get("/topic/recommend/{user_id}")
def topic_recommend(
    user_id: str,
    request: Request,
):
    auth_user_id = request.state.user_id

    if auth_user_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="Forbidden",
        )

    return handle_topic_recommend(user_id)


@router.post("/topic/recommend/problems")
def topic_problem_recommend(
    body: TopicRecommendRequest,
    request: Request,
):
    auth_user_id = request.state.user_id

    if auth_user_id != body.userId:
        raise HTTPException(
            status_code=403,
            detail="Forbidden",
        )

    return handle_topic_problem_recommend(body.userId, body.topicId, body.limit)
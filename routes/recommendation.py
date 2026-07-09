from fastapi import APIRouter
from controllers.recommendation_controller import handle_recommend

MAX_RECOMMENDATIONS = 50
MIN_RECOMMENDATIONS = 1

router = APIRouter()


@router.get("/recommend/{user_id}")
async def recommend(user_id: str, limit: int = 10):
    limit = max(MIN_RECOMMENDATIONS, min(limit, MAX_RECOMMENDATIONS))
    return handle_recommend(user_id, limit)
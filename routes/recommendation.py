from fastapi import APIRouter, Depends, HTTPException
from controllers.recommendation_controller import handle_recommend
from middlewares.auth import verify_session

MAX_RECOMMENDATIONS = 50
MIN_RECOMMENDATIONS = 1

router = APIRouter()


@router.get("/recommend/{user_id}")
async def recommend(
    user_id: str,
    limit: int = 10,
    auth_user_id: str = Depends(verify_session),
):
    if auth_user_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="Forbidden"
        )

    limit = max(MIN_RECOMMENDATIONS, min(limit, MAX_RECOMMENDATIONS))
    return handle_recommend(user_id, limit)
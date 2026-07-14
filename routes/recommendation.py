from fastapi import APIRouter, HTTPException, Request

from controllers.recommendation_controller import handle_recommend

MAX_RECOMMENDATIONS = 50
MIN_RECOMMENDATIONS = 1

router = APIRouter()


@router.get("/recommend/{user_id}")
async def recommend(
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
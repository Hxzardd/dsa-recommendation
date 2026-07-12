from fastapi import APIRouter, Depends, HTTPException
from controllers.mastery_controller import handle_get_mastery, handle_get_urgency
from middlewares.auth import verify_session

router = APIRouter()


@router.get("/mastery/{user_id}")
def get_mastery(
    user_id: str,
    auth_user_id: str = Depends(verify_session),
):
    if auth_user_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="Forbidden"
        )

    return handle_get_mastery(user_id)


@router.get("/urgency/{user_id}")
def get_urgency(
    user_id: str,
    auth_user_id: str = Depends(verify_session),
):
    if auth_user_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="Forbidden"
        )

    return handle_get_urgency(user_id)
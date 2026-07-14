from fastapi import APIRouter, HTTPException, Request

from controllers.mastery_controller import (
    handle_get_mastery,
    handle_get_urgency,
)

router = APIRouter()


@router.get("/mastery/{user_id}")
def get_mastery(
    user_id: str,
    request: Request,
):
    auth_user_id = request.state.user_id

    if auth_user_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="Forbidden",
        )

    return handle_get_mastery(user_id)


@router.get("/urgency/{user_id}")
def get_urgency(
    user_id: str,
    request: Request,
):
    auth_user_id = request.state.user_id

    if auth_user_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="Forbidden",
        )

    return handle_get_urgency(user_id)
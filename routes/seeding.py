from fastapi import APIRouter, HTTPException, Request

from controllers.seeding_controller import (
    handle_seed_bkt,
    handle_seed_hlr,
)

router = APIRouter()


@router.post("/seed_hlr/{user_id}")
def seed_hlr(
    user_id: str,
    request: Request,
):
    auth_user_id = request.state.user_id

    if auth_user_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="Forbidden",
        )

    return handle_seed_hlr(user_id)


@router.post("/seed_bkt/{user_id}")
def seed_bkt(
    user_id: str,
    request: Request,
):
    auth_user_id = request.state.user_id

    if auth_user_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="Forbidden",
        )

    return handle_seed_bkt(user_id)
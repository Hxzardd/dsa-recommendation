from fastapi import APIRouter, Depends, HTTPException
from controllers.seeding_controller import handle_seed_hlr, handle_seed_bkt
from middlewares.auth import verify_session

router = APIRouter()


@router.post("/seed_hlr/{user_id}")
def seed_hlr(
    user_id: str,
    auth_user_id: str = Depends(verify_session),
):
    if auth_user_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="Forbidden",
        )

    return handle_seed_hlr(user_id)


@router.post("/seed_bkt/{user_id}")
def seed_bkt(
    user_id: str,
    auth_user_id: str = Depends(verify_session),
):
    if auth_user_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="Forbidden",
        )

    return handle_seed_bkt(user_id)
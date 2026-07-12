from fastapi import APIRouter
from controllers.seeding_controller import handle_seed_hlr
from controllers.seeding_controller import handle_seed_hlr, handle_seed_bkt

router = APIRouter()

@router.post("/seed_hlr/{user_id}")
def seed_hlr(user_id: str):
    return handle_seed_hlr(user_id)

@router.post("/seed_bkt/{user_id}")
def seed_bkt(user_id: str):
    return handle_seed_bkt(user_id)
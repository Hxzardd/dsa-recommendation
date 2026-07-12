from fastapi import APIRouter, Depends, HTTPException
from models.schemas.submission import Submission
from controllers.submission_controller import handle_update
from middlewares.auth import verify_session

router = APIRouter()


@router.post("/update")
def update_endpoint(
    submission: Submission,
    auth_user_id: str = Depends(verify_session),
):
    if auth_user_id != submission.userId:
        raise HTTPException(
            status_code=403,
            detail="Forbidden",
        )

    return handle_update(submission)
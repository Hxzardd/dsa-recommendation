from fastapi import APIRouter, HTTPException, Request
from models.schemas.submission import Submission
from controllers.submission_controller import handle_update


router = APIRouter()

@router.post("/update")
def update_endpoint(
    submission: Submission,
    request: Request,
):
    auth_user_id = request.state.user_id

    if auth_user_id != submission.userId:
        raise HTTPException(
            status_code=403,
            detail="Forbidden",
        )

    return handle_update(submission) 
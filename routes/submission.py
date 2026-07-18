from fastapi import APIRouter, HTTPException, Request
from models.schemas.submission import Submission
from controllers.submission_controller import handle_update


router = APIRouter()

@router.post("/update")
def update_endpoint(
    submission: Submission,
    request: Request,
):
    # Service calls (backend's judge0 submission webhook -- see
    # middlewares/auth.py's ML_SERVICE_TOKEN) have no end-user session, so
    # there's no auth_user_id to compare submission.userId against -- the
    # service acts on whatever userId the request body names, on the
    # backend's own authority. A real per-user token still can't update
    # someone else's mastery.
    if not getattr(request.state, "is_service_call", False):
        auth_user_id = request.state.user_id

        if auth_user_id != submission.userId:
            raise HTTPException(
                status_code=403,
                detail="Forbidden",
            )

    return handle_update(submission)
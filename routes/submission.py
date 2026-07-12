from fastapi import APIRouter
from models.schemas.submission import Submission
from controllers.submission_controller import handle_update

router = APIRouter()


@router.post("/update")
def update_endpoint(submission: Submission):
    return handle_update(submission)
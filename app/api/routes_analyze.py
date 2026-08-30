"""POST /analyze — code-analysis feedback for a completed submission."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.api.deps import require_service_auth
from app.logging.logger import bind_submission_id, reset_submission_id
from app.models.request_schemas import AnalyzeRequest
from app.models.response_schemas import AnalyzeResponse, ErrorResponse
from app.orchestrator.analyze import analyze_submission
from app.security.sanitizer import UnsafeInputError

router = APIRouter(tags=["analyze"])


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    dependencies=[Depends(require_service_auth)],
)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse | JSONResponse:
    """Analyze a submission and return mentor-style feedback.

    Unsafe input → 400. LLM outages degrade to a valid response (never 5xx).
    """

    token = bind_submission_id(request.submission_id)
    try:
        return await analyze_submission(request)
    except UnsafeInputError as exc:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error_code="unsafe_input",
                message=str(exc),
                submission_id=request.submission_id,
            ).model_dump(),
        )
    finally:
        reset_submission_id(token)

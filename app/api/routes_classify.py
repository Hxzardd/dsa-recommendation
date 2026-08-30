"""POST /classify-approach — which topics/patterns a submission actually used."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import require_service_auth
from app.logging.logger import bind_submission_id, reset_submission_id
from app.models.request_schemas import ClassifyApproachRequest
from app.models.response_schemas import ClassifyApproachResponse
from app.orchestrator.classify import classify_approach

router = APIRouter(tags=["classify"])


@router.post(
    "/classify-approach",
    response_model=ClassifyApproachResponse,
    dependencies=[Depends(require_service_auth)],
)
async def classify(request: ClassifyApproachRequest) -> ClassifyApproachResponse:
    """Classify the submission's approach for approach-aware mastery gating.

    LLM outages degrade to confidence 0 (backend stays weights-only), never 5xx.
    """

    token = bind_submission_id(request.submission_id)
    try:
        return await classify_approach(request)
    finally:
        reset_submission_id(token)

"""HTTP exception handlers."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.logging.logger import get_logger, submission_id_var
from app.models.response_schemas import ErrorResponse

logger = get_logger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Register application-wide exception handlers."""

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        """Return a stable validation error envelope."""

        logger.info(
            "request validation failed",
            extra={"path": request.url.path, "errors": exc.errors()},
        )
        body = ErrorResponse(
            error_code="validation_error",
            message="Request validation failed.",
            submission_id=submission_id_var.get(),
        )
        return JSONResponse(status_code=422, content=body.model_dump())

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Return a generic error envelope while logging details internally."""

        logger.exception("unhandled application error", extra={"path": request.url.path})
        body = ErrorResponse(
            error_code="internal_error",
            message="An internal error occurred.",
            submission_id=submission_id_var.get(),
        )
        return JSONResponse(status_code=500, content=body.model_dump())


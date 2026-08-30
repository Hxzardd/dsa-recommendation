"""FastAPI application entrypoint."""

from fastapi import FastAPI

from app.api.routes_analyze import router as analyze_router
from app.api.routes_classify import router as classify_router
from app.config.settings import get_settings
from app.logging.logger import configure_logging
from app.middleware.error_handler import register_exception_handlers
from app.middleware.logging_middleware import RequestLoggingMiddleware

settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(title="KNode AI Code Analysis", version="0.1.0")
app.add_middleware(RequestLoggingMiddleware)
register_exception_handlers(app)
app.include_router(analyze_router)
app.include_router(classify_router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Return service health."""

    return {"status": "ok"}

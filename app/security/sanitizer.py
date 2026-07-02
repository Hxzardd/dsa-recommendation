"""Input and output safety helpers."""

from app.config.settings import get_settings
from app.models.request_schemas import AnalyzeRequest


class UnsafeInputError(ValueError):
    """Raised when a request contains unsafe prompt-breaking input."""


def validate_input_safety(request: AnalyzeRequest) -> None:
    """Validate request input for basic prompt safety concerns."""

    settings = get_settings()
    if len(request.source_code) > settings.max_source_code_chars:
        msg = "source_code exceeds configured maximum length"
        raise UnsafeInputError(msg)

    text_fields = {
        "source_code": request.source_code,
        "stdin": request.stdin,
        "expected_output": request.expected_output,
        "actual_output": request.actual_output,
        "stdout": request.stdout,
        "stderr": request.stderr,
        "compile_output": request.compile_output,
    }
    for field_name, value in text_fields.items():
        if "\x00" in value:
            msg = f"{field_name} contains a null byte"
            raise UnsafeInputError(msg)


def scrub_llm_output(text: str, source_code: str) -> str:
    """Scrub LLM-derived text before returning it to the backend."""

    _ = source_code
    # TODO Phase 3: detect and withhold likely solution leaks.
    return text


"""Structured JSON logging helpers."""

from __future__ import annotations

import contextvars
import logging
import sys
from datetime import UTC, datetime
from typing import Any

submission_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "submission_id",
    default=None,
)

_STANDARD_LOG_RECORD_FIELDS = frozenset(
    logging.LogRecord(
        name="",
        level=0,
        pathname="",
        lineno=0,
        msg="",
        args=(),
        exc_info=None,
    ).__dict__,
)


def _merge_log_fields(
    *,
    submission_id: str | None,
    extra_fields: dict[str, Any] | None,
) -> dict[str, Any]:
    """Flatten structured fields while keeping submission_id authoritative."""

    merged_fields = dict(extra_fields or {})
    merged_fields["submission_id"] = submission_id
    return merged_fields


def _structured_record_fields(record: logging.LogRecord) -> dict[str, Any]:
    """Return fields added through logging's standard extra mechanism."""

    fields = {
        key: value
        for key, value in record.__dict__.items()
        if key not in _STANDARD_LOG_RECORD_FIELDS and key != "message"
    }

    legacy_extra = fields.pop("extra", None)
    if isinstance(legacy_extra, dict):
        fields.update(legacy_extra)

    fields["submission_id"] = (
        getattr(record, "submission_id", None) or submission_id_var.get()
    )
    return fields


class JsonFormatter(logging.Formatter):
    """Format log records as JSON-compatible single-line dictionaries."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON line."""

        import json

        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_structured_record_fields(record))

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


class ContextLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds submission context and structured extra fields."""

    def process(
        self,
        msg: str,
        kwargs: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Attach the current submission id and preserve structured extras."""

        extra = kwargs.pop("extra", {})
        if not isinstance(extra, dict):
            extra = {"extra_value": extra}

        kwargs["extra"] = _merge_log_fields(
            submission_id=submission_id_var.get(),
            extra_fields=extra,
        )
        return msg, kwargs


def bind_submission_id(submission_id: str | None) -> contextvars.Token[str | None]:
    """Bind a submission id to log lines emitted in the current context."""

    return submission_id_var.set(submission_id)


def reset_submission_id(token: contextvars.Token[str | None]) -> None:
    """Reset the submission id context to a previous token."""

    submission_id_var.reset(token)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once for JSON output."""

    root = logging.getLogger()
    root.setLevel(level.upper())

    if root.handlers:
        for handler in root.handlers:
            handler.setFormatter(JsonFormatter())
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)


def get_logger(name: str) -> ContextLoggerAdapter:
    """Return a structured logger adapter."""

    configure_logging()
    return ContextLoggerAdapter(logging.getLogger(name), {})

"""LLM client seam (vLLM only).

Small interface — ``complete(prompt) -> str`` — with two adapters: the real
``VLLMClient`` (httpx against an OpenAI-compatible ``/v1/chat/completions``) and
an in-memory ``FakeLLMClient`` for tests. The orchestrator depends on the
``LLMClient`` protocol and never constructs a concrete client itself, so tests
inject the fake and production injects vLLM.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import httpx

from app.config.settings import Settings, get_settings
from app.logging.logger import get_logger
from app.models.domain import LLMPrompt

logger = get_logger(__name__)


class LLMUnavailableError(RuntimeError):
    """Raised when the LLM cannot produce a completion (down, timeout, bad status)."""


class LLMClient(Protocol):
    """Anything that can turn a prompt into raw model text."""

    async def complete(self, prompt: LLMPrompt) -> str:
        """Return the model's raw text response for ``prompt``."""
        ...


class VLLMClient:
    """Calls a vLLM OpenAI-compatible chat completions endpoint."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def complete(self, prompt: LLMPrompt) -> str:
        """POST the prompt to vLLM and return the assistant message content."""

        settings = self._settings
        url = f"{settings.vllm_base_url.rstrip('/')}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if settings.vllm_api_key:
            headers["Authorization"] = f"Bearer {settings.vllm_api_key}"

        payload = {
            "model": settings.vllm_model,
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            # Low temperature: we want deterministic, schema-following analysis.
            "temperature": 0.1,
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError) as exc:  # timeouts, conn errors, bad JSON
            msg = f"vLLM request failed: {exc.__class__.__name__}: {exc}"
            logger.warning("llm unavailable", extra={"error": str(exc)})
            raise LLMUnavailableError(msg) from exc

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            msg = "vLLM response missing choices[0].message.content"
            raise LLMUnavailableError(msg) from exc


class FakeLLMClient:
    """Deterministic in-memory client for tests.

    ``response`` may be a fixed string or a callable taking the prompt. Set
    ``raises`` to simulate an outage.
    """

    def __init__(
        self,
        response: str | Callable[[LLMPrompt], str] = "",
        *,
        raises: Exception | None = None,
    ) -> None:
        self._response = response
        self._raises = raises
        self.calls: list[LLMPrompt] = []

    async def complete(self, prompt: LLMPrompt) -> str:
        """Record the call and return the canned response (or raise)."""

        self.calls.append(prompt)
        if self._raises is not None:
            raise self._raises
        if callable(self._response):
            return self._response(prompt)
        return self._response


def get_llm_client() -> LLMClient:
    """Return the production LLM client (vLLM)."""

    return VLLMClient(get_settings())

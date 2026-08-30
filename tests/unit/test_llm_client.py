"""LLM client tests (fake + vLLM adapter)."""

import asyncio

import httpx
import pytest

from app.config.settings import Settings
from app.llm import client as client_mod
from app.llm.client import FakeLLMClient, LLMUnavailableError, VLLMClient
from app.models.domain import LLMPrompt

PROMPT = LLMPrompt(system="sys", user="usr")


def test_fake_client_returns_and_records() -> None:
    fake = FakeLLMClient("hello")
    result = asyncio.run(fake.complete(PROMPT))
    assert result == "hello"
    assert fake.calls == [PROMPT]


def test_fake_client_raises_when_configured() -> None:
    fake = FakeLLMClient(raises=LLMUnavailableError("down"))
    with pytest.raises(LLMUnavailableError):
        asyncio.run(fake.complete(PROMPT))


class _FakeResponse:
    def __init__(self, data: dict) -> None:
        self._data = data

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._data


class _FakeAsyncClient:
    last_request: dict = {}

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        pass

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *args) -> bool:  # type: ignore[no-untyped-def]
        return False

    async def post(self, url, json, headers):  # type: ignore[no-untyped-def]
        _FakeAsyncClient.last_request = {"url": url, "json": json, "headers": headers}
        return _FakeResponse({"choices": [{"message": {"content": "MODEL OUTPUT"}}]})


def test_vllm_client_success_builds_openai_request(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(client_mod.httpx, "AsyncClient", _FakeAsyncClient)
    settings = Settings(_env_file=None, vllm_base_url="http://vllm:9000/", vllm_model="m1")

    result = asyncio.run(VLLMClient(settings).complete(PROMPT))

    assert result == "MODEL OUTPUT"
    req = _FakeAsyncClient.last_request
    assert req["url"] == "http://vllm:9000/v1/chat/completions"
    assert req["json"]["model"] == "m1"
    assert req["json"]["messages"][0] == {"role": "system", "content": "sys"}
    assert req["json"]["messages"][1] == {"role": "user", "content": "usr"}


def test_vllm_client_wraps_transport_errors(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class _BoomClient(_FakeAsyncClient):
        async def post(self, url, json, headers):  # type: ignore[no-untyped-def]
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(client_mod.httpx, "AsyncClient", _BoomClient)
    with pytest.raises(LLMUnavailableError):
        asyncio.run(VLLMClient(Settings(_env_file=None)).complete(PROMPT))


def test_vllm_client_raises_on_malformed_response(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class _EmptyClient(_FakeAsyncClient):
        async def post(self, url, json, headers):  # type: ignore[no-untyped-def]
            return _FakeResponse({"choices": []})

    monkeypatch.setattr(client_mod.httpx, "AsyncClient", _EmptyClient)
    with pytest.raises(LLMUnavailableError):
        asyncio.run(VLLMClient(Settings(_env_file=None)).complete(PROMPT))

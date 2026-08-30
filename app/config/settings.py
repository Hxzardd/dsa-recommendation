"""Application settings loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed runtime settings for the AI analysis service.

    LLM inference is vLLM-only (an OpenAI-compatible server). Code analysis is
    demanding, so we target a capable served code model rather than a small
    local model. When ``vllm_base_url`` is unreachable the orchestrator degrades
    gracefully (rule-only / neutral responses) instead of failing.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # vLLM (OpenAI-compatible /v1/chat/completions)
    vllm_base_url: str = "http://localhost:8001"
    vllm_model: str = "Qwen/Qwen2.5-Coder-32B-Instruct"
    vllm_api_key: str | None = None
    llm_timeout_seconds: float = Field(default=30, gt=0)

    # Optional shared bearer secret for backend -> AI calls. Unset = open (dev).
    service_api_key: str | None = None

    log_level: str = "INFO"
    max_source_code_chars: int = Field(default=20000, gt=0)
    prompt_max_chars: int = Field(default=12000, gt=0)
    rule_engine_enabled: bool = True
    max_concept_gaps: int = Field(default=8, ge=0)
    max_used_items: int = Field(default=32, ge=0)


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()

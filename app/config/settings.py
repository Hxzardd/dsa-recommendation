"""Application settings loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed runtime settings for the AI analysis service."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: Literal["ollama", "vllm"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:7b"
    vllm_base_url: str = "http://localhost:8000"
    vllm_model: str = "qwen2.5-coder-7b"
    llm_timeout_seconds: float = Field(default=20, gt=0)
    log_level: str = "INFO"
    max_source_code_chars: int = Field(default=20000, gt=0)
    prompt_max_chars: int = Field(default=12000, gt=0)
    rule_engine_enabled: bool = True
    max_concept_gaps: int = Field(default=8, ge=0)


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()


"""Settings tests."""

from app.config.settings import Settings

ENV_VARS = [
    "LLM_PROVIDER",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "VLLM_BASE_URL",
    "VLLM_MODEL",
    "LLM_TIMEOUT_SECONDS",
    "LOG_LEVEL",
    "MAX_SOURCE_CODE_CHARS",
    "PROMPT_MAX_CHARS",
    "RULE_ENGINE_ENABLED",
    "MAX_CONCEPT_GAPS",
]


def test_settings_defaults(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Settings expose the documented defaults."""

    for env_var in ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)

    settings = Settings(_env_file=None)

    assert settings.llm_provider == "ollama"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_model == "qwen2.5-coder:7b"
    assert settings.vllm_base_url == "http://localhost:8000"
    assert settings.vllm_model == "qwen2.5-coder-7b"
    assert settings.llm_timeout_seconds == 20
    assert settings.log_level == "INFO"
    assert settings.max_source_code_chars == 20000
    assert settings.prompt_max_chars == 12000
    assert settings.rule_engine_enabled is True
    assert settings.max_concept_gaps == 8


def test_settings_load_from_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Environment variables override defaults with typed values."""

    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "8")
    monkeypatch.setenv("RULE_ENGINE_ENABLED", "false")

    settings = Settings(_env_file=None)

    assert settings.llm_provider == "vllm"
    assert settings.llm_timeout_seconds == 8
    assert settings.rule_engine_enabled is False

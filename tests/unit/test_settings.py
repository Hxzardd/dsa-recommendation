"""Settings tests."""

from app.config.settings import Settings

ENV_VARS = [
    "VLLM_BASE_URL",
    "VLLM_MODEL",
    "VLLM_API_KEY",
    "SERVICE_API_KEY",
    "LLM_TIMEOUT_SECONDS",
    "LOG_LEVEL",
    "MAX_SOURCE_CODE_CHARS",
    "PROMPT_MAX_CHARS",
    "RULE_ENGINE_ENABLED",
    "MAX_CONCEPT_GAPS",
    "MAX_USED_ITEMS",
]


def test_settings_defaults(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Settings expose the documented vLLM-only defaults."""

    for env_var in ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)

    settings = Settings(_env_file=None)

    assert settings.vllm_base_url == "http://localhost:8001"
    assert settings.vllm_model == "Qwen/Qwen2.5-Coder-32B-Instruct"
    assert settings.vllm_api_key is None
    assert settings.service_api_key is None
    assert settings.llm_timeout_seconds == 30
    assert settings.log_level == "INFO"
    assert settings.max_source_code_chars == 20000
    assert settings.prompt_max_chars == 12000
    assert settings.rule_engine_enabled is True
    assert settings.max_concept_gaps == 8
    assert settings.max_used_items == 32


def test_settings_load_from_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Environment variables override defaults with typed values."""

    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm.internal:9000")
    monkeypatch.setenv("VLLM_MODEL", "my-code-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "8")
    monkeypatch.setenv("RULE_ENGINE_ENABLED", "false")

    settings = Settings(_env_file=None)

    assert settings.vllm_base_url == "http://vllm.internal:9000"
    assert settings.vllm_model == "my-code-model"
    assert settings.llm_timeout_seconds == 8
    assert settings.rule_engine_enabled is False

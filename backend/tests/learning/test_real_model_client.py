from __future__ import annotations

import requests

from songguo.backend.services.learning.real_model_client import (
    OpenAICompatibleModelClient,
    RealModelConfig,
    load_real_model_config,
)


def _config(**overrides):
    values = {
        "binding": "deepseek",
        "model": "deepseek-chat",
        "api_key": "test-key",
        "base_url": "https://api.deepseek.com",
        "timeout_seconds": 7.5,
        "retry_attempts": 2,
        "retry_backoff_seconds": 0.0,
    }
    values.update(overrides)
    return RealModelConfig(**values)


def test_real_model_client_retries_timeout_with_configured_timeout(monkeypatch) -> None:
    calls: list[float] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    def fake_post(_endpoint, *, headers, json, timeout):
        calls.append(timeout)
        if len(calls) == 1:
            raise requests.Timeout("slow provider")
        return FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)

    client = OpenAICompatibleModelClient(config=_config())

    assert client.complete_sync("prompt", system_prompt="system", temperature=0) == '{"ok": true}'
    assert calls == [7.5, 7.5]


def test_load_real_model_config_reads_timeout_and_retry_env(monkeypatch) -> None:
    monkeypatch.setenv("LLM_BINDING", "deepseek")
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("SONGGUO_REAL_MODEL_TIMEOUT_SECONDS", "9")
    monkeypatch.setenv("SONGGUO_REAL_MODEL_RETRY_ATTEMPTS", "3")
    monkeypatch.setenv("SONGGUO_REAL_MODEL_RETRY_BACKOFF_SECONDS", "0")

    config = load_real_model_config()

    assert config.timeout_seconds == 9
    assert config.retry_attempts == 3
    assert config.retry_backoff_seconds == 0

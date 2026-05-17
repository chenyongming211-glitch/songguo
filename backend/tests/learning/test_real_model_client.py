from __future__ import annotations

import requests

import songguo.backend.services.learning.real_model_client as real_model_client
from songguo.backend.services.learning.real_model_client import (
    AliyunEduOCRConfig,
    OpenAICompatibleModelClient,
    RealModelConfig,
    load_aliyun_edu_ocr_config,
    load_real_model_config,
    load_vision_model_config,
    supports_vision,
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


def test_load_real_model_config_defaults_to_deepseek_v4_flash(monkeypatch) -> None:
    monkeypatch.setattr(real_model_client, "_env", lambda _key: "")

    config = load_real_model_config()

    assert config.binding == "deepseek"
    assert config.model == "deepseek-v4-flash"


def test_supports_aliyun_qwen_visual_understanding_models() -> None:
    assert supports_vision("aliyun", "qwen3.6-flash") is True
    assert supports_vision("aliyun", "qwen3.6-plus") is True
    assert supports_vision("aliyun", "deepseek-v4-flash") is False


def test_load_vision_model_config_prefers_songguo_vision_env(monkeypatch) -> None:
    values = {
        "LLM_BINDING": "deepseek",
        "LLM_MODEL": "deepseek-v4-flash",
        "LLM_API_KEY": "text-key",
        "LLM_HOST": "https://api.deepseek.com",
        "SONGGUO_VISION_BINDING": "aliyun",
        "SONGGUO_VISION_MODEL": "qwen3.6-flash",
        "SONGGUO_VISION_API_KEY": "vision-key",
        "SONGGUO_VISION_HOST": "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    }
    monkeypatch.setattr(real_model_client, "_env", lambda key: values.get(key, ""))

    config = load_vision_model_config()

    assert config.binding == "aliyun"
    assert config.model == "qwen3.6-flash"
    assert config.api_key == "vision-key"
    assert config.base_url == "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"


def test_load_vision_model_config_does_not_reuse_text_key_for_aliyun(monkeypatch) -> None:
    values = {
        "LLM_BINDING": "deepseek",
        "LLM_API_KEY": "text-key",
        "DEEPSEEK_API_KEY": "deepseek-key",
        "SONGGUO_VISION_BINDING": "aliyun",
        "SONGGUO_VISION_MODEL": "qwen3.6-flash",
    }
    monkeypatch.setattr(real_model_client, "_env", lambda key: values.get(key, ""))

    config = load_vision_model_config()

    assert config.api_key == ""


def test_load_aliyun_edu_ocr_config_reads_songguo_env(monkeypatch) -> None:
    values = {
        "SONGGUO_ALIYUN_EDU_OCR_ACCESS_KEY_ID": "ak-id",
        "SONGGUO_ALIYUN_EDU_OCR_ACCESS_KEY_SECRET": "ak-secret",
        "SONGGUO_ALIYUN_EDU_OCR_API_KEY": "api-key",
        "SONGGUO_ALIYUN_EDU_OCR_ENDPOINT": "https://ocr-api.cn-hangzhou.aliyuncs.com",
        "SONGGUO_ALIYUN_EDU_OCR_REGION": "cn-hangzhou",
        "SONGGUO_ALIYUN_EDU_OCR_SCENE": "paper_cut",
        "SONGGUO_ALIYUN_EDU_OCR_CUT_TYPE": "question",
        "SONGGUO_ALIYUN_EDU_OCR_TIMEOUT_SECONDS": "11",
        "SONGGUO_ALIYUN_EDU_OCR_RETRY_ATTEMPTS": "3",
        "SONGGUO_ALIYUN_EDU_OCR_RETRY_BACKOFF_SECONDS": "0",
        "SONGGUO_ALIYUN_EDU_OCR_OUTPUT_ORICOORD": "true",
        "SONGGUO_ALIYUN_EDU_OCR_NEED_ROTATE": "false",
        "SONGGUO_ALIYUN_EDU_OCR_FALLBACK_PROVIDER": "vision",
        "SONGGUO_ALIYUN_EDU_OCR_HYBRID_TEXT_FALLBACK": "true",
        "SONGGUO_ALIYUN_EDU_OCR_HYBRID_TEXT_FALLBACK_MIN_ANSWER_RATE": "0.6",
        "SONGGUO_ALIYUN_EDU_OCR_ROUTER_MODE": "auto",
        "SONGGUO_ALIYUN_EDU_OCR_MAX_SECONDARY_ACTIONS": "1",
    }
    monkeypatch.setattr(real_model_client, "_env", lambda key: values.get(key, ""))

    config = load_aliyun_edu_ocr_config()

    assert isinstance(config, AliyunEduOCRConfig)
    assert config.access_key_id == "ak-id"
    assert config.access_key_secret == "ak-secret"
    assert config.api_key == "api-key"
    assert config.endpoint == "https://ocr-api.cn-hangzhou.aliyuncs.com"
    assert config.region == "cn-hangzhou"
    assert config.scene == "paper_cut"
    assert config.cut_type == "question"
    assert config.output_oricoord is True
    assert config.need_rotate is False
    assert config.timeout_seconds == 11
    assert config.retry_attempts == 3
    assert config.retry_backoff_seconds == 0
    assert config.fallback_provider == "vision"
    assert config.hybrid_text_fallback is True
    assert config.hybrid_text_fallback_min_answer_rate == 0.6
    assert config.router_mode == "auto"
    assert config.max_secondary_actions == 1


def test_load_aliyun_edu_ocr_config_defaults_to_auto_scene(monkeypatch) -> None:
    monkeypatch.setattr(real_model_client, "_env", lambda _key: "")

    config = load_aliyun_edu_ocr_config()

    assert config.scene == "auto"
    assert config.fallback_provider == "none"
    assert config.router_mode == "auto"
    assert config.max_secondary_actions == 1

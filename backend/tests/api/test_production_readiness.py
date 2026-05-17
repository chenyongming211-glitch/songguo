from __future__ import annotations

import pytest

from songguo.backend.services.production_readiness import check_production_readiness


@pytest.fixture(autouse=True)
def _ignore_local_dotenv(monkeypatch) -> None:
    monkeypatch.setattr(
        "songguo.backend.services.production_readiness._read_local_dotenv_value",
        lambda key: "",
    )


def test_production_readiness_reports_missing_required_env(monkeypatch) -> None:
    for key in [
        "PUBLIC_BASE_URL",
        "WECHAT_APPID",
        "WECHAT_SECRET",
        "SONGGUO_SESSION_SECRET",
    ]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        "songguo.backend.services.production_readiness._read_local_dotenv_value",
        lambda key: "",
    )

    result = check_production_readiness()

    assert result.ready is False
    assert "PUBLIC_BASE_URL" in result.missing
    assert "WECHAT_APPID" in result.missing


def test_production_readiness_reads_local_dotenv_values(monkeypatch) -> None:
    values = {
        "PUBLIC_BASE_URL": "https://example.com",
        "WECHAT_APPID": "appid",
        "WECHAT_SECRET": "secret",
        "SONGGUO_SESSION_SECRET": "session-secret",
    }
    for key in values:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        "songguo.backend.services.production_readiness._read_local_dotenv_value",
        lambda key: values.get(key, ""),
    )

    result = check_production_readiness()

    assert result.ready is True
    assert result.missing == []


def test_production_readiness_passes_required_env(monkeypatch) -> None:
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://example.com")
    monkeypatch.setenv("WECHAT_APPID", "appid")
    monkeypatch.setenv("WECHAT_SECRET", "secret")
    monkeypatch.setenv("SONGGUO_SESSION_SECRET", "session-secret")

    result = check_production_readiness()

    assert result.ready is True
    assert result.missing == []


def test_production_readiness_requires_explicit_vision_model_when_ocr_uses_vision(
    monkeypatch,
) -> None:
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://example.com")
    monkeypatch.setenv("WECHAT_APPID", "appid")
    monkeypatch.setenv("WECHAT_SECRET", "secret")
    monkeypatch.setenv("SONGGUO_SESSION_SECRET", "session-secret")
    monkeypatch.setenv("SONGGUO_PHOTO_OCR_PROVIDER", "vision")
    monkeypatch.delenv("SONGGUO_VISION_MODEL", raising=False)

    result = check_production_readiness()

    assert result.ready is False
    assert "SONGGUO_VISION_MODEL" in result.missing


def test_production_readiness_accepts_known_vision_model(monkeypatch) -> None:
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://example.com")
    monkeypatch.setenv("WECHAT_APPID", "appid")
    monkeypatch.setenv("WECHAT_SECRET", "secret")
    monkeypatch.setenv("SONGGUO_SESSION_SECRET", "session-secret")
    monkeypatch.setenv("SONGGUO_PHOTO_OCR_PROVIDER", "vision")
    monkeypatch.setenv("SONGGUO_VISION_MODEL", "llava:latest")
    monkeypatch.setenv("SONGGUO_VISION_API_KEY", "vision-key")
    monkeypatch.setenv("LLM_BINDING", "ollama")

    result = check_production_readiness()

    assert result.ready is True
    assert result.missing == []
    assert not any("vision support" in warning for warning in result.warnings)


def test_production_readiness_accepts_aliyun_qwen_vision_model(monkeypatch) -> None:
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://example.com")
    monkeypatch.setenv("WECHAT_APPID", "appid")
    monkeypatch.setenv("WECHAT_SECRET", "secret")
    monkeypatch.setenv("SONGGUO_SESSION_SECRET", "session-secret")
    monkeypatch.setenv("SONGGUO_PHOTO_OCR_PROVIDER", "vision")
    monkeypatch.setenv("SONGGUO_VISION_BINDING", "aliyun")
    monkeypatch.setenv("SONGGUO_VISION_MODEL", "qwen3.6-flash")
    monkeypatch.setenv("SONGGUO_VISION_API_KEY", "vision-key")

    result = check_production_readiness()

    assert result.ready is True
    assert result.missing == []
    assert not any("vision support" in warning for warning in result.warnings)


def test_production_readiness_requires_aliyun_edu_ocr_credentials(monkeypatch) -> None:
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://example.com")
    monkeypatch.setenv("WECHAT_APPID", "appid")
    monkeypatch.setenv("WECHAT_SECRET", "secret")
    monkeypatch.setenv("SONGGUO_SESSION_SECRET", "session-secret")
    monkeypatch.setenv("SONGGUO_PHOTO_OCR_PROVIDER", "aliyun_edu")
    monkeypatch.delenv("SONGGUO_ALIYUN_EDU_OCR_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("SONGGUO_ALIYUN_EDU_OCR_ACCESS_KEY_SECRET", raising=False)

    result = check_production_readiness()

    assert result.ready is False
    assert "SONGGUO_ALIYUN_EDU_OCR_ACCESS_KEY_ID" in result.missing
    assert "SONGGUO_ALIYUN_EDU_OCR_ACCESS_KEY_SECRET" in result.missing


def test_production_readiness_accepts_aliyun_edu_ocr_credentials(monkeypatch) -> None:
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://example.com")
    monkeypatch.setenv("WECHAT_APPID", "appid")
    monkeypatch.setenv("WECHAT_SECRET", "secret")
    monkeypatch.setenv("SONGGUO_SESSION_SECRET", "session-secret")
    monkeypatch.setenv("SONGGUO_PHOTO_OCR_PROVIDER", "aliyun_edu")
    monkeypatch.setenv("SONGGUO_ALIYUN_EDU_OCR_ACCESS_KEY_ID", "ak-id")
    monkeypatch.setenv("SONGGUO_ALIYUN_EDU_OCR_ACCESS_KEY_SECRET", "ak-secret")

    result = check_production_readiness()

    assert "SONGGUO_ALIYUN_EDU_OCR_ACCESS_KEY_ID" not in result.missing
    assert "SONGGUO_ALIYUN_EDU_OCR_ACCESS_KEY_SECRET" not in result.missing


def test_production_readiness_warns_when_wechat_production_paths_are_disabled(
    monkeypatch,
) -> None:
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://example.com")
    monkeypatch.setenv("WECHAT_APPID", "appid")
    monkeypatch.setenv("WECHAT_SECRET", "secret")
    monkeypatch.setenv("SONGGUO_SESSION_SECRET", "session-secret")
    monkeypatch.delenv("WECHAT_CONTENT_SAFETY_PROVIDER", raising=False)
    monkeypatch.delenv("WECHAT_SEND_SUBSCRIBE_MESSAGES", raising=False)

    result = check_production_readiness()

    assert result.ready is True
    assert any("WECHAT_CONTENT_SAFETY_PROVIDER" in warning for warning in result.warnings)
    assert any("WECHAT_SEND_SUBSCRIBE_MESSAGES" in warning for warning in result.warnings)

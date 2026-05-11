from __future__ import annotations

from songguo.backend.services.production_readiness import check_production_readiness


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
    monkeypatch.setenv("LLM_BINDING", "ollama")

    result = check_production_readiness()

    assert result.ready is True
    assert result.missing == []
    assert not any("vision support" in warning for warning in result.warnings)


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

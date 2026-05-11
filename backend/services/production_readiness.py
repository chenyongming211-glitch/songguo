from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from songguo.backend.services.learning.real_model_client import supports_vision


REQUIRED_ENV = (
    "PUBLIC_BASE_URL",
    "WECHAT_APPID",
    "WECHAT_SECRET",
    "SONGGUO_SESSION_SECRET",
)


class ProductionReadiness(BaseModel):
    ready: bool
    missing: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def check_production_readiness() -> ProductionReadiness:
    missing = [key for key in REQUIRED_ENV if not _env_with_legacy(key)]
    warnings: list[str] = []
    photo_ocr_provider = _env_with_legacy("SONGGUO_PHOTO_OCR_PROVIDER")
    if photo_ocr_provider == "vision":
        vision_model = _env_with_legacy("SONGGUO_VISION_MODEL")
        if not vision_model:
            missing.append("SONGGUO_VISION_MODEL")
        else:
            binding = _env_with_legacy("LLM_BINDING") or "openai"
            if not supports_vision(binding, vision_model):
                warnings.append(
                    "SONGGUO_VISION_MODEL does not advertise vision support for the configured LLM_BINDING"
                )
    elif not photo_ocr_provider:
        warnings.append("SONGGUO_PHOTO_OCR_PROVIDER is not set; photo review will use local fallback OCR")
    if _env_with_legacy("WECHAT_CONTENT_SAFETY_PROVIDER") != "wechat":
        warnings.append("WECHAT_CONTENT_SAFETY_PROVIDER is not set to wechat; text safety uses local rules only")
    if _env_with_legacy("WECHAT_SEND_SUBSCRIBE_MESSAGES").lower() not in {"1", "true", "yes"}:
        warnings.append("WECHAT_SEND_SUBSCRIBE_MESSAGES is disabled; reminder jobs will not call WeChat")
    if _env_with_legacy("PUBLIC_BASE_URL").startswith("http://"):
        warnings.append("PUBLIC_BASE_URL should use https for WeChat production")
    return ProductionReadiness(
        ready=not missing,
        missing=missing,
        warnings=warnings,
    )


def _env_with_legacy(key: str) -> str:
    value = os.getenv(key)
    if value:
        return value
    value = _read_local_dotenv_value(key)
    if value:
        return value
    legacy = {
        "SONGGUO_SESSION_SECRET": "DEEPTUTOR_SESSION_SECRET",
        "SONGGUO_PHOTO_OCR_PROVIDER": "DEEPTUTOR_PHOTO_OCR_PROVIDER",
        "SONGGUO_VISION_MODEL": "DEEPTUTOR_VISION_MODEL",
    }.get(key)
    if not legacy:
        return ""
    return os.getenv(legacy) or _read_local_dotenv_value(legacy)


def _read_local_dotenv_value(key: str) -> str:
    root = Path(__file__).resolve().parents[3]
    for env_path in (root / "songguo" / ".env", root / ".env"):
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            current_key, value = line.split("=", 1)
            if current_key.strip() == key:
                return value.strip().strip("\"'")
    return ""

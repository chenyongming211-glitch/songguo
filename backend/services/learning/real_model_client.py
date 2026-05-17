from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import time
from typing import Any

import requests


DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_ALIYUN_VISION_MODEL = "qwen3.6-flash"
DEFAULT_ALIYUN_OPENAI_BASE_URL = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
DEFAULT_ALIYUN_EDU_OCR_ENDPOINT = "https://ocr-api.cn-hangzhou.aliyuncs.com"
DEFAULT_ALIYUN_EDU_OCR_REGION = "cn-hangzhou"


@dataclass(frozen=True)
class RealModelConfig:
    binding: str
    model: str
    api_key: str
    base_url: str
    timeout_seconds: float = 8.0
    retry_attempts: int = 2
    retry_backoff_seconds: float = 0.4

    def model_copy(self, update: dict[str, Any] | None = None) -> "RealModelConfig":
        return replace(self, **(update or {}))


@dataclass(frozen=True)
class AliyunEduOCRConfig:
    access_key_id: str = ""
    access_key_secret: str = ""
    api_key: str = ""
    endpoint: str = DEFAULT_ALIYUN_EDU_OCR_ENDPOINT
    region: str = DEFAULT_ALIYUN_EDU_OCR_REGION
    scene: str = "auto"
    cut_type: str = "question"
    image_type: str = "photo"
    subject: str = "default"
    output_oricoord: bool = True
    need_rotate: bool = True
    timeout_seconds: float = 12.0
    retry_attempts: int = 2
    retry_backoff_seconds: float = 0.4
    fallback_provider: str = "none"
    hybrid_text_fallback: bool = False
    hybrid_text_fallback_min_answer_rate: float = 0.6
    router_mode: str = "auto"
    max_secondary_actions: int = 1

    def model_copy(self, update: dict[str, Any] | None = None) -> "AliyunEduOCRConfig":
        return replace(self, **(update or {}))


class OpenAICompatibleModelClient:
    def __init__(self, config: RealModelConfig | None = None) -> None:
        self.config = config or load_real_model_config()

    def complete_sync(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0,
    ) -> str:
        if not self.config.api_key:
            raise RuntimeError("LLM_API_KEY is not configured")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = request_chat_completion(
            self.config,
            json_payload={
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature,
            },
        )
        return str(payload["choices"][0]["message"]["content"])

    def get_vision_model_func(self):
        if not supports_vision(self.config.binding, self.config.model):
            raise RuntimeError(
                f"Model {self.config.model} does not support vision for {self.config.binding}"
            )

        async def _call_vision(*, prompt: str, image_data: str) -> str:
            json_payload: dict[str, Any] = {
                "model": self.config.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_data}},
                        ],
                    }
                ],
                "temperature": 0,
            }
            max_tokens = _int_env("SONGGUO_VISION_MAX_TOKENS", 900, minimum=128)
            if max_tokens:
                json_payload["max_tokens"] = max_tokens
            payload = request_chat_completion(
                self.config,
                json_payload=json_payload,
            )
            return str(payload["choices"][0]["message"]["content"])

        return _call_vision


def get_llm_client() -> OpenAICompatibleModelClient:
    return OpenAICompatibleModelClient()


def load_real_model_config() -> RealModelConfig:
    binding = _env("LLM_BINDING") or "deepseek"
    return RealModelConfig(
        binding=binding,
        model=_env("LLM_MODEL") or _env("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL,
        api_key=_env("LLM_API_KEY") or _env("DEEPSEEK_API_KEY"),
        base_url=_env("LLM_HOST") or _default_base_url(binding),
        timeout_seconds=_float_env("SONGGUO_REAL_MODEL_TIMEOUT_SECONDS", 8.0, minimum=0.1),
        retry_attempts=_int_env("SONGGUO_REAL_MODEL_RETRY_ATTEMPTS", 2, minimum=1),
        retry_backoff_seconds=_float_env(
            "SONGGUO_REAL_MODEL_RETRY_BACKOFF_SECONDS",
            0.4,
            minimum=0.0,
        ),
    )


def load_vision_model_config(*, model: str | None = None) -> RealModelConfig:
    binding = _env("SONGGUO_VISION_BINDING") or _env("VISION_BINDING") or _env("LLM_BINDING") or "aliyun"
    resolved_model = (
        model
        or _env("SONGGUO_VISION_MODEL")
        or _env("VISION_MODEL")
        or DEFAULT_ALIYUN_VISION_MODEL
    )
    api_key = (
        _env("SONGGUO_VISION_API_KEY")
        or _env("VISION_API_KEY")
        or _env("ALIYUN_API_KEY")
    )
    if not api_key and (binding or "").lower() not in {"aliyun", "dashscope", "qwen", "maas"}:
        api_key = _env("LLM_API_KEY") or _env("DEEPSEEK_API_KEY")
    return RealModelConfig(
        binding=binding,
        model=resolved_model,
        api_key=api_key,
        base_url=(
            _env("SONGGUO_VISION_HOST")
            or _env("VISION_HOST")
            or _env("ALIYUN_OPENAI_BASE_URL")
            or _default_base_url(binding)
        ),
        timeout_seconds=_float_env("SONGGUO_VISION_TIMEOUT_SECONDS", 20.0, minimum=0.1),
        retry_attempts=_int_env("SONGGUO_VISION_RETRY_ATTEMPTS", 2, minimum=1),
        retry_backoff_seconds=_float_env(
            "SONGGUO_VISION_RETRY_BACKOFF_SECONDS",
            0.4,
            minimum=0.0,
        ),
    )


def load_aliyun_edu_ocr_config() -> AliyunEduOCRConfig:
    return AliyunEduOCRConfig(
        access_key_id=(
            _env("SONGGUO_ALIYUN_EDU_OCR_ACCESS_KEY_ID")
            or _env("ALIYUN_ACCESS_KEY_ID")
        ),
        access_key_secret=(
            _env("SONGGUO_ALIYUN_EDU_OCR_ACCESS_KEY_SECRET")
            or _env("ALIYUN_ACCESS_KEY_SECRET")
        ),
        api_key=_env("SONGGUO_ALIYUN_EDU_OCR_API_KEY") or _env("ALIYUN_EDU_OCR_API_KEY"),
        endpoint=_env("SONGGUO_ALIYUN_EDU_OCR_ENDPOINT") or DEFAULT_ALIYUN_EDU_OCR_ENDPOINT,
        region=_env("SONGGUO_ALIYUN_EDU_OCR_REGION") or DEFAULT_ALIYUN_EDU_OCR_REGION,
        scene=_env("SONGGUO_ALIYUN_EDU_OCR_SCENE") or "auto",
        cut_type=_env("SONGGUO_ALIYUN_EDU_OCR_CUT_TYPE") or "question",
        image_type=_env("SONGGUO_ALIYUN_EDU_OCR_IMAGE_TYPE") or "photo",
        subject=_env("SONGGUO_ALIYUN_EDU_OCR_SUBJECT") or "default",
        output_oricoord=_bool_env("SONGGUO_ALIYUN_EDU_OCR_OUTPUT_ORICOORD", True),
        need_rotate=_bool_env("SONGGUO_ALIYUN_EDU_OCR_NEED_ROTATE", True),
        timeout_seconds=_float_env("SONGGUO_ALIYUN_EDU_OCR_TIMEOUT_SECONDS", 12.0, minimum=0.1),
        retry_attempts=_int_env("SONGGUO_ALIYUN_EDU_OCR_RETRY_ATTEMPTS", 2, minimum=1),
        retry_backoff_seconds=_float_env(
            "SONGGUO_ALIYUN_EDU_OCR_RETRY_BACKOFF_SECONDS",
            0.4,
            minimum=0.0,
        ),
        fallback_provider=(
            _env("SONGGUO_ALIYUN_EDU_OCR_FALLBACK_PROVIDER")
            or "none"
        ),
        hybrid_text_fallback=_bool_env("SONGGUO_ALIYUN_EDU_OCR_HYBRID_TEXT_FALLBACK", False),
        hybrid_text_fallback_min_answer_rate=_float_env(
            "SONGGUO_ALIYUN_EDU_OCR_HYBRID_TEXT_FALLBACK_MIN_ANSWER_RATE",
            0.6,
            minimum=0.0,
        ),
        router_mode=_env("SONGGUO_ALIYUN_EDU_OCR_ROUTER_MODE") or "auto",
        max_secondary_actions=_int_env(
            "SONGGUO_ALIYUN_EDU_OCR_MAX_SECONDARY_ACTIONS",
            1,
            minimum=0,
        ),
    )


def request_chat_completion(
    config: RealModelConfig,
    *,
    json_payload: dict[str, Any],
) -> dict[str, Any]:
    endpoint = f"{config.base_url.rstrip('/')}/chat/completions"
    attempts = max(1, int(config.retry_attempts))
    last_error: Exception | None = None
    for attempt_index in range(attempts):
        try:
            response = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {config.api_key}",
                    "Content-Type": "application/json",
                },
                json=json_payload,
                timeout=config.timeout_seconds,
            )
            status_code = getattr(response, "status_code", None)
            if _is_retryable_status(status_code) and attempt_index + 1 < attempts:
                _sleep_before_retry(config, attempt_index)
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("LLM response payload must be a JSON object")
            return payload
        except requests.Timeout as exc:
            last_error = exc
        except requests.ConnectionError as exc:
            last_error = exc
        except requests.RequestException as exc:
            last_error = exc
            if not _is_retryable_request_exception(exc):
                raise
        if attempt_index + 1 < attempts:
            _sleep_before_retry(config, attempt_index)
    if last_error is not None:
        raise last_error
    raise RuntimeError("LLM request failed")


def supports_vision(binding: str, model: str) -> bool:
    normalized_binding = (binding or "").lower()
    normalized_model = (model or "").lower()
    if normalized_binding == "deepseek":
        return False
    if normalized_binding in {"aliyun", "dashscope", "qwen", "maas"}:
        return normalized_model in {
            "qwen3.6-flash",
            "qwen3.6-plus",
            "kimi-k2.5",
            "kimi-k2.6",
        }
    if normalized_binding == "ollama" and any(
        token in normalized_model for token in ("llava", "bakllava", "moondream", "minicpm-v")
    ):
        return True
    return any(token in normalized_model for token in ("vision", "vl", "gpt-4o", "qwen-vl"))


def songguo_data_root() -> Path:
    return Path(_env("SONGGUO_DATA_ROOT") or "songguo/data").resolve()


def _default_base_url(binding: str) -> str:
    normalized = (binding or "").lower()
    if normalized == "deepseek":
        return "https://api.deepseek.com"
    if normalized in {"aliyun", "dashscope", "qwen", "maas"}:
        return _env("ALIYUN_OPENAI_BASE_URL") or DEFAULT_ALIYUN_OPENAI_BASE_URL
    return _env("OPENAI_BASE_URL") or "https://api.openai.com/v1"


def _env(key: str) -> str:
    value = os.getenv(key)
    if value:
        return value.strip().strip("\"'")
    root = Path(__file__).resolve().parents[4]
    for env_path in (root / "songguo" / ".env", root / ".env"):
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            current_key, current_value = line.split("=", 1)
            if current_key.strip() == key:
                return current_value.strip().strip("\"'")
    return ""


def _float_env(key: str, default: float, *, minimum: float) -> float:
    raw = _env(key)
    if not raw:
        return default
    try:
        return max(minimum, float(raw))
    except ValueError:
        return default


def _int_env(key: str, default: int, *, minimum: int) -> int:
    raw = _env(key)
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _bool_env(key: str, default: bool) -> bool:
    raw = _env(key).strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def _is_retryable_status(status_code: Any) -> bool:
    if not isinstance(status_code, int):
        return False
    return status_code == 429 or 500 <= status_code <= 599


def _is_retryable_request_exception(exc: requests.RequestException) -> bool:
    response = getattr(exc, "response", None)
    return _is_retryable_status(getattr(response, "status_code", None))


def _sleep_before_retry(config: RealModelConfig, attempt_index: int) -> None:
    backoff = max(0.0, float(config.retry_backoff_seconds))
    if backoff <= 0:
        return
    time.sleep(backoff * (2**attempt_index))

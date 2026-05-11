from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import hmac
import os
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json

from pydantic import BaseModel

from songguo.backend.services.learning.input_safety import check_learning_input


class WechatLoginResult(BaseModel):
    openid: str
    session_token: str
    child_id: str
    expires_in: int = 7200


class ContentSafetyResult(BaseModel):
    safe: bool
    reason: str = "safe"
    provider: str = "local"
    message: str = ""


@dataclass
class WechatConfig:
    appid: str = ""
    secret: str = ""
    mock_openid: str = ""
    session_secret: str = ""

    @classmethod
    def from_env(cls) -> "WechatConfig":
        appid = os.getenv("WECHAT_APPID", "")
        secret = os.getenv("WECHAT_SECRET", "")
        mock_openid = os.getenv("WECHAT_MOCK_OPENID", "")
        if not appid and not secret and not mock_openid:
            mock_openid = "openid_local_dev"
        return cls(
            appid=appid,
            secret=secret,
            mock_openid=mock_openid,
            session_secret=(
                os.getenv("SONGGUO_SESSION_SECRET")
                or os.getenv("DEEPTUTOR_SESSION_SECRET")
                or "local-dev-session-secret"
            ),
        )


class WechatService:
    def __init__(
        self,
        config: WechatConfig | None = None,
        *,
        http_get_json: Callable[[str], dict[str, Any]] | None = None,
        http_post_json: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        content_safety_provider: str | None = None,
    ) -> None:
        self.config = config or WechatConfig.from_env()
        self.http_get_json = http_get_json or _http_get_json
        self.http_post_json = http_post_json or _http_post_json
        self.content_safety_provider = content_safety_provider or os.getenv(
            "WECHAT_CONTENT_SAFETY_PROVIDER",
            "local",
        )

    def login(self, code: str) -> WechatLoginResult:
        openid = self.config.mock_openid or self._exchange_code_for_openid(code)
        return WechatLoginResult(
            openid=openid,
            session_token=self._sign_session(openid),
            child_id=f"child_{openid}",
        )

    def parse_session_token(self, token: str, *, max_age_seconds: int = 7200) -> str:
        try:
            openid, timestamp, encoded_sig = token.rsplit(".", 2)
            issued_at = int(timestamp)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("Invalid session token") from exc

        body = f"{openid}.{timestamp}"
        expected = hmac.new(
            self.config.session_secret.encode("utf-8"),
            body.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        padding = "=" * (-len(encoded_sig) % 4)
        try:
            actual = base64.urlsafe_b64decode(f"{encoded_sig}{padding}")
        except ValueError as exc:
            raise ValueError("Invalid session token") from exc
        if not hmac.compare_digest(actual, expected):
            raise ValueError("Invalid session token")
        if max_age_seconds > 0 and int(time.time()) - issued_at > max_age_seconds:
            raise ValueError("Session token expired")
        return openid

    def check_text(self, text: str, *, openid: str | None = None) -> ContentSafetyResult:
        verdict = check_learning_input(text)
        if not verdict.allowed:
            return ContentSafetyResult(
                safe=False,
                reason=verdict.reason,
                provider="local",
                message=verdict.message,
            )
        if self.content_safety_provider == "wechat":
            return self._check_text_with_wechat(text, openid=openid)
        return ContentSafetyResult(safe=True)

    def send_subscribe_message(
        self,
        *,
        openid: str,
        template_id: str,
        data: dict[str, Any],
        page: str | None = None,
    ) -> None:
        token = self._get_access_token()
        payload: dict[str, Any] = {
            "touser": openid,
            "template_id": template_id,
            "data": data,
        }
        if page:
            payload["page"] = page
        response = self.http_post_json(
            f"https://api.weixin.qq.com/cgi-bin/message/subscribe/send?access_token={token}",
            payload,
        )
        if response.get("errcode"):
            raise ValueError(str(response.get("errmsg") or response["errcode"]))

    def _exchange_code_for_openid(self, code: str) -> str:
        if not self.config.appid or not self.config.secret:
            raise ValueError("WECHAT_APPID and WECHAT_SECRET are required for WeChat login")
        query = urlencode(
            {
                "appid": self.config.appid,
                "secret": self.config.secret,
                "js_code": code,
                "grant_type": "authorization_code",
            }
        )
        payload = self.http_get_json(f"https://api.weixin.qq.com/sns/jscode2session?{query}")
        if payload.get("errcode"):
            raise ValueError(str(payload.get("errmsg") or payload["errcode"]))
        openid = payload.get("openid")
        if not openid:
            raise ValueError("WeChat login response did not include openid")
        return str(openid)

    def _check_text_with_wechat(self, text: str, *, openid: str | None) -> ContentSafetyResult:
        if not openid:
            return ContentSafetyResult(
                safe=False,
                reason="openid_required",
                provider="wechat",
                message="WeChat content safety requires openid",
            )
        token = self._get_access_token()
        payload = {
            "content": text,
            "version": 2,
            "scene": 2,
            "openid": openid,
        }
        response = self.http_post_json(
            f"https://api.weixin.qq.com/wxa/msg_sec_check?access_token={token}",
            payload,
        )
        if response.get("errcode"):
            return ContentSafetyResult(
                safe=False,
                reason="wechat_error",
                provider="wechat",
                message=str(response.get("errmsg") or response["errcode"]),
            )
        result = response.get("result") or {}
        suggest = str(result.get("suggest") or "pass")
        return ContentSafetyResult(
            safe=suggest == "pass",
            reason="safe" if suggest == "pass" else suggest,
            provider="wechat",
            message=str(result.get("label") or ""),
        )

    def _get_access_token(self) -> str:
        if not self.config.appid or not self.config.secret:
            raise ValueError("WECHAT_APPID and WECHAT_SECRET are required for WeChat access token")
        query = urlencode(
            {
                "grant_type": "client_credential",
                "appid": self.config.appid,
                "secret": self.config.secret,
            }
        )
        payload = self.http_get_json(f"https://api.weixin.qq.com/cgi-bin/token?{query}")
        if payload.get("errcode"):
            raise ValueError(str(payload.get("errmsg") or payload["errcode"]))
        token = payload.get("access_token")
        if not token:
            raise ValueError("WeChat access token response did not include access_token")
        return str(token)

    def _sign_session(self, openid: str) -> str:
        timestamp = str(int(time.time()))
        body = f"{openid}.{timestamp}"
        signature = hmac.new(
            self.config.session_secret.encode("utf-8"),
            body.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        encoded_sig = base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")
        return f"{body}.{encoded_sig}"


def _http_get_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=8) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=8) as resp:
        return json.loads(resp.read().decode("utf-8"))

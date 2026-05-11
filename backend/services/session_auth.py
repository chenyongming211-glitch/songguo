from __future__ import annotations

import os
from typing import Protocol

from songguo.backend.services.wechat import WechatService


class ChildBindingStore(Protocol):
    def bind_child_to_openid(self, *, openid: str, child_id: str) -> None: ...

    def is_child_bound_to_openid(self, *, openid: str, child_id: str) -> bool: ...


def openid_from_session_token(session_token: str) -> str:
    return WechatService().parse_session_token(session_token)


def strict_auth_enabled() -> bool:
    return os.getenv("SONGGUO_AUTH_MODE", "").strip().lower() in {
        "strict",
        "production",
        "prod",
    }


def authorize_child_access(
    store: ChildBindingStore,
    *,
    child_id: str,
    session_token: str | None,
) -> str | None:
    if not session_token:
        if strict_auth_enabled():
            raise ValueError("X-Session-Token is required")
        return None
    openid = openid_from_session_token(session_token)
    if child_id == f"child_{openid}":
        store.bind_child_to_openid(openid=openid, child_id=child_id)
        return openid
    if store.is_child_bound_to_openid(openid=openid, child_id=child_id):
        return openid
    raise PermissionError("child_id is not bound to current openid")

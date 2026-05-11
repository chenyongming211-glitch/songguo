from __future__ import annotations

import pytest

from songguo.backend.services.learning.store import InMemoryLearningStore
from songguo.backend.services.session_auth import authorize_child_access, strict_auth_enabled
from songguo.backend.services.wechat import WechatService


def test_strict_auth_mode_is_explicit(monkeypatch) -> None:
    monkeypatch.delenv("SONGGUO_AUTH_MODE", raising=False)
    assert strict_auth_enabled() is False

    monkeypatch.setenv("SONGGUO_AUTH_MODE", "strict")
    assert strict_auth_enabled() is True


def test_authorize_child_access_requires_token_only_in_strict_mode(monkeypatch) -> None:
    store = InMemoryLearningStore()
    monkeypatch.delenv("SONGGUO_AUTH_MODE", raising=False)
    assert authorize_child_access(store, child_id="child_001", session_token=None) is None

    monkeypatch.setenv("SONGGUO_AUTH_MODE", "strict")
    with pytest.raises(ValueError, match="X-Session-Token is required"):
        authorize_child_access(store, child_id="child_001", session_token=None)


def test_authorize_child_access_binds_current_openid_child(monkeypatch) -> None:
    monkeypatch.setenv("SONGGUO_AUTH_MODE", "strict")
    store = InMemoryLearningStore()
    token = WechatService()._sign_session("openid_001")

    openid = authorize_child_access(
        store,
        child_id="child_openid_001",
        session_token=token,
    )

    assert openid == "openid_001"
    assert store.is_child_bound_to_openid(openid="openid_001", child_id="child_openid_001")

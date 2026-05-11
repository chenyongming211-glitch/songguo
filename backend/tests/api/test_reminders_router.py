from __future__ import annotations

import importlib

import pytest

pytest.importorskip("fastapi")

FastAPI = pytest.importorskip("fastapi").FastAPI
TestClient = pytest.importorskip("fastapi.testclient").TestClient

reminders_router_module = importlib.import_module("songguo.backend.api.routers.reminders")
router = reminders_router_module.router

from songguo.backend.services.learning.store import InMemoryLearningStore
from songguo.backend.services.learning.reminders import send_due_reminders
from songguo.backend.services.wechat import WechatService


def _build_app(store: InMemoryLearningStore) -> FastAPI:
    app = FastAPI()
    reminders_router_module.get_learning_store = lambda: store
    app.include_router(router, prefix="/api/v1/reminders")
    return app


def test_save_subscription_preference_and_list_due_reminders() -> None:
    store = InMemoryLearningStore()
    store.upsert_child(
        child_id="child_001",
        name="小明",
        grade=3,
        term_label="2026春季",
    )

    with TestClient(_build_app(store)) as client:
        saved = client.post(
            "/api/v1/reminders/subscriptions",
            json={
                "openid": "openid_mock",
                "child_id": "child_001",
                "template_id": "tmpl_review",
                "enabled": True,
                "scope": "weekly",
            },
        )
        due = client.get("/api/v1/reminders/due?child_id=child_001")

    assert saved.status_code == 200
    assert due.status_code == 200
    payload = due.json()
    assert payload["items"][0]["template_id"] == "tmpl_review"
    assert payload["items"][0]["scope"] == "weekly"


def test_send_due_reminders_marks_jobs_sent() -> None:
    store = InMemoryLearningStore()
    store.save_reminder_subscription(
        openid="openid_mock",
        child_id="child_001",
        template_id="tmpl_review",
        enabled=True,
        scope="weekly",
    )

    with TestClient(_build_app(store)) as client:
        res = client.post("/api/v1/reminders/send-due", json={"child_id": "child_001"})

    assert res.status_code == 200
    payload = res.json()
    assert payload["sent_count"] == 1
    assert payload["items"][0]["status"] == "sent"


def test_send_due_reminders_can_use_wechat_subscribe_sender() -> None:
    class FakeWechatService:
        def __init__(self) -> None:
            self.sent: list[dict[str, object]] = []

        def send_subscribe_message(self, **kwargs: object) -> None:
            self.sent.append(kwargs)

    store = InMemoryLearningStore()
    store.save_reminder_subscription(
        openid="openid_mock",
        child_id="child_001",
        template_id="tmpl_review",
        enabled=True,
        scope="weekly",
    )
    wechat = FakeWechatService()

    result = send_due_reminders(store, child_id="child_001", wechat_service=wechat)

    assert result.sent_count == 1
    assert result.failed_count == 0
    assert wechat.sent[0]["openid"] == "openid_mock"
    assert wechat.sent[0]["template_id"] == "tmpl_review"


def test_reminders_require_child_access_in_strict_mode(monkeypatch) -> None:
    monkeypatch.setenv("SONGGUO_AUTH_MODE", "strict")
    store = InMemoryLearningStore()

    with TestClient(_build_app(store)) as client:
        saved = client.post(
            "/api/v1/reminders/subscriptions",
            json={
                "openid": "openid_mock",
                "child_id": "child_001",
                "template_id": "tmpl_review",
                "enabled": True,
                "scope": "weekly",
            },
        )
        due = client.get("/api/v1/reminders/due?child_id=child_001")
        sent = client.post("/api/v1/reminders/send-due", json={"child_id": "child_001"})

    assert saved.status_code == 401
    assert due.status_code == 401
    assert sent.status_code == 401


def test_reminder_subscription_uses_session_openid_in_strict_mode(monkeypatch) -> None:
    monkeypatch.setenv("SONGGUO_AUTH_MODE", "strict")
    store = InMemoryLearningStore()
    token = WechatService()._sign_session("openid_001")

    with TestClient(_build_app(store)) as client:
        saved = client.post(
            "/api/v1/reminders/subscriptions",
            headers={"X-Session-Token": token},
            json={
                "openid": "attacker_openid",
                "child_id": "child_openid_001",
                "template_id": "tmpl_review",
                "enabled": True,
                "scope": "weekly",
            },
        )

    assert saved.status_code == 200
    assert saved.json()["openid"] == "openid_001"

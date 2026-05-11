from __future__ import annotations

from fastapi.testclient import TestClient

from songguo.backend.api.app import create_app


def test_songguo_app_exposes_system_status() -> None:
    app = create_app()

    with TestClient(app) as client:
        res = client.get("/api/v1/system/status")

    assert res.status_code == 200
    payload = res.json()
    assert payload["backend"]["status"] == "ok"
    assert payload["llm"]["model"]
    assert payload["runtime"] in {"kernel", "llm", "langgraph"}


def test_songguo_app_mounts_learning_router() -> None:
    app = create_app()

    with TestClient(app) as client:
        res = client.get("/api/v1/learning/sessions")

    assert res.status_code == 200
    assert "sessions" in res.json()


def test_songguo_app_hides_experimental_routers_by_default(monkeypatch) -> None:
    monkeypatch.delenv("SONGGUO_ENABLE_EXPERIMENTAL_FEATURES", raising=False)
    app = create_app()

    with TestClient(app) as client:
        reminders = client.get("/api/v1/reminders/due")
        artifacts = client.post(
            "/api/v1/learning-artifacts/diagram",
            json={
                "child_id": "child_001",
                "question_text": "36 x 5 = ?",
                "knowledge_point": "two_digit_times_one_digit",
            },
        )

    assert reminders.status_code == 404
    assert artifacts.status_code == 404


def test_songguo_app_mounts_experimental_routers_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("SONGGUO_ENABLE_EXPERIMENTAL_FEATURES", "1")
    app = create_app()

    with TestClient(app) as client:
        reminders = client.get("/api/v1/reminders/due")

    assert reminders.status_code == 200

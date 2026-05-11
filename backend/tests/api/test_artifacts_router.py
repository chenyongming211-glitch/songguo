from __future__ import annotations

import importlib

import pytest

pytest.importorskip("fastapi")

FastAPI = pytest.importorskip("fastapi").FastAPI
TestClient = pytest.importorskip("fastapi.testclient").TestClient

artifacts_router_module = importlib.import_module("songguo.backend.api.routers.learning_artifacts")
router = artifacts_router_module.router

from songguo.backend.services.learning.store import InMemoryLearningStore
from songguo.backend.services.wechat import WechatService


def _build_app(tmp_path) -> FastAPI:
    app = FastAPI()
    artifacts_router_module.get_artifact_root = lambda: tmp_path
    artifacts_router_module.get_learning_store = lambda: InMemoryLearningStore()
    app.include_router(router, prefix="/api/v1/learning-artifacts")
    return app


def test_generate_static_diagram_returns_svg_artifact(tmp_path) -> None:
    with TestClient(_build_app(tmp_path)) as client:
        res = client.post(
            "/api/v1/learning-artifacts/diagram",
            json={
                "child_id": "child_001",
                "question_text": "36 x 5 = ?",
                "knowledge_point": "two_digit_times_one_digit",
            },
        )

    assert res.status_code == 200
    payload = res.json()
    assert payload["artifact_type"] == "svg"
    assert payload["status"] == "ready"
    assert payload["path"].endswith(".svg")
    assert (tmp_path / payload["path"]).exists()


def test_request_animation_returns_async_artifact_job(tmp_path) -> None:
    with TestClient(_build_app(tmp_path)) as client:
        res = client.post(
            "/api/v1/learning-artifacts/animation",
            json={
                "child_id": "child_001",
                "question_text": "36 x 5 = ?",
                "knowledge_point": "two_digit_times_one_digit",
            },
        )

    assert res.status_code == 200
    payload = res.json()
    assert payload["artifact_type"] == "animation"
    assert payload["status"] in {"queued", "ready"}
    assert payload["artifact_id"].startswith("art_")


def test_learning_artifacts_require_child_access_in_strict_mode(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SONGGUO_AUTH_MODE", "strict")
    store = InMemoryLearningStore()
    app = FastAPI()
    artifacts_router_module.get_artifact_root = lambda: tmp_path
    artifacts_router_module.get_learning_store = lambda: store
    app.include_router(router, prefix="/api/v1/learning-artifacts")

    with TestClient(app) as client:
        missing = client.post(
            "/api/v1/learning-artifacts/diagram",
            json={
                "child_id": "child_001",
                "question_text": "36 x 5 = ?",
                "knowledge_point": "two_digit_times_one_digit",
            },
        )
        allowed = client.post(
            "/api/v1/learning-artifacts/diagram",
            headers={"X-Session-Token": WechatService()._sign_session("openid_001")},
            json={
                "child_id": "child_openid_001",
                "question_text": "36 x 5 = ?",
                "knowledge_point": "two_digit_times_one_digit",
            },
        )

    assert missing.status_code == 401
    assert allowed.status_code == 200

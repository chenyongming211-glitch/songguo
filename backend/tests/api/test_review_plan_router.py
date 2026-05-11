from __future__ import annotations

import importlib

import pytest

pytest.importorskip("fastapi")

FastAPI = pytest.importorskip("fastapi").FastAPI
TestClient = pytest.importorskip("fastapi.testclient").TestClient

parent_router_module = importlib.import_module("songguo.backend.api.routers.parent_reports")
router = parent_router_module.router

from songguo.backend.services.learning.service import LearningService
from songguo.backend.services.learning.store import InMemoryLearningStore


def _build_app() -> FastAPI:
    app = FastAPI()
    store = InMemoryLearningStore()
    service = LearningService(store=store)
    created = service.create_session(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="36 x 5 = ?",
    )
    service.submit_attempt(created.session_id, child_answer="360")
    parent_router_module.get_learning_store = lambda: store
    app.include_router(router, prefix="/api/v1/parent")
    return app


def test_review_plan_endpoint() -> None:
    with TestClient(_build_app()) as client:
        res = client.get("/api/v1/parent/children/child_001/review-plan?scope=monthly")

    assert res.status_code == 200
    payload = res.json()
    assert payload["scope"] == "monthly"
    assert payload["items"][0]["knowledge_point_label"] == "两位数乘一位数"


def test_review_plan_endpoint_accepts_yearly_scope() -> None:
    with TestClient(_build_app()) as client:
        res = client.get("/api/v1/parent/children/child_001/review-plan?scope=yearly")

    assert res.status_code == 200
    assert res.json()["scope"] == "yearly"

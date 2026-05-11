from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib

import pytest

pytest.importorskip("fastapi")

FastAPI = pytest.importorskip("fastapi").FastAPI
TestClient = pytest.importorskip("fastapi.testclient").TestClient

parent_router_module = importlib.import_module("songguo.backend.api.routers.parent_reports")
router = parent_router_module.router

from songguo.backend.services.learning.ai_engine import DeterministicFallbackProvider
from songguo.backend.services.learning.models import LearningSummary
from songguo.backend.services.learning.service import LearningService
from songguo.backend.services.learning.store import InMemoryLearningStore
from songguo.backend.services.wechat import WechatService


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


def test_parent_weekly_report_endpoint() -> None:
    with TestClient(_build_app()) as client:
        res = client.get("/api/v1/parent/children/child_001/weekly-report")

    assert res.status_code == 200
    payload = res.json()
    assert payload["child_id"] == "child_001"
    assert payload["wrong_question_count"] == 1
    assert payload["common_misconceptions"] == ["treated_x5_like_x10"]


def test_parent_session_feedback_endpoint_returns_single_session_feedback() -> None:
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

    with TestClient(app) as client:
        res = client.get(
            f"/api/v1/parent/children/child_001/session-feedback?session_id={created.session_id}"
        )

    assert res.status_code == 200
    payload = res.json()
    assert payload["child_id"] == "child_001"
    assert payload["session_id"] == created.session_id
    assert payload["question_text"] == "36 x 5 = ?"
    assert payload["knowledge_point_label"] == "两位数乘一位数"
    assert payload["main_error_reason"] == "treated_x5_like_x10"
    assert payload["main_error_reason_label"] == "把乘以 5 当成乘以 10"
    assert "360" in payload["evidence"]
    assert payload["next_practice_count"] == 3
    assert payload["answer_unlocked"] is False


def test_parent_session_feedback_for_submission_tutor_keeps_full_item_evidence() -> None:
    app = FastAPI()
    store = InMemoryLearningStore()
    service = LearningService(store=store)
    created = service.create_submission(
        child_id="child_001",
        subject="math",
        grade=3,
        source_type="text",
        raw_text="36颗松果平均分给5只松鼠，每只最多分几颗，还剩几颗？\n孩子答案：每只6颗，还剩6颗",
    )
    confirmed = service.confirm_submission(created.submission_id)
    session_id = confirmed.active_tutor_session_id
    assert session_id is not None

    store.append_event(
        session_id=session_id,
        child_id="child_001",
        event_type="child.attempt_submitted",
        payload={"child_answer": "每只7颗，还剩1颗"},
    )
    store.append_event(
        session_id=session_id,
        child_id="child_001",
        event_type="attempt.evaluated",
        payload={"correct": True},
    )
    parent_router_module.get_learning_store = lambda: store
    app.include_router(router, prefix="/api/v1/parent")

    with TestClient(app) as client:
        res = client.get(
            f"/api/v1/parent/children/child_001/session-feedback?session_id={session_id}"
        )

    assert res.status_code == 200
    payload = res.json()
    assert "提交时孩子答“每只6颗，还剩6颗”" in payload["evidence"]
    assert "系统判为错误" in payload["evidence"]
    assert "参考答案是“每只7颗，还剩1颗”" in payload["evidence"]
    assert "陪练后孩子改答“每只7颗，还剩1颗”" in payload["evidence"]


def test_parent_wrong_questions_for_submission_exposes_tutor_session_id() -> None:
    app = FastAPI()
    store = InMemoryLearningStore()
    service = LearningService(store=store)
    created = service.create_submission(
        child_id="child_001",
        subject="math",
        grade=3,
        source_type="text",
        raw_text="36颗松果平均分给5只松鼠，每只最多分几颗，还剩几颗？\n孩子答案：每只6颗，还剩6颗",
    )
    confirmed = service.confirm_submission(created.submission_id)
    assert confirmed.active_tutor_session_id is not None
    parent_router_module.get_learning_store = lambda: store
    app.include_router(router, prefix="/api/v1/parent")

    with TestClient(app) as client:
        res = client.get("/api/v1/parent/children/child_001/wrong-questions")

    assert res.status_code == 200
    payload = res.json()
    assert payload["items"][0]["session_id"] == confirmed.active_tutor_session_id


def test_parent_learning_deposit_endpoint_returns_structured_learning_asset() -> None:
    app = FastAPI()
    store = InMemoryLearningStore()
    service = LearningService(store=store, ai_provider=DeterministicFallbackProvider())
    created = service.create_session(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="学校组织三年级学生春游，一共有4个班，每班32人。如果每辆大巴车限坐45人，那么至少需要多少辆大巴车？",
    )
    service.submit_attempt(created.session_id, child_answer="128")
    service.submit_attempt(created.session_id, child_answer="2")
    service.submit_attempt(created.session_id, child_answer="3")
    parent_router_module.get_learning_store = lambda: store
    app.include_router(router, prefix="/api/v1/parent")

    with TestClient(app) as client:
        res = client.get(
            f"/api/v1/parent/children/child_001/learning-deposit?session_id={created.session_id}"
        )

    assert res.status_code == 200
    payload = res.json()
    assert payload["child_id"] == "child_001"
    assert payload["session_id"] == created.session_id
    assert payload["question_record"]["knowledge_point"] == "capacity_round_up"
    assert payload["question_record"]["knowledge_point_label"] == "限载进一问题"
    assert payload["mistake_record"]["main_error_reason"] == "math_capacity_ignored_remainder_round_up"
    assert payload["mistake_record"]["student_answer"] == "2"
    assert 1 <= len(payload["practice_records"]) <= 3
    assert payload["student_memory"]["top_weaknesses"][0]["knowledge_point"] == "capacity_round_up"
    assert "math_capacity" not in payload["parent_summary"]


def test_parent_session_feedback_endpoint_rejects_child_mismatch() -> None:
    app = FastAPI()
    store = InMemoryLearningStore()
    service = LearningService(store=store)
    created = service.create_session(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="36 x 5 = ?",
    )
    parent_router_module.get_learning_store = lambda: store
    app.include_router(router, prefix="/api/v1/parent")

    with TestClient(app) as client:
        res = client.get(
            f"/api/v1/parent/children/child_999/session-feedback?session_id={created.session_id}"
        )

    assert res.status_code == 404


def test_parent_summary_draft_endpoint_is_not_full_weekly_report_contract() -> None:
    with TestClient(_build_app()) as client:
        res = client.get("/api/v1/parent/children/child_001/summary-draft?scope=weekly")

    assert res.status_code == 200
    payload = res.json()
    assert payload["child_id"] == "child_001"
    assert payload["scope"] == "weekly"
    assert payload["wrong_question_count"] == 1
    assert payload["next_actions"]
    assert "两位数乘一位数" in payload["parent_summary"]


def test_parent_summary_rollup_endpoint_and_weekly_report_prefers_saved_summary() -> None:
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

    with TestClient(app) as client:
        rolled_up = client.post("/api/v1/parent/children/child_001/summary-rollup?scope=weekly")

    assert rolled_up.status_code == 200
    assert store.list_learning_summaries(child_id="child_001", scope="weekly")

    store.save_learning_summary(
        LearningSummary(
            child_id="child_001",
            scope="weekly",
            session_count=1,
            wrong_question_count=1,
            completed_session_count=0,
            top_knowledge_points=["two_digit_times_one_digit"],
            common_misconceptions=["treated_x5_like_x10"],
            parent_summary="这是已经沉淀的周级摘要，不应被实时草稿覆盖。",
            next_actions=["下周继续练同类题"],
        )
    )

    with TestClient(app) as client:
        report = client.get("/api/v1/parent/children/child_001/weekly-report")

    assert report.status_code == 200
    assert report.json()["parent_summary"] == "这是已经沉淀的周级摘要，不应被实时草稿覆盖。"


def test_parent_children_profile_endpoints() -> None:
    app = FastAPI()
    store = InMemoryLearningStore()
    parent_router_module.get_learning_store = lambda: store
    app.include_router(router, prefix="/api/v1/parent")

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/parent/children",
            json={"child_id": "child_001", "name": "小明", "grade": 3, "term_label": "2026春季"},
        )
        listed = client.get("/api/v1/parent/children")

    assert created.status_code == 200
    assert listed.status_code == 200
    assert created.json()["name"] == "小明"
    assert listed.json()["children"][0]["child_id"] == "child_001"


def test_parent_endpoints_filter_and_authorize_bound_children() -> None:
    app = FastAPI()
    store = InMemoryLearningStore()
    parent_router_module.get_learning_store = lambda: store
    app.include_router(router, prefix="/api/v1/parent")
    token = WechatService()._sign_session("openid_001")

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/parent/children",
            headers={"X-Session-Token": token},
            json={"child_id": "child_001", "name": "小明", "grade": 3},
        )
        listed = client.get("/api/v1/parent/children", headers={"X-Session-Token": token})
        forbidden = client.get(
            "/api/v1/parent/children/child_999/weekly-report",
            headers={"X-Session-Token": token},
        )

    assert created.status_code == 200
    assert listed.status_code == 200
    assert [item["child_id"] for item in listed.json()["children"]] == ["child_001"]
    assert forbidden.status_code == 403


def test_parent_children_requires_session_token_in_strict_auth_mode(monkeypatch) -> None:
    monkeypatch.setenv("SONGGUO_AUTH_MODE", "strict")
    app = FastAPI()
    store = InMemoryLearningStore()
    parent_router_module.get_learning_store = lambda: store
    app.include_router(router, prefix="/api/v1/parent")

    with TestClient(app) as client:
        create_res = client.post(
            "/api/v1/parent/children",
            json={"child_id": "child_001", "name": "小明", "grade": 3},
        )
        list_res = client.get("/api/v1/parent/children")

    assert create_res.status_code == 401
    assert list_res.status_code == 401


def test_parent_children_strict_mode_lists_only_bound_children(monkeypatch) -> None:
    monkeypatch.setenv("SONGGUO_AUTH_MODE", "strict")
    app = FastAPI()
    store = InMemoryLearningStore()
    parent_router_module.get_learning_store = lambda: store
    app.include_router(router, prefix="/api/v1/parent")
    openid = "openid_001"
    child_id = f"child_{openid}"
    token = WechatService()._sign_session(openid)

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/parent/children",
            headers={"X-Session-Token": token},
            json={"child_id": child_id, "name": "小明", "grade": 3},
        )
        listed = client.get("/api/v1/parent/children", headers={"X-Session-Token": token})
        forbidden = client.get(
            "/api/v1/parent/children/child_other/weekly-report",
            headers={"X-Session-Token": token},
        )

    assert created.status_code == 200
    assert listed.status_code == 200
    assert [item["child_id"] for item in listed.json()["children"]] == [child_id]
    assert forbidden.status_code == 403


def test_parent_wrong_questions_endpoint() -> None:
    with TestClient(_build_app()) as client:
        res = client.get("/api/v1/parent/children/child_001/wrong-questions")

    assert res.status_code == 200
    payload = res.json()
    assert len(payload["items"]) == 1
    assert payload["items"][0]["highest_hint_level"] == 2
    assert payload["items"][0]["knowledge_point_label"] == "两位数乘一位数"
    assert payload["items"][0]["misconception_label"] == "把乘以 5 当成乘以 10"


def test_parent_safety_events_endpoint() -> None:
    app = FastAPI()
    store = InMemoryLearningStore()
    store.record_safety_event(
        session_id="s_001",
        child_id="child_001",
        event_type="safety.blocked",
        input_text="答案是 180。",
        blocked_text="答案是 180。",
        reason="exact_answer_leak",
    )
    parent_router_module.get_learning_store = lambda: store
    app.include_router(router, prefix="/api/v1/parent")

    with TestClient(app) as client:
        res = client.get("/api/v1/parent/children/child_001/safety-events")

    assert res.status_code == 200
    payload = res.json()
    assert payload["child_id"] == "child_001"
    assert payload["items"][0]["reason"] == "exact_answer_leak"


def test_parent_learning_memory_endpoint() -> None:
    with TestClient(_build_app()) as client:
        res = client.get("/api/v1/parent/children/child_001/learning-memory")

    assert res.status_code == 200
    payload = res.json()
    assert payload["child_id"] == "child_001"
    assert payload["top_weaknesses"][0]["knowledge_point"] == "two_digit_times_one_digit"
    assert payload["top_weaknesses"][0]["mastery_score"] < 80
    assert payload["top_weaknesses"][0]["risk_level"] in {"medium", "high"}


def test_parent_learning_memory_endpoint_accepts_scope() -> None:
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
    old_time = datetime.now(timezone.utc) - timedelta(days=40)
    store.wrong_questions["child_001"][0] = store.wrong_questions["child_001"][0].model_copy(
        update={"created_at": old_time, "updated_at": old_time}
    )
    parent_router_module.get_learning_store = lambda: store
    app.include_router(router, prefix="/api/v1/parent")

    with TestClient(app) as client:
        weekly = client.get("/api/v1/parent/children/child_001/learning-memory?scope=weekly")
        term = client.get("/api/v1/parent/children/child_001/learning-memory?scope=term")

    assert weekly.status_code == 200
    assert term.status_code == 200
    assert weekly.json()["top_weaknesses"] == []
    assert term.json()["top_weaknesses"][0]["wrong_count"] == 1

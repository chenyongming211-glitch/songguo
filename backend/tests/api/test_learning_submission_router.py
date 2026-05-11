from __future__ import annotations

import importlib

import pytest

pytest.importorskip("fastapi")

FastAPI = pytest.importorskip("fastapi").FastAPI
TestClient = pytest.importorskip("fastapi.testclient").TestClient

learning_router_module = importlib.import_module("songguo.backend.api.routers.learning")
router = learning_router_module.router

from songguo.backend.services.learning.llm_session_runner import LLMSessionRunner
from songguo.backend.services.learning.math_structuring import ProblemAnalysis
from songguo.backend.services.learning.service import LearningService
from songguo.backend.services.learning.store import InMemoryLearningStore
from songguo.backend.services.wechat import WechatService


def _build_app() -> FastAPI:
    app = FastAPI()
    store = InMemoryLearningStore()
    service = LearningService(
        store=store,
        session_runner=_fake_runner(),
        math_gateway=_fixture_gateway(),
        agent_runtime="langgraph",
    )
    learning_router_module.get_learning_service = lambda: service
    app.include_router(router, prefix="/api/v1/learning")
    return app


def test_create_submission_api_parses_items_before_confirmation() -> None:
    with TestClient(_build_app()) as client:
        res = client.post(
            "/api/v1/learning/submissions",
            json={
                "child_id": "child_001",
                "subject": "math",
                "grade": 4,
                "source_type": "text",
                "raw_text": """
                1. 36 颗松果平均分给 5 只小松鼠，每只几颗，还剩几颗？
                孩子答案：每只 6 颗，还剩 6 颗

                2. 48 ÷ 6 = ?
                孩子答案：8
                """,
            },
        )

    assert res.status_code == 200
    payload = res.json()
    assert payload["submission_id"].startswith("sub_")
    assert payload["status"] == "intake_pending"
    assert payload["item_count"] == 2
    assert payload["correct_count"] == 0
    assert payload["wrong_count"] == 0
    assert len(payload["items"]) == 2
    assert payload["items"][0]["judge_result"] == "unknown"
    assert payload["mastery_evidence"] == []
    assert payload["tutor_queue"] == []
    assert payload["active_tutor_session"] is None


def test_get_submission_api_returns_snapshot() -> None:
    with TestClient(_build_app()) as client:
        created = client.post(
            "/api/v1/learning/submissions",
            json={
                "child_id": "child_001",
                "subject": "math",
                "grade": 4,
                "source_type": "text",
                "raw_text": "36 颗松果平均分给 5 只小松鼠，每只几颗，还剩几颗？\n孩子答案：每只 6 颗，还剩 6 颗",
            },
        ).json()
        client.post(
            f"/api/v1/learning/submissions/{created['submission_id']}/confirm",
            json={"raw_text": "36 颗松果平均分给 5 只小松鼠，每只几颗，还剩几颗？\n孩子答案：每只 6 颗，还剩 6 颗"},
        )
        res = client.get(
            f"/api/v1/learning/submissions/{created['submission_id']}?child_id=child_001"
        )

    assert res.status_code == 200
    payload = res.json()
    assert payload["submission_id"] == created["submission_id"]
    assert payload["items"][0]["judge_result"] == "wrong"
    assert payload["tutor_queue"][0]["status"] == "active"


def test_confirm_submission_api_deposits_items_and_starts_wrong_tutor() -> None:
    with TestClient(_build_app()) as client:
        created = client.post(
            "/api/v1/learning/submissions",
            json={
                "child_id": "child_001",
                "subject": "math",
                "grade": 4,
                "source_type": "text",
                "raw_text": """
                1. 36 颗松果平均分给 5 只小松鼠，每只几颗，还剩几颗？
                孩子答案：每只 6 颗，还剩 6 颗

                2. 48 ÷ 6 = ?
                孩子答案：8
                """,
            },
        ).json()
        res = client.post(
            f"/api/v1/learning/submissions/{created['submission_id']}/confirm",
            json={"raw_text": created["items"][0]["question_text"]},
        )

    assert res.status_code == 200
    payload = res.json()
    assert payload["status"] == "tutoring"
    assert payload["item_count"] == 2
    assert payload["correct_count"] == 1
    assert payload["wrong_count"] == 1
    assert len(payload["mastery_evidence"]) == 2
    assert len(payload["tutor_queue"]) == 1
    assert payload["active_tutor_session"]["session_id"].startswith("s_")
    assert payload["tutor_queue"][0]["tutor_session_id"] == payload["active_tutor_session"]["session_id"]


def test_submission_tutor_next_and_attempt_complete_submission() -> None:
    with TestClient(_build_app()) as client:
        created = client.post(
            "/api/v1/learning/submissions",
            json={
                "child_id": "child_001",
                "subject": "math",
                "grade": 4,
                "source_type": "text",
                "raw_text": "36 颗松果平均分给 5 只小松鼠，每只几颗，还剩几颗？\n孩子答案：每只 6 颗，还剩 6 颗",
            },
        ).json()
        client.post(
            f"/api/v1/learning/submissions/{created['submission_id']}/confirm",
            json={"raw_text": "36 颗松果平均分给 5 只小松鼠，每只几颗，还剩几颗？\n孩子答案：每只 6 颗，还剩 6 颗"},
        )
        next_res = client.post(
            f"/api/v1/learning/submissions/{created['submission_id']}/tutor/next"
        )
        attempt_res = client.post(
            f"/api/v1/learning/submissions/{created['submission_id']}/tutor/attempt",
            json={"child_answer": "每只7颗，还剩1颗"},
        )

    assert next_res.status_code == 200
    assert next_res.json()["active_tutor_session"]["session_id"].startswith("s_")
    assert attempt_res.status_code == 200
    payload = attempt_res.json()
    assert payload["attempt"]["correct"] is True
    assert payload["submission"]["status"] == "completed"
    assert payload["submission"]["active_tutor_session"] is None


def test_completed_submission_tutor_attempt_returns_child_safe_state() -> None:
    with TestClient(_build_app()) as client:
        created = client.post(
            "/api/v1/learning/submissions",
            json={
                "child_id": "child_001",
                "subject": "math",
                "grade": 4,
                "source_type": "text",
                "raw_text": "36 颗松果平均分给 5 只小松鼠，每只几颗，还剩几颗？\n孩子答案：每只 6 颗，还剩 6 颗",
            },
        ).json()
        client.post(
            f"/api/v1/learning/submissions/{created['submission_id']}/confirm",
            json={"raw_text": "36 颗松果平均分给 5 只小松鼠，每只几颗，还剩几颗？\n孩子答案：每只 6 颗，还剩 6 颗"},
        )
        client.post(f"/api/v1/learning/submissions/{created['submission_id']}/tutor/next")
        client.post(
            f"/api/v1/learning/submissions/{created['submission_id']}/tutor/attempt",
            json={"child_answer": "每只7颗，还剩1颗"},
        )

        res = client.post(
            f"/api/v1/learning/submissions/{created['submission_id']}/tutor/attempt",
            json={"child_answer": "暂时先不练了"},
        )

    assert res.status_code == 409
    assert res.json()["detail"] == "本次错题陪练已经结束，请返回本次总结。"


def test_get_submission_api_rejects_child_id_mismatch() -> None:
    with TestClient(_build_app()) as client:
        created = client.post(
            "/api/v1/learning/submissions",
            json={
                "child_id": "child_001",
                "subject": "math",
                "grade": 4,
                "source_type": "text",
                "raw_text": "48 ÷ 6 = ?\n孩子答案：8",
            },
        ).json()
        res = client.get(
            f"/api/v1/learning/submissions/{created['submission_id']}?child_id=child_999"
        )

    assert res.status_code == 403


def test_submission_api_rejects_unbound_openid_in_strict_mode(monkeypatch) -> None:
    monkeypatch.setenv("SONGGUO_AUTH_MODE", "strict")
    app = _build_app()
    token_owner = WechatService()._sign_session("openid_001")
    token_other = WechatService()._sign_session("openid_002")

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/learning/submissions",
            headers={"X-Session-Token": token_owner},
            json={
                "child_id": "child_openid_001",
                "subject": "math",
                "grade": 4,
                "source_type": "text",
                "raw_text": "48 ÷ 6 = ?\n孩子答案：8",
            },
        )
        read_by_other = client.get(
            f"/api/v1/learning/submissions/{created.json()['submission_id']}?child_id=child_openid_001",
            headers={"X-Session-Token": token_other},
        )

    assert created.status_code == 200
    assert read_by_other.status_code == 403


def _fixture_gateway():
    class FixtureMathGateway:
        def analyze(self, *, question_text: str, grade: int, subject: str) -> ProblemAnalysis:
            if "36" in question_text:
                return ProblemAnalysis(
                    subject=subject,
                    grade=grade,
                    problem_type="division_with_remainder",
                    knowledge_points=["有余数除法"],
                    target="每只几颗，还剩几颗",
                    final_answer="每只7颗，还剩1颗",
                    confidence=0.96,
                    source="fixture",
                    solution_steps=[
                        {"id": "step_1", "goal": "先试商", "expression": "36 ÷ 5", "result": "7余1"}
                    ],
                    common_misconceptions=[
                        {"tag": "remainder_not_less_than_divisor", "description": "余数没有小于除数"}
                    ],
                    key_points=[
                        {
                            "id": "kp_division_remainder",
                            "name": "求商和余数",
                            "teaching_goal": "理解平均分时商和余数的含义",
                            "release_stage": "HINT_STEP_1",
                            "unlock_condition": "question_started",
                            "child_prompt": "36 除以 5，先想一想商可能是几？",
                            "expected_child_response": ["7", "7余1"],
                            "forbidden_content": ["每只7颗，还剩1颗"],
                        }
                    ],
                )
            return ProblemAnalysis(
                subject=subject,
                grade=grade,
                problem_type="division",
                knowledge_points=["表内除法"],
                target="计算商",
                final_answer="8",
                confidence=0.98,
                source="fixture",
                solution_steps=[
                    {"id": "step_1", "goal": "计算除法", "expression": "48 ÷ 6", "result": "8"}
                ],
                key_points=[
                    {
                        "id": "kp_division",
                        "name": "表内除法",
                        "teaching_goal": "计算除法结果",
                        "release_stage": "HINT_STEP_1",
                        "unlock_condition": "question_started",
                        "child_prompt": "48 里面有几个 6？",
                        "expected_child_response": ["8"],
                        "forbidden_content": ["8"],
                    }
                ],
            )

    return FixtureMathGateway()


def _fake_runner() -> LLMSessionRunner:
    class FakeRunner:
        def start(self, **kwargs):
            from songguo.backend.services.learning.llm_session_runner import LLMSessionOutput

            return LLMSessionOutput(child_message="先说说你看到题目里要平均分给几只？")

        def run(self, **kwargs):
            from songguo.backend.services.learning.llm_session_runner import LLMSessionOutput

            return LLMSessionOutput(child_message="我们先核对商和余数的关系。")

    return FakeRunner()

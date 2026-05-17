from __future__ import annotations

import importlib

import pytest

pytest.importorskip("fastapi")

FastAPI = pytest.importorskip("fastapi").FastAPI
TestClient = pytest.importorskip("fastapi.testclient").TestClient

learning_router_module = importlib.import_module("songguo.backend.api.routers.learning")
router = learning_router_module.router

from songguo.backend.services.learning.llm_session_runner import LLMSessionRunner
from songguo.backend.services.learning.intent_router import IntentRoutingDecision
from songguo.backend.services.learning.math_structuring import ProblemAnalysis
from songguo.backend.services.learning.photo_review import OCRDraft, OCRItemDraft
from songguo.backend.services.learning.service import LearningService
from songguo.backend.services.learning.store import InMemoryLearningStore
from songguo.backend.services.wechat import WechatService


def _build_app(intent_router=None, visual_fallback_ocr_provider=None) -> FastAPI:
    app = FastAPI()
    store = InMemoryLearningStore()
    service = LearningService(
        store=store,
        session_runner=_fake_runner(),
        math_gateway=_fixture_gateway(),
        agent_runtime="langgraph",
        intent_router=intent_router,
        visual_fallback_ocr_provider=visual_fallback_ocr_provider,
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


def test_create_submission_api_preserves_photo_item_bboxes() -> None:
    with TestClient(_build_app()) as client:
        res = client.post(
            "/api/v1/learning/submissions",
            json={
                "child_id": "child_001",
                "subject": "auto",
                "grade": 4,
                "source_type": "photo",
                "raw_text": "48 ÷ 6 = ?\n孩子答案：8",
                "image_refs": ["artifact://photo_001"],
                "draft_items": [
                    {
                        "item_index": 1,
                        "bbox": {"x": 60, "y": 80, "width": 880, "height": 720},
                    }
                ],
            },
        )
        created = res.json()
        confirmed = client.post(
            f"/api/v1/learning/submissions/{created['submission_id']}/confirm",
            json={"raw_text": created["items"][0]["question_text"]},
        )

    assert res.status_code == 200
    assert created["items"][0]["bbox"] == {"x": 60, "y": 80, "width": 880, "height": 720}
    assert confirmed.status_code == 200
    assert confirmed.json()["items"][0]["bbox"] == {"x": 60, "y": 80, "width": 880, "height": 720}


def test_create_submission_api_returns_agent_route_result_for_auto_subject() -> None:
    with TestClient(
        _build_app(
            intent_router=_fake_intent_router(
                IntentRoutingDecision(
                    subject="english",
                    task_type="grammar_fix",
                    user_intent="check_answer",
                    confidence=0.9,
                    evidence=["题干是英文时态填空"],
                    needs_clarification=False,
                    route_to="english_basic_tutor",
                )
            )
        )
    ) as client:
        res = client.post(
            "/api/v1/learning/submissions",
            json={
                "child_id": "child_001",
                "subject": "auto",
                "grade": 4,
                "source_type": "text",
                "raw_text": "Choose the correct tense: He ____ to school yesterday.\n孩子答案：go",
            },
        )

    assert res.status_code == 200
    payload = res.json()
    assert payload["subject"] == "english"
    assert payload["detected_subject"] == "english"
    assert payload["detected_task_type"] == "grammar_fix"
    assert payload["detected_intent"] == "check_answer"
    assert payload["subject_confidence"] == 0.9
    assert payload["route_to"] == "english_basic_tutor"
    assert payload["routing_evidence"] == ["题干是英文时态填空"]
    assert payload["needs_clarification"] is False


def test_confirm_submission_api_records_correct_english_rubric() -> None:
    with TestClient(
        _build_app(
            intent_router=_fake_intent_router(
                IntentRoutingDecision(
                    subject="english",
                    task_type="grammar_fix",
                    user_intent="check_answer",
                    confidence=0.91,
                    evidence=["题干是英文时态填空"],
                    needs_clarification=False,
                    route_to="english_basic_tutor",
                )
            )
        )
    ) as client:
        created = client.post(
            "/api/v1/learning/submissions",
            json={
                "child_id": "child_001",
                "subject": "auto",
                "grade": 4,
                "source_type": "text",
                "raw_text": "Choose the correct tense: He ____ to school yesterday.\n孩子答案：went",
            },
        ).json()
        res = client.post(
            f"/api/v1/learning/submissions/{created['submission_id']}/confirm",
            json={"raw_text": created["items"][0]["question_text"]},
        )

    assert res.status_code == 200
    payload = res.json()
    assert payload["status"] == "completed"
    assert payload["correct_count"] == 1
    assert payload["wrong_count"] == 0
    assert payload["active_tutor_session"] is None
    item = payload["items"][0]
    assert item["judge_result"] == "correct"
    assert item["rubric_outcome"] == "correct"
    assert "时间线索" in item["rubric_feedback"]
    assert item["rubric_scores"]["form_accuracy"] == 2
    assert item["misconception_tag"] is None


def test_confirm_submission_api_queues_wrong_english_with_rubric_feedback() -> None:
    with TestClient(
        _build_app(
            intent_router=_fake_intent_router(
                IntentRoutingDecision(
                    subject="english",
                    task_type="grammar_fix",
                    user_intent="check_answer",
                    confidence=0.91,
                    evidence=["题干是英文时态填空"],
                    needs_clarification=False,
                    route_to="english_basic_tutor",
                )
            )
        )
    ) as client:
        created = client.post(
            "/api/v1/learning/submissions",
            json={
                "child_id": "child_001",
                "subject": "auto",
                "grade": 4,
                "source_type": "text",
                "raw_text": "Choose the correct tense: He ____ to school yesterday.\n孩子答案：go",
            },
        ).json()
        res = client.post(
            f"/api/v1/learning/submissions/{created['submission_id']}/confirm",
            json={"raw_text": created["items"][0]["question_text"]},
        )

    assert res.status_code == 200
    payload = res.json()
    assert payload["status"] == "tutoring"
    assert payload["correct_count"] == 0
    assert payload["wrong_count"] == 1
    assert payload["active_tutor_session"] is not None
    item = payload["items"][0]
    assert item["judge_result"] == "wrong"
    assert item["rubric_outcome"] == "wrong"
    assert "时间线索" in item["rubric_feedback"]
    assert item["rubric_scores"]["form_accuracy"] == 0
    assert item["misconception_tag"] == "english_past_tense_missing"


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


def test_confirm_submission_api_can_defer_wrong_tutor_start() -> None:
    with TestClient(_build_app()) as client:
        created = client.post(
            "/api/v1/learning/submissions",
            json={
                "child_id": "child_001",
                "subject": "math",
                "grade": 4,
                "source_type": "photo",
                "raw_text": "36 颗松果平均分给 5 只小松鼠，每只几颗，还剩几颗？\n孩子答案：每只 6 颗，还剩 6 颗",
            },
        ).json()
        res = client.post(
            f"/api/v1/learning/submissions/{created['submission_id']}/confirm",
            json={
                "raw_text": created["items"][0]["question_text"],
                "start_tutor": False,
            },
        )
        next_res = client.post(
            f"/api/v1/learning/submissions/{created['submission_id']}/tutor/next"
        )

    assert res.status_code == 200
    payload = res.json()
    assert payload["status"] == "tutoring"
    assert payload["wrong_count"] == 1
    assert payload["active_tutor_session"] is None
    assert payload["tutor_queue"][0]["status"] == "pending"
    assert payload["tutor_queue"][0]["tutor_session_id"] is None

    assert next_res.status_code == 200
    started_payload = next_res.json()
    assert started_payload["active_tutor_session"]["session_id"].startswith("s_")
    assert started_payload["tutor_queue"][0]["status"] == "active"


def test_confirm_photo_submission_returns_partial_result_with_visual_fallback_pending(tmp_path) -> None:
    image_path = tmp_path / "homework.jpg"
    image_path.write_bytes(b"not-a-real-image")
    with TestClient(_build_app()) as client:
        created = client.post(
            "/api/v1/learning/submissions",
            json={
                "child_id": "child_001",
                "subject": "math",
                "grade": 4,
                "source_type": "photo",
                "raw_text": "1. 48 ÷ 6 = ?\n孩子答案：8\n\n2. 36 ÷ 5 = ?",
                "image_refs": [str(image_path)],
                "draft_items": [
                    {
                        "item_index": 2,
                        "bbox": {"x": 100, "y": 500, "width": 800, "height": 300},
                    }
                ],
            },
        ).json()
        res = client.post(
            f"/api/v1/learning/submissions/{created['submission_id']}/confirm",
            json={"raw_text": created["items"][0]["question_text"], "start_tutor": False},
        )

    assert res.status_code == 200
    payload = res.json()
    assert payload["correct_count"] == 1
    assert payload["needs_manual_confirm_count"] == 1
    assert payload["pending_visual_fallback_count"] == 1
    assert payload["visual_fallback_active"] is True
    pending_item = payload["items"][1]
    assert pending_item["judge_result"] == "needs_manual_confirm"
    assert pending_item["visual_fallback_status"] == "pending"
    assert pending_item["visual_fallback_reason"] == "missing_question_or_answer"
    assert "其他题" in pending_item["visual_fallback_message"]


def test_confirm_photo_submission_explains_ocr_answer_missing_fallback(tmp_path) -> None:
    image_path = tmp_path / "homework.jpg"
    image_path.write_bytes(b"not-a-real-image")
    with TestClient(_build_app()) as client:
        created = client.post(
            "/api/v1/learning/submissions",
            json={
                "child_id": "child_001",
                "subject": "math",
                "grade": 4,
                "source_type": "photo",
                "raw_text": "1. 48 ÷ 6 = ?",
                "image_refs": [str(image_path)],
                "draft_items": [
                    {
                        "item_index": 1,
                        "bbox": {"x": 100, "y": 120, "width": 800, "height": 260},
                        "ocr_action": "RecognizeEduPaperCut",
                        "ocr_source": "aliyun_edu_paper_cut",
                        "quality_warnings": ["answer_missing"],
                    }
                ],
            },
        ).json()
        res = client.post(
            f"/api/v1/learning/submissions/{created['submission_id']}/confirm",
            json={"raw_text": created["items"][0]["question_text"], "start_tutor": False},
        )

    assert res.status_code == 200
    payload = res.json()
    assert payload["pending_visual_fallback_count"] == 1
    pending_item = payload["items"][0]
    assert pending_item["visual_fallback_status"] == "pending"
    assert pending_item["visual_fallback_reason"] == "ocr_answer_missing"
    assert "缺少孩子答案" in pending_item["visual_fallback_message"]
    assert pending_item["display_status"] == "fallback_running"


def test_visual_fallback_step_updates_pending_photo_item(tmp_path) -> None:
    image_path = tmp_path / "homework.jpg"
    image_path.write_bytes(b"not-a-real-image")
    with TestClient(
        _build_app(visual_fallback_ocr_provider=_fake_visual_fallback_provider())
    ) as client:
        created = client.post(
            "/api/v1/learning/submissions",
            json={
                "child_id": "child_001",
                "subject": "math",
                "grade": 4,
                "source_type": "photo",
                "raw_text": "48 ÷ 6 = ?",
                "image_refs": [str(image_path)],
                "draft_items": [
                    {
                        "item_index": 1,
                        "bbox": {"x": 100, "y": 100, "width": 800, "height": 300},
                    }
                ],
            },
        ).json()
        confirmed = client.post(
            f"/api/v1/learning/submissions/{created['submission_id']}/confirm",
            json={"raw_text": created["items"][0]["question_text"], "start_tutor": False},
        ).json()
        res = client.post(
            f"/api/v1/learning/submissions/{created['submission_id']}/visual-fallback/step"
        )

    assert confirmed["pending_visual_fallback_count"] == 1
    assert res.status_code == 200
    payload = res.json()
    assert payload["pending_visual_fallback_count"] == 0
    assert payload["visual_fallback_active"] is False
    assert payload["correct_count"] == 1
    item = payload["items"][0]
    assert item["question_text"] == "48 ÷ 6 = ?"
    assert item["child_answer"] == "8"
    assert item["judge_result"] == "correct"
    assert item["visual_fallback_status"] == "done"


def test_visual_fallback_step_expands_zero_item_photo_submission(tmp_path) -> None:
    image_path = tmp_path / "whole-page.jpg"
    image_path.write_bytes(b"not-a-real-image")
    with TestClient(
        _build_app(visual_fallback_ocr_provider=_fake_multi_visual_fallback_provider())
    ) as client:
        created = client.post(
            "/api/v1/learning/submissions",
            json={
                "child_id": "child_001",
                "subject": "math",
                "grade": 4,
                "source_type": "photo",
                "raw_text": "图片文字没有被结构化成题目",
                "image_refs": [str(image_path)],
            },
        ).json()
        confirmed = client.post(
            f"/api/v1/learning/submissions/{created['submission_id']}/confirm",
            json={"raw_text": created["items"][0]["question_text"] if created["items"] else "", "start_tutor": False},
        ).json()
        res = client.post(
            f"/api/v1/learning/submissions/{created['submission_id']}/visual-fallback/step"
        )

    assert created["item_count"] == 0
    assert confirmed["item_count"] == 1
    assert confirmed["pending_visual_fallback_count"] == 1
    assert confirmed["items"][0]["visual_fallback_reason"] == "no_structured_items"
    assert res.status_code == 200
    payload = res.json()
    assert payload["item_count"] == 2
    assert payload["pending_visual_fallback_count"] == 0
    assert payload["correct_count"] == 1
    assert payload["wrong_count"] == 1
    assert [item["visual_fallback_status"] for item in payload["items"]] == ["done", "done"]


def test_confirm_photo_submission_judges_ten_photo_items_by_default() -> None:
    raw_text = "\n\n".join(
        f"{index}. 48 ÷ 6 = ?\n孩子答案：8"
        for index in range(1, 11)
    )
    with TestClient(_build_app()) as client:
        created = client.post(
            "/api/v1/learning/submissions",
            json={
                "child_id": "child_001",
                "subject": "math",
                "grade": 4,
                "source_type": "photo",
                "raw_text": raw_text,
                "image_refs": ["artifact://photo_001"],
            },
        ).json()
        confirmed = client.post(
            f"/api/v1/learning/submissions/{created['submission_id']}/confirm",
            json={"raw_text": raw_text, "start_tutor": False},
        ).json()

    assert confirmed["item_count"] == 10
    assert confirmed["correct_count"] == 10
    assert confirmed["pending_visual_fallback_count"] == 0
    assert confirmed["visual_fallback_active"] is False
    assert {item["judge_result"] for item in confirmed["items"]} == {"correct"}


def test_confirm_photo_submission_defers_extra_judgement_items(monkeypatch) -> None:
    monkeypatch.setenv("SONGGUO_PHOTO_SYNC_JUDGEMENT_LIMIT", "4")
    raw_text = "\n\n".join(
        f"{index}. 48 ÷ 6 = ?\n孩子答案：8"
        for index in range(1, 7)
    )
    with TestClient(_build_app()) as client:
        created = client.post(
            "/api/v1/learning/submissions",
            json={
                "child_id": "child_001",
                "subject": "math",
                "grade": 4,
                "source_type": "photo",
                "raw_text": raw_text,
                "image_refs": ["artifact://photo_001"],
            },
        ).json()
        confirmed = client.post(
            f"/api/v1/learning/submissions/{created['submission_id']}/confirm",
            json={"raw_text": raw_text, "start_tutor": False},
        ).json()
        stepped = client.post(
            f"/api/v1/learning/submissions/{created['submission_id']}/visual-fallback/step"
        ).json()

    assert confirmed["item_count"] == 6
    assert confirmed["status"] != "completed"
    assert confirmed["correct_count"] == 4
    assert confirmed["pending_visual_fallback_count"] == 2
    assert confirmed["items"][4]["visual_fallback_reason"] == "deferred_judgement"
    assert confirmed["items"][4]["visual_fallback_status"] == "pending"
    assert stepped["correct_count"] == 5
    assert stepped["pending_visual_fallback_count"] == 1
    assert stepped["items"][4]["visual_fallback_status"] == "done"


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


def _fake_visual_fallback_provider():
    class FakeVisualFallbackProvider:
        async def recognize_async(self, content, *, filename, region_hints=None):
            return OCRDraft(
                raw_text="48 ÷ 6 = ?\n孩子答案：8",
                question_text="48 ÷ 6 = ?",
                child_answer="8",
                confidence=0.96,
                needs_confirmation=False,
                items=[
                    OCRItemDraft(
                        item_index=1,
                        question_text="48 ÷ 6 = ?",
                        child_answer="8",
                        confidence=0.96,
                    )
                ],
                provider="fake_visual_ocr",
                model="fixture",
                source="test",
            )

    return FakeVisualFallbackProvider()


def _fake_multi_visual_fallback_provider():
    class FakeMultiVisualFallbackProvider:
        async def recognize_async(self, content, *, filename, region_hints=None):
            return OCRDraft(
                raw_text=(
                    "1. 48 ÷ 6 = ?\n孩子答案：8\n\n"
                    "2. 36 颗松果平均分给 5 只小松鼠，每只几颗，还剩几颗？\n"
                    "孩子答案：每只 6 颗，还剩 6 颗"
                ),
                confidence=0.95,
                needs_confirmation=False,
                items=[
                    OCRItemDraft(
                        item_index=1,
                        question_text="48 ÷ 6 = ?",
                        child_answer="8",
                        confidence=0.97,
                    ),
                    OCRItemDraft(
                        item_index=2,
                        question_text="36 颗松果平均分给 5 只小松鼠，每只几颗，还剩几颗？",
                        child_answer="每只 6 颗，还剩 6 颗",
                        confidence=0.93,
                    ),
                ],
                provider="fake_visual_ocr",
                model="fixture",
                source="test",
            )

    return FakeMultiVisualFallbackProvider()


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


def _fake_intent_router(decision: IntentRoutingDecision):
    class FakeIntentRouter:
        def route(self, context):
            return decision

    return FakeIntentRouter()

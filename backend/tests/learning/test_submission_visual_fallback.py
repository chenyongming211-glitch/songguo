from __future__ import annotations

import cv2
import numpy as np
import pytest

from songguo.backend.services.learning.models import AICallLog
from songguo.backend.services.learning.photo_review import OCRDraft, OCRItemDraft
from songguo.backend.services.learning.submission_models import (
    JudgeResult,
    LearningItem,
    LearningItemStatus,
    LearningSubmission,
    LearningSubmissionSnapshot,
    LearningSubmissionStatus,
    SourceType,
)
from songguo.backend.services.learning.submission_visual_fallback import (
    VISUAL_FALLBACK_KEY,
    _fallback_message_for_reason,
    _fallback_reason_for_item,
    run_visual_fallback_step_async,
    visual_fallback_state,
)


def _manual_item(**kwargs) -> LearningItem:
    values = {
        "submission_id": "sub_001",
        "child_id": "child_001",
        "item_index": 1,
        "question_text": "48 ÷ 6 = ?",
        "child_answer": None,
        "judge_result": JudgeResult.NEEDS_MANUAL_CONFIRM,
        "status": LearningItemStatus.NEEDS_MANUAL_CONFIRM,
        "data_json": {},
    }
    values.update(kwargs)
    return LearningItem(**values)


def test_ocr_answer_missing_fallback_reason_is_specific() -> None:
    item = _manual_item(
        data_json={
            "ocr_action": "RecognizeEduPaperCut",
            "ocr_source": "aliyun_edu_paper_cut",
            "quality_warnings": ["answer_missing"],
        }
    )

    reason = _fallback_reason_for_item(item)

    assert reason == "ocr_answer_missing"
    assert "缺少孩子答案" in _fallback_message_for_reason(reason)


def test_ocr_low_confidence_fallback_reason_uses_warning_or_score() -> None:
    warning_item = _manual_item(
        child_answer="8",
        data_json={"ocr_action": "RecognizeEduPaperCut", "quality_warnings": ["low_confidence"]},
    )
    score_item = _manual_item(
        child_answer="8",
        confidence=0.62,
        data_json={"ocr_action": "RecognizeEduPaperCut"},
    )

    assert _fallback_reason_for_item(warning_item) == "ocr_low_confidence"
    assert _fallback_reason_for_item(score_item) == "ocr_low_confidence"


def test_formula_uncertain_fallback_reason_is_preserved() -> None:
    item = _manual_item(
        child_answer="x=3",
        data_json={"ocr_action": "RecognizeEduFormula"},
    )

    reason = _fallback_reason_for_item(item)

    assert reason == "formula_recognition_uncertain"
    assert "公式识别不稳定" in _fallback_message_for_reason(reason)


class _FailingProviderWithNoneModel:
    provider = "vision"
    model = None

    async def recognize_async(self, *_args, **_kwargs):
        raise RuntimeError("vision unavailable")


class _MismatchedVisualProvider:
    provider = "aliyun_edu_ocr"
    model = "RecognizeEduPaperCut"

    async def recognize_async(self, *_args, **_kwargs):
        return OCRDraft(
            provider=self.provider,
            model=self.model,
            source="aliyun_edu_paper_cut",
            confidence=0.82,
            items=[
                OCRItemDraft(
                    item_index=1,
                    question_text="世界杯于2026年6月11日至7月19日举办，历时( )天。A.38 B.39 C.40",
                    child_answer="3",
                    confidence=0.82,
                )
            ],
        )


class _InMemoryVisualFallbackStore:
    def __init__(self, *, submission: LearningSubmission, item: LearningItem) -> None:
        self.submission = submission
        self.items = {item.item_id: item}
        self.ai_logs: list[AICallLog] = []

    def require_submission(self, submission_id: str) -> LearningSubmission:
        assert submission_id == self.submission.submission_id
        return self.submission

    def list_submission_items(self, submission_id: str) -> list[LearningItem]:
        assert submission_id == self.submission.submission_id
        return list(self.items.values())

    def update_submission_item(self, item_id: str, **updates) -> LearningItem:
        item = self.items[item_id]
        updated = item.model_copy(update=updates)
        self.items[item_id] = updated
        return updated

    def record_ai_call(self, **kwargs) -> AICallLog:
        log = AICallLog(**kwargs)
        self.ai_logs.append(log)
        return log

    def get_submission_snapshot(self, submission_id: str) -> LearningSubmissionSnapshot:
        assert submission_id == self.submission.submission_id
        return LearningSubmissionSnapshot(submission=self.submission, items=list(self.items.values()))

    def update_submission(self, submission_id: str, **updates) -> LearningSubmission:
        assert submission_id == self.submission.submission_id
        self.submission = self.submission.model_copy(update=updates)
        return self.submission

    def complete_submission_if_queue_done(self, submission_id: str) -> LearningSubmission:
        assert submission_id == self.submission.submission_id
        self.submission = self.submission.model_copy(update={"status": LearningSubmissionStatus.COMPLETED})
        return self.submission


@pytest.mark.asyncio
async def test_visual_fallback_error_log_uses_default_model_when_provider_model_is_none(tmp_path) -> None:
    image = np.full((80, 120, 3), 245, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    image_path = tmp_path / "homework.jpg"
    image_path.write_bytes(encoded.tobytes())
    submission = LearningSubmission(
        submission_id="sub_visual",
        child_id="child_001",
        source_type=SourceType.PHOTO,
        status=LearningSubmissionStatus.NEEDS_MANUAL_CONFIRM,
        image_refs=[str(image_path)],
        needs_manual_confirm_count=1,
    )
    item = _manual_item(
        submission_id=submission.submission_id,
        data_json={
            VISUAL_FALLBACK_KEY: {
                "status": "pending",
                "reason": "ocr_low_confidence",
                "message": "正在复核",
                "attempts": 0,
            }
        },
    )
    store = _InMemoryVisualFallbackStore(submission=submission, item=item)

    await run_visual_fallback_step_async(
        store=store,
        submission_id=submission.submission_id,
        ocr_provider=_FailingProviderWithNoneModel(),
    )

    assert store.ai_logs[0].model == "configured_vision_model"
    assert visual_fallback_state(store.items[item.item_id])["status"] == "failed"


@pytest.mark.asyncio
async def test_visual_fallback_does_not_overwrite_item_when_recognized_question_mismatches(tmp_path) -> None:
    image = np.full((80, 120, 3), 245, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    image_path = tmp_path / "homework.jpg"
    image_path.write_bytes(encoded.tobytes())
    submission = LearningSubmission(
        submission_id="sub_visual",
        child_id="child_001",
        source_type=SourceType.PHOTO,
        status=LearningSubmissionStatus.NEEDS_MANUAL_CONFIRM,
        image_refs=[str(image_path)],
        needs_manual_confirm_count=1,
    )
    item = _manual_item(
        submission_id=submission.submission_id,
        question_text="中国少年先锋队到2049年10月13日建队( )周年。",
        data_json={
            VISUAL_FALLBACK_KEY: {
                "status": "pending",
                "reason": "ocr_answer_missing",
                "message": "正在复核",
                "attempts": 0,
            }
        },
    )
    store = _InMemoryVisualFallbackStore(submission=submission, item=item)

    await run_visual_fallback_step_async(
        store=store,
        submission_id=submission.submission_id,
        ocr_provider=_MismatchedVisualProvider(),
    )

    updated = store.items[item.item_id]
    assert updated.question_text == "中国少年先锋队到2049年10月13日建队( )周年。"
    assert updated.child_answer is None
    assert visual_fallback_state(updated)["status"] == "needs_manual_confirm"
    assert visual_fallback_state(updated)["reason"] == "visual_result_mismatch"

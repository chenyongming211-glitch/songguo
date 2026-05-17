from __future__ import annotations

from songguo.backend.services.learning.submission_models import (
    JudgeResult,
    LearningItem,
    LearningItemStatus,
)
from songguo.backend.services.learning.submission_visual_fallback import (
    _fallback_message_for_reason,
    _fallback_reason_for_item,
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

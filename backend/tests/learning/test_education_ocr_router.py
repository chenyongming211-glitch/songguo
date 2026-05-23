from __future__ import annotations

from songguo.backend.services.learning.education_ocr_router import (
    EducationOcrAction,
    EducationOcrRouter,
    OcrQualitySignal,
)
from songguo.backend.services.learning.real_model_client import AliyunEduOCRConfig


def test_router_uses_config_scene_when_not_auto() -> None:
    router = EducationOcrRouter(AliyunEduOCRConfig(scene="paper_ocr"))

    plan = router.initial_plan(filename="paper.jpg", content_type="image/jpeg")

    assert plan.primary_action == EducationOcrAction.PAPER_OCR
    assert plan.reason == "configured_scene"


def test_router_defaults_auto_to_paper_cut() -> None:
    router = EducationOcrRouter(AliyunEduOCRConfig(scene="auto"))

    plan = router.initial_plan(filename="homework.jpg", content_type="image/jpeg")

    assert plan.primary_action == EducationOcrAction.PAPER_CUT
    assert plan.expected_output == "question_boxes"


def test_router_falls_back_to_paper_cut_when_paper_structed_has_no_items() -> None:
    router = EducationOcrRouter(AliyunEduOCRConfig(scene="auto"))

    actions = router.secondary_actions(
        OcrQualitySignal(
            primary_action=EducationOcrAction.PAPER_STRUCTED,
            item_count=0,
            answer_count=0,
            raw_text_length=0,
            confidence=0.0,
        )
    )

    assert actions == [EducationOcrAction.PAPER_CUT]


def test_router_uses_paper_ocr_to_enrich_paper_structed_with_low_answer_coverage() -> None:
    router = EducationOcrRouter(AliyunEduOCRConfig(scene="auto", hybrid_text_fallback_min_answer_rate=0.6))

    actions = router.secondary_actions(
        OcrQualitySignal(
            primary_action=EducationOcrAction.PAPER_STRUCTED,
            item_count=4,
            answer_count=1,
            raw_text_length=120,
            confidence=0.82,
        )
    )

    assert actions == [EducationOcrAction.PAPER_OCR]


def test_router_triggers_paper_ocr_when_answer_coverage_is_low() -> None:
    router = EducationOcrRouter(AliyunEduOCRConfig(scene="auto", hybrid_text_fallback_min_answer_rate=0.6))

    actions = router.secondary_actions(
        OcrQualitySignal(
            primary_action=EducationOcrAction.PAPER_CUT,
            item_count=5,
            answer_count=1,
            raw_text_length=120,
            confidence=0.82,
        )
    )

    assert actions == [EducationOcrAction.PAPER_OCR]


def test_router_does_not_add_secondary_action_when_quality_is_good() -> None:
    router = EducationOcrRouter(AliyunEduOCRConfig(scene="auto", hybrid_text_fallback_min_answer_rate=0.6))

    actions = router.secondary_actions(
        OcrQualitySignal(
            primary_action=EducationOcrAction.PAPER_CUT,
            item_count=5,
            answer_count=5,
            raw_text_length=300,
            confidence=0.9,
        )
    )

    assert actions == []


def test_router_does_not_add_paper_ocr_for_rich_text_page_with_few_answers() -> None:
    router = EducationOcrRouter(AliyunEduOCRConfig(scene="auto", hybrid_text_fallback_min_answer_rate=0.6))

    actions = router.secondary_actions(
        OcrQualitySignal(
            primary_action=EducationOcrAction.PAPER_CUT,
            item_count=5,
            answer_count=1,
            raw_text_length=520,
            confidence=0.72,
        )
    )

    assert actions == []


def test_router_uses_paper_ocr_when_paper_cut_answers_are_all_missing_even_with_rich_text() -> None:
    router = EducationOcrRouter(AliyunEduOCRConfig(scene="auto", hybrid_text_fallback_min_answer_rate=0.6))

    actions = router.secondary_actions(
        OcrQualitySignal(
            primary_action=EducationOcrAction.PAPER_CUT,
            item_count=5,
            answer_count=0,
            raw_text_length=520,
            confidence=0.82,
        )
    )

    assert actions == [EducationOcrAction.PAPER_OCR]

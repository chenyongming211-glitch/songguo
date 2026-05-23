from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class EducationOcrAction(StrEnum):
    PAPER_CUT = "RecognizeEduPaperCut"
    PAPER_OCR = "RecognizeEduPaperOcr"
    PAPER_STRUCTED = "RecognizeEduPaperStructed"
    ORAL_CALCULATION = "RecognizeEduOralCalculation"
    QUESTION_OCR = "RecognizeEduQuestionOcr"
    FORMULA = "RecognizeEduFormula"


SCENE_TO_ACTION = {
    "auto": EducationOcrAction.PAPER_CUT,
    "paper_cut": EducationOcrAction.PAPER_CUT,
    "cut": EducationOcrAction.PAPER_CUT,
    "paper_ocr": EducationOcrAction.PAPER_OCR,
    "paper": EducationOcrAction.PAPER_OCR,
    "paper_structed": EducationOcrAction.PAPER_STRUCTED,
    "paper_structured": EducationOcrAction.PAPER_STRUCTED,
    "oral": EducationOcrAction.ORAL_CALCULATION,
    "oral_calculation": EducationOcrAction.ORAL_CALCULATION,
    "kousuan": EducationOcrAction.ORAL_CALCULATION,
    "question": EducationOcrAction.QUESTION_OCR,
    "question_ocr": EducationOcrAction.QUESTION_OCR,
    "formula": EducationOcrAction.FORMULA,
}


@dataclass(frozen=True)
class EducationOcrPlan:
    primary_action: EducationOcrAction
    secondary_actions: list[EducationOcrAction] = field(default_factory=list)
    reason: str = "auto_default"
    expected_output: str = "question_boxes"
    cost_level: str = "normal"
    allow_async_fallback: bool = True
    subject_hint: str = "default"

    def model_dump(self) -> dict[str, Any]:
        return {
            "primary_action": self.primary_action.value,
            "secondary_actions": [action.value for action in self.secondary_actions],
            "reason": self.reason,
            "expected_output": self.expected_output,
            "cost_level": self.cost_level,
            "allow_async_fallback": self.allow_async_fallback,
            "subject_hint": self.subject_hint,
        }


@dataclass(frozen=True)
class OcrQualitySignal:
    primary_action: EducationOcrAction
    item_count: int
    answer_count: int
    raw_text_length: int
    confidence: float


class EducationOcrRouter:
    def __init__(self, config: Any) -> None:
        self.config = config

    def initial_plan(self, *, filename: str, content_type: str = "") -> EducationOcrPlan:
        scene = _normalize_scene(str(getattr(self.config, "scene", "") or "auto"))
        action = SCENE_TO_ACTION.get(scene, EducationOcrAction.PAPER_CUT)
        reason = "configured_scene" if scene != "auto" else "auto_default_multi_question"
        expected_output = "full_page_text" if action == EducationOcrAction.PAPER_OCR else "question_boxes"
        return EducationOcrPlan(
            primary_action=action,
            reason=reason,
            expected_output=expected_output,
            subject_hint=str(getattr(self.config, "subject", "default") or "default"),
        )

    def secondary_actions(self, signal: OcrQualitySignal) -> list[EducationOcrAction]:
        if signal.primary_action != EducationOcrAction.PAPER_CUT:
            return []
        if signal.item_count <= 0:
            return [EducationOcrAction.PAPER_OCR]
        answer_rate = signal.answer_count / signal.item_count
        min_rate = max(
            0.0,
            min(
                1.0,
                float(getattr(self.config, "hybrid_text_fallback_min_answer_rate", 0.6) or 0.6),
            ),
        )
        if answer_rate < min_rate and (signal.raw_text_length < 180 or signal.confidence < 0.7):
            return [EducationOcrAction.PAPER_OCR]
        if signal.raw_text_length < 20 and signal.confidence < 0.75:
            return [EducationOcrAction.PAPER_OCR]
        return []


def _normalize_scene(scene: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in scene.strip().lower()).strip("_") or "auto"

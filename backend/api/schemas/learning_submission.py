from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from songguo.backend.services.learning.service import SubmitAttemptResult
from songguo.backend.services.learning.submission_models import (
    EvidenceType,
    JudgeResult,
    LearningItemStatus,
    LearningSubmissionStatus,
    MasteryState,
    SourceType,
    TutorQueueStatus,
)


class CreateLearningSubmissionRequest(BaseModel):
    child_id: str
    subject: str = "auto"
    grade: int = 3
    source_type: SourceType = SourceType.TEXT
    raw_text: str = ""
    image_refs: list[str] = Field(default_factory=list)
    draft_items: list["LearningSubmissionDraftItemRequest"] = Field(default_factory=list)


class LearningSubmissionDraftItemRequest(BaseModel):
    item_index: int
    bbox: dict[str, int] | None = None
    ocr_action: str = ""
    ocr_source: str = ""
    ocr_judgement: str = ""
    marking_source: str = ""
    correct_answer: str = ""
    evidence_points: list[str] = Field(default_factory=list)
    display_status: str = ""
    quality_warnings: list[str] = Field(default_factory=list)


class ConfirmLearningSubmissionRequest(BaseModel):
    raw_text: str | None = None
    start_tutor: bool = True


class TutorAttemptRequest(BaseModel):
    child_answer: str


class LearningSubmissionItemResponse(BaseModel):
    item_id: str
    item_index: int
    question_text: str
    child_answer: str | None = None
    detected_subject: str = ""
    detected_task_type: str = ""
    evaluation_mode: str = ""
    correct_answer: str | None = None
    judge_result: JudgeResult
    question_type_id: str = ""
    knowledge_point: str = ""
    misconception_tag: str | None = None
    rubric_outcome: str = ""
    rubric_feedback: str = ""
    rubric_scores: dict[str, int] = Field(default_factory=dict)
    evidence_points: list[str] = Field(default_factory=list)
    bbox: dict[str, int] | None = None
    status: LearningItemStatus
    display_status: str = "pending"
    visual_fallback_status: str = ""
    visual_fallback_reason: str = ""
    visual_fallback_message: str = ""
    visual_fallback_attempts: int = 0


class MasteryEvidenceResponse(BaseModel):
    evidence_id: str
    item_id: str
    question_type_id: str
    evidence_type: EvidenceType
    is_correct: bool
    mastery_state_after: MasteryState
    review_due: bool = False
    misconception_tag: str | None = None


class TutorQueueItemResponse(BaseModel):
    queue_item_id: str
    item_id: str
    question_type_id: str
    status: TutorQueueStatus
    tutor_session_id: str | None = None


class ActiveTutorSessionResponse(BaseModel):
    session_id: str
    question_text: str
    current_prompt: str
    hint_level: int
    attempt_count: int
    answer_unlocked: bool


class LearningSubmissionResponse(BaseModel):
    submission_id: str
    child_id: str
    subject: str
    detected_subject: str = ""
    detected_task_type: str = ""
    detected_intent: str = ""
    subject_confidence: float = 0.0
    route_to: str = ""
    routing_evidence: list[str] = Field(default_factory=list)
    guard_reason: str = ""
    router_version: str = ""
    needs_clarification: bool = False
    grade: int
    source_type: SourceType
    status: LearningSubmissionStatus
    item_count: int = 0
    correct_count: int = 0
    wrong_count: int = 0
    needs_manual_confirm_count: int = 0
    pending_visual_fallback_count: int = 0
    visual_fallback_active: bool = False
    active_queue_item_id: str | None = None
    active_tutor_session: ActiveTutorSessionResponse | None = None
    items: list[LearningSubmissionItemResponse] = Field(default_factory=list)
    mastery_evidence: list[MasteryEvidenceResponse] = Field(default_factory=list)
    tutor_queue: list[TutorQueueItemResponse] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


class TutorAttemptResponse(BaseModel):
    attempt: SubmitAttemptResult
    submission: LearningSubmissionResponse

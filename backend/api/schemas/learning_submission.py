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
    subject: str = "math"
    grade: int = 3
    source_type: SourceType = SourceType.TEXT
    raw_text: str = ""


class ConfirmLearningSubmissionRequest(BaseModel):
    raw_text: str | None = None


class TutorAttemptRequest(BaseModel):
    child_answer: str


class LearningSubmissionItemResponse(BaseModel):
    item_id: str
    item_index: int
    question_text: str
    child_answer: str | None = None
    correct_answer: str | None = None
    judge_result: JudgeResult
    question_type_id: str = ""
    knowledge_point: str = ""
    misconception_tag: str | None = None
    status: LearningItemStatus


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
    grade: int
    source_type: SourceType
    status: LearningSubmissionStatus
    item_count: int = 0
    correct_count: int = 0
    wrong_count: int = 0
    needs_manual_confirm_count: int = 0
    active_queue_item_id: str | None = None
    active_tutor_session: ActiveTutorSessionResponse | None = None
    items: list[LearningSubmissionItemResponse] = Field(default_factory=list)
    mastery_evidence: list[MasteryEvidenceResponse] = Field(default_factory=list)
    tutor_queue: list[TutorQueueItemResponse] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


class TutorAttemptResponse(BaseModel):
    attempt: SubmitAttemptResult
    submission: LearningSubmissionResponse

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from songguo.backend.services.learning.models import utc_now


class SourceType(StrEnum):
    TEXT = "text"
    PHOTO = "photo"
    VOICE = "voice"


class LearningSubmissionStatus(StrEnum):
    INTAKE_PENDING = "intake_pending"
    JUDGED = "judged"
    TUTORING = "tutoring"
    COMPLETED = "completed"
    NEEDS_MANUAL_CONFIRM = "needs_manual_confirm"


class JudgeResult(StrEnum):
    UNKNOWN = "unknown"
    CORRECT = "correct"
    WRONG = "wrong"
    NEEDS_MANUAL_CONFIRM = "needs_manual_confirm"


class LearningItemStatus(StrEnum):
    INTAKE_PENDING = "intake_pending"
    JUDGED = "judged"
    QUEUED_FOR_TUTORING = "queued_for_tutoring"
    TUTORING = "tutoring"
    COMPLETED = "completed"
    NEEDS_MANUAL_CONFIRM = "needs_manual_confirm"


class EvidenceType(StrEnum):
    SUBMISSION_CORRECT = "submission_correct"
    WRONG_UNRESOLVED = "wrong_unresolved"
    TUTOR_COMPLETED = "tutor_completed"
    REVIEW_CORRECT = "review_correct"


class MasteryState(StrEnum):
    OBSERVED = "observed"
    NEEDS_REVIEW = "needs_review"
    REVIEWING = "reviewing"
    MASTERED = "mastered"


class TutorQueueStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    SKIPPED = "skipped"


class LearningSubmission(BaseModel):
    submission_id: str = Field(default_factory=lambda: f"sub_{uuid4().hex}")
    child_id: str
    family_id: str = ""
    subject: str = "math"
    grade: int = 3
    source_type: SourceType = SourceType.TEXT
    status: LearningSubmissionStatus = LearningSubmissionStatus.INTAKE_PENDING
    raw_text: str = ""
    image_refs: list[str] = Field(default_factory=list)
    audio_refs: list[str] = Field(default_factory=list)
    item_count: int = 0
    correct_count: int = 0
    wrong_count: int = 0
    needs_manual_confirm_count: int = 0
    active_queue_item_id: str | None = None
    data_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None


class LearningItem(BaseModel):
    item_id: str = Field(default_factory=lambda: f"item_{uuid4().hex}")
    submission_id: str
    child_id: str
    family_id: str = ""
    item_index: int
    question_text: str
    child_answer: str | None = None
    correct_answer: str | None = None
    judge_result: JudgeResult = JudgeResult.UNKNOWN
    question_type_id: str = ""
    knowledge_point: str = ""
    misconception_tag: str | None = None
    confidence: float = 0.0
    bbox_json: dict[str, Any] | None = None
    status: LearningItemStatus = LearningItemStatus.INTAKE_PENDING
    tutor_session_id: str | None = None
    data_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class MasteryEvidence(BaseModel):
    evidence_id: str = Field(default_factory=lambda: f"ev_{uuid4().hex}")
    item_id: str
    child_id: str
    family_id: str = ""
    question_type_id: str
    evidence_type: EvidenceType
    is_correct: bool
    mastery_state_after: MasteryState
    review_due: bool = False
    solved_without_help: bool = False
    misconception_tag: str | None = None
    confidence: float = 0.0
    data_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class TutorQueueItem(BaseModel):
    queue_item_id: str = Field(default_factory=lambda: f"tq_{uuid4().hex}")
    submission_id: str
    item_id: str
    child_id: str
    family_id: str = ""
    question_type_id: str
    priority: int = 0
    status: TutorQueueStatus = TutorQueueStatus.PENDING
    tutor_session_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None


class LearningItemDraft(BaseModel):
    item_index: int
    question_text: str
    child_answer: str | None = None
    confidence: float = 0.0


class LearningSubmissionDraft(BaseModel):
    child_id: str
    subject: str
    grade: int
    source_type: SourceType
    raw_text: str
    items: list[LearningItemDraft] = Field(default_factory=list)
    needs_manual_confirm: bool = False


class LearningSubmissionSnapshot(BaseModel):
    submission: LearningSubmission
    items: list[LearningItem] = Field(default_factory=list)
    mastery_evidence: list[MasteryEvidence] = Field(default_factory=list)
    tutor_queue: list[TutorQueueItem] = Field(default_factory=list)

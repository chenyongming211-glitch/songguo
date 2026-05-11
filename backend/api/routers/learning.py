from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel

from songguo.backend.api.schemas.learning_submission import (
    ActiveTutorSessionResponse,
    ConfirmLearningSubmissionRequest,
    CreateLearningSubmissionRequest,
    LearningSubmissionItemResponse,
    LearningSubmissionResponse,
    MasteryEvidenceResponse,
    TutorAttemptRequest,
    TutorAttemptResponse,
    TutorQueueItemResponse,
)
from songguo.backend.services.learning.models import ResumeSnapshot, TeachingProgress, WrongQuestion
from songguo.backend.services.learning.progress import build_teaching_progress
from songguo.backend.services.learning.photo_review import (
    ConfirmPhotoReviewRequest,
    DeterministicOCRProvider,
    PhotoReview,
    PhotoReviewService,
    VisionOCRProvider,
)
from songguo.backend.services.learning.practice_recommender import (
    PracticeRecommendation,
    recommend_targeted_practice,
)
from songguo.backend.services.learning.service import (
    CreateLearningSessionResult,
    LearningService,
    SubmitAttemptResult,
    get_global_learning_service,
)
from songguo.backend.services.learning.voice_input import (
    DeterministicASRProvider,
    OpenAICompatibleASRProvider,
    VoiceInputService,
)
from songguo.backend.services.session_auth import authorize_child_access, strict_auth_enabled

router = APIRouter()
_PHOTO_REVIEW_SERVICE: PhotoReviewService | None = None
_VOICE_INPUT_SERVICE: VoiceInputService | None = None


class CreateLearningSessionRequest(BaseModel):
    child_id: str
    subject: str
    grade: int
    input_type: str = "text"
    question_text: str


class SubmitAttemptRequest(BaseModel):
    child_answer: str


class PracticeResultRequest(BaseModel):
    child_id: str
    correct: bool


class SubmissionPhotoDraftResponse(BaseModel):
    source_type: str = "photo"
    image_path: str
    raw_text: str
    question_text: str = ""
    child_answer: str = ""
    work_steps: str = ""
    confidence: float = 0.0
    needs_confirmation: bool = True


class SubmissionVoiceDraftResponse(BaseModel):
    source_type: str = "voice"
    audio_path: str
    raw_text: str
    transcript: str = ""
    confidence: float = 0.0
    needs_confirmation: bool = True


class LearningSessionListItem(BaseModel):
    session_id: str
    child_id: str
    title: str
    subject: str
    grade: int
    phase: str
    hint_level: int
    attempt_count: int
    answer_unlocked: bool
    current_prompt: str
    teaching_progress: TeachingProgress
    updated_at: str


class LearningSessionListResponse(BaseModel):
    sessions: list[LearningSessionListItem]


def get_learning_service() -> LearningService:
    return get_global_learning_service()


def get_photo_review_service() -> PhotoReviewService:
    global _PHOTO_REVIEW_SERVICE
    if _PHOTO_REVIEW_SERVICE is None:
        ocr_provider_name = (
            os.getenv("SONGGUO_PHOTO_OCR_PROVIDER")
            or os.getenv("DEEPTUTOR_PHOTO_OCR_PROVIDER")
            or ""
        )
        vision_model = os.getenv("SONGGUO_VISION_MODEL") or os.getenv("DEEPTUTOR_VISION_MODEL")
        ocr_provider = (
            VisionOCRProvider(model=vision_model)
            if ocr_provider_name == "vision"
            else DeterministicOCRProvider()
        )
        _PHOTO_REVIEW_SERVICE = PhotoReviewService(
            store=get_learning_service().store,
            artifact_root=Path("data/user/learning_artifacts"),
            ocr_provider=ocr_provider,
            learning_service=get_learning_service(),
        )
    return _PHOTO_REVIEW_SERVICE


def get_voice_input_service() -> VoiceInputService:
    global _VOICE_INPUT_SERVICE
    if _VOICE_INPUT_SERVICE is None:
        asr_provider_name = os.getenv("SONGGUO_ASR_PROVIDER") or ""
        asr_model = os.getenv("SONGGUO_ASR_MODEL") or "whisper-1"
        asr_provider = (
            OpenAICompatibleASRProvider(model=asr_model)
            if asr_provider_name == "openai"
            else DeterministicASRProvider()
        )
        _VOICE_INPUT_SERVICE = VoiceInputService(
            artifact_root=Path("data/user/learning_artifacts"),
            asr_provider=asr_provider,
        )
    return _VOICE_INPUT_SERVICE


def _authorize_child(child_id: str, session_token: str | None) -> None:
    try:
        authorize_child_access(
            get_learning_service().store,
            child_id=child_id,
            session_token=session_token,
        )
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def _authorize_submission_access(
    submission_id: str,
    *,
    child_id: str | None,
    session_token: str | None,
):
    submission = get_learning_service().store.get_submission(submission_id)
    if submission is None:
        raise HTTPException(status_code=404, detail="Learning submission not found")
    if child_id and child_id != submission.child_id:
        raise HTTPException(status_code=403, detail="child_id does not match submission")
    if session_token or strict_auth_enabled():
        _authorize_child(submission.child_id, session_token)
    return submission


def _build_photo_submission_raw_text(*, question_text: str, child_answer: str, work_steps: str) -> str:
    return "\n".join(
        item
        for item in [
            question_text.strip(),
            f"孩子答案：{child_answer.strip()}" if child_answer.strip() else "",
            f"解题过程：{work_steps.strip()}" if work_steps.strip() else "",
        ]
        if item
    )


def _submission_response(submission_id: str) -> LearningSubmissionResponse:
    service = get_learning_service()
    snapshot = service.get_submission_snapshot(submission_id)
    submission = snapshot.submission
    active_session = None
    active_queue = service.store.get_active_tutor_item(submission_id)
    if active_queue and active_queue.tutor_session_id:
        session = service.store.get_session(active_queue.tutor_session_id)
        if session is not None:
            active_session = ActiveTutorSessionResponse(
                session_id=session.session_id,
                question_text=session.question_text,
                current_prompt=session.current_prompt,
                hint_level=session.hint_level,
                attempt_count=session.attempt_count,
                answer_unlocked=session.answer_unlocked,
            )
    return LearningSubmissionResponse(
        submission_id=submission.submission_id,
        child_id=submission.child_id,
        subject=submission.subject,
        grade=submission.grade,
        source_type=submission.source_type,
        status=submission.status,
        item_count=submission.item_count,
        correct_count=submission.correct_count,
        wrong_count=submission.wrong_count,
        needs_manual_confirm_count=submission.needs_manual_confirm_count,
        active_queue_item_id=submission.active_queue_item_id,
        active_tutor_session=active_session,
        items=[
            LearningSubmissionItemResponse(
                item_id=item.item_id,
                item_index=item.item_index,
                question_text=item.question_text,
                child_answer=item.child_answer,
                correct_answer=item.correct_answer,
                judge_result=item.judge_result,
                question_type_id=item.question_type_id,
                knowledge_point=item.knowledge_point,
                misconception_tag=item.misconception_tag,
                status=item.status,
            )
            for item in snapshot.items
        ],
        mastery_evidence=[
            MasteryEvidenceResponse(
                evidence_id=evidence.evidence_id,
                item_id=evidence.item_id,
                question_type_id=evidence.question_type_id,
                evidence_type=evidence.evidence_type,
                is_correct=evidence.is_correct,
                mastery_state_after=evidence.mastery_state_after,
                review_due=evidence.review_due,
                misconception_tag=evidence.misconception_tag,
            )
            for evidence in snapshot.mastery_evidence
        ],
        tutor_queue=[
            TutorQueueItemResponse(
                queue_item_id=item.queue_item_id,
                item_id=item.item_id,
                question_type_id=item.question_type_id,
                status=item.status,
                tutor_session_id=item.tutor_session_id,
            )
            for item in snapshot.tutor_queue
        ],
        summary={
            "item_count": submission.item_count,
            "correct_count": submission.correct_count,
            "wrong_count": submission.wrong_count,
            "tutor_queue_count": len(snapshot.tutor_queue),
        },
    )


@router.post("/submissions/photo-draft", response_model=SubmissionPhotoDraftResponse)
async def create_submission_photo_draft(
    child_id: str = Form(...),
    subject: str = Form("math"),
    grade: int = Form(3),
    file: UploadFile = File(...),
    x_session_token: str | None = Header(None, alias="X-Session-Token"),
) -> SubmissionPhotoDraftResponse:
    _authorize_child(child_id, x_session_token)
    content = await file.read()
    image_path, draft = await get_photo_review_service().recognize_submission_draft_async(
        filename=file.filename or "upload.bin",
        content=content,
        content_type=file.content_type or "application/octet-stream",
    )
    return SubmissionPhotoDraftResponse(
        image_path=image_path,
        raw_text=_build_photo_submission_raw_text(
            question_text=draft.question_text,
            child_answer=draft.child_answer,
            work_steps=draft.work_steps,
        ),
        question_text=draft.question_text,
        child_answer=draft.child_answer,
        work_steps=draft.work_steps,
        confidence=draft.confidence,
        needs_confirmation=draft.needs_confirmation,
    )


@router.post("/submissions/voice-draft", response_model=SubmissionVoiceDraftResponse)
async def create_submission_voice_draft(
    child_id: str = Form(...),
    subject: str = Form("math"),
    grade: int = Form(3),
    file: UploadFile = File(...),
    x_session_token: str | None = Header(None, alias="X-Session-Token"),
) -> SubmissionVoiceDraftResponse:
    _authorize_child(child_id, x_session_token)
    content = await file.read()
    audio_path, draft = await get_voice_input_service().recognize_submission_draft_async(
        filename=file.filename or "voice.mp3",
        content=content,
        content_type=file.content_type or "application/octet-stream",
    )
    return SubmissionVoiceDraftResponse(
        audio_path=audio_path,
        raw_text=draft.transcript,
        transcript=draft.transcript,
        confidence=draft.confidence,
        needs_confirmation=draft.needs_confirmation,
    )


@router.post("/submissions", response_model=LearningSubmissionResponse)
def create_learning_submission(
    request: CreateLearningSubmissionRequest,
    x_session_token: str | None = Header(None, alias="X-Session-Token"),
) -> LearningSubmissionResponse:
    _authorize_child(request.child_id, x_session_token)
    try:
        result = get_learning_service().create_submission(
            child_id=request.child_id,
            subject=request.subject,
            grade=request.grade,
            source_type=request.source_type,
            raw_text=request.raw_text,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return _submission_response(result.submission_id)


@router.get("/submissions/{submission_id}", response_model=LearningSubmissionResponse)
def get_learning_submission(
    submission_id: str,
    child_id: str | None = None,
    x_session_token: str | None = Header(None, alias="X-Session-Token"),
) -> LearningSubmissionResponse:
    _authorize_submission_access(
        submission_id,
        child_id=child_id,
        session_token=x_session_token,
    )
    return _submission_response(submission_id)


@router.post("/submissions/{submission_id}/confirm", response_model=LearningSubmissionResponse)
def confirm_learning_submission(
    submission_id: str,
    request: ConfirmLearningSubmissionRequest,
    x_session_token: str | None = Header(None, alias="X-Session-Token"),
) -> LearningSubmissionResponse:
    submission = _authorize_submission_access(
        submission_id,
        child_id=None,
        session_token=x_session_token,
    )
    if request.raw_text is not None:
        get_learning_service().store.update_submission(submission.submission_id, raw_text=request.raw_text)
    get_learning_service().confirm_submission(submission.submission_id)
    return _submission_response(submission_id)


@router.post("/submissions/{submission_id}/tutor/next", response_model=LearningSubmissionResponse)
def start_next_submission_tutor_item(
    submission_id: str,
    x_session_token: str | None = Header(None, alias="X-Session-Token"),
) -> LearningSubmissionResponse:
    _authorize_submission_access(
        submission_id,
        child_id=None,
        session_token=x_session_token,
    )
    try:
        result = get_learning_service().start_next_tutor_item(submission_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Learning submission not found") from None
    return _submission_response(result.submission_id)


@router.post("/submissions/{submission_id}/tutor/attempt", response_model=TutorAttemptResponse)
def submit_submission_tutor_attempt(
    submission_id: str,
    request: TutorAttemptRequest,
    x_session_token: str | None = Header(None, alias="X-Session-Token"),
) -> TutorAttemptResponse:
    _authorize_submission_access(
        submission_id,
        child_id=None,
        session_token=x_session_token,
    )
    try:
        result = get_learning_service().submit_submission_tutor_attempt(
            submission_id,
            child_answer=request.child_answer,
        )
    except KeyError:
        raise HTTPException(status_code=409, detail="本次错题陪练已经结束，请返回本次总结。") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return TutorAttemptResponse(
        attempt=result.attempt,
        submission=_submission_response(result.submission.submission_id),
    )


@router.post("/session", response_model=CreateLearningSessionResult)
def create_learning_session(
    request: CreateLearningSessionRequest,
    x_session_token: str | None = Header(None, alias="X-Session-Token"),
) -> CreateLearningSessionResult:
    if request.input_type != "text":
        raise HTTPException(status_code=400, detail="Only text input is supported in P0")
    _authorize_child(request.child_id, x_session_token)
    try:
        return get_learning_service().create_session(
            child_id=request.child_id,
            subject=request.subject,
            grade=request.grade,
            question_text=request.question_text,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@router.get("/sessions", response_model=LearningSessionListResponse)
def list_learning_sessions(
    child_id: str | None = None,
    x_session_token: str | None = Header(None, alias="X-Session-Token"),
) -> LearningSessionListResponse:
    if strict_auth_enabled() and not child_id:
        raise HTTPException(status_code=400, detail="child_id is required in strict auth mode")
    if child_id:
        _authorize_child(child_id, x_session_token)
    sessions = get_learning_service().store.list_sessions(child_id=child_id)
    return LearningSessionListResponse(
        sessions=[
            LearningSessionListItem(
                session_id=session.session_id,
                child_id=session.child_id,
                title=session.question_text[:60],
                subject=session.subject,
                grade=session.grade,
                phase=session.phase.value,
                hint_level=session.hint_level,
                attempt_count=session.attempt_count,
                answer_unlocked=session.answer_unlocked,
                current_prompt=session.current_prompt,
                teaching_progress=build_teaching_progress(session),
                updated_at=session.updated_at.isoformat(),
            )
            for session in sessions
        ]
    )


@router.post("/session/{session_id}/attempt", response_model=SubmitAttemptResult)
def submit_attempt(
    session_id: str,
    request: SubmitAttemptRequest,
    x_session_token: str | None = Header(None, alias="X-Session-Token"),
) -> SubmitAttemptResult:
    session = get_learning_service().store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Learning session not found")
    if x_session_token or strict_auth_enabled():
        _authorize_child(session.child_id, x_session_token)
    try:
        return get_learning_service().submit_attempt(
            session_id,
            child_answer=request.child_answer,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Learning session not found") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@router.get("/session/{session_id}/resume", response_model=ResumeSnapshot)
def resume_learning_session(
    session_id: str,
    x_session_token: str | None = Header(None, alias="X-Session-Token"),
) -> ResumeSnapshot:
    session = get_learning_service().store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Learning session not found")
    if x_session_token or strict_auth_enabled():
        _authorize_child(session.child_id, x_session_token)
    try:
        return get_learning_service().store.get_resume_snapshot(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Learning session not found") from None


@router.get("/children/{child_id}/targeted-practice", response_model=PracticeRecommendation)
def get_targeted_practice(
    child_id: str,
    limit: int = 3,
    scope: str = "weekly",
    x_session_token: str | None = Header(None, alias="X-Session-Token"),
) -> PracticeRecommendation:
    _authorize_child(child_id, x_session_token)
    return recommend_targeted_practice(
        get_learning_service().store,
        child_id=child_id,
        limit=limit,
        scope=scope,
    )


@router.post("/wrong-questions/{question_id}/practice-result", response_model=WrongQuestion)
def submit_practice_result(
    question_id: str,
    request: PracticeResultRequest,
    x_session_token: str | None = Header(None, alias="X-Session-Token"),
) -> WrongQuestion:
    _authorize_child(request.child_id, x_session_token)
    try:
        return get_learning_service().store.update_wrong_question_practice_result(
            question_id=question_id,
            child_id=request.child_id,
            correct=request.correct,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Wrong question not found") from None


@router.post("/photo-review", response_model=PhotoReview)
async def create_photo_review(
    child_id: str = Form(...),
    subject: str = Form("math"),
    grade: int = Form(3),
    file: UploadFile = File(...),
    x_session_token: str | None = Header(None, alias="X-Session-Token"),
) -> PhotoReview:
    _authorize_child(child_id, x_session_token)
    content = await file.read()
    try:
        return await get_photo_review_service().create_from_upload_async(
            child_id=child_id,
            subject=subject,
            grade=grade,
            filename=file.filename or "upload.bin",
            content=content,
            content_type=file.content_type or "application/octet-stream",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@router.get("/photo-review/{review_id}", response_model=PhotoReview)
def get_photo_review(review_id: str) -> PhotoReview:
    try:
        return get_photo_review_service().get(review_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Photo review not found") from None


@router.post("/photo-review/{review_id}/confirm", response_model=PhotoReview)
def confirm_photo_review(review_id: str, request: ConfirmPhotoReviewRequest) -> PhotoReview:
    try:
        return get_photo_review_service().confirm(
            review_id,
            question_text=request.question_text,
            child_answer=request.child_answer,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Photo review not found") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

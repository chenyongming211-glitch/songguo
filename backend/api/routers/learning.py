from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
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
    AliyunEduOCRProvider,
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
from songguo.backend.services.learning.submission_visual_fallback import (
    visual_fallback_counts,
    visual_fallback_state,
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


class SubmissionPhotoDraftItemResponse(BaseModel):
    item_index: int
    question_text: str
    child_answer: str = ""
    work_steps: str = ""
    confidence: float = 0.0
    bbox: dict[str, int] | None = None
    ocr_action: str = ""
    ocr_source: str = ""
    ocr_judgement: str = ""
    marking_source: str = ""
    correct_answer: str = ""
    evidence_points: list[str] = []
    quality_warnings: list[str] = []
    display_status: str = "pending"


class SubmissionPhotoDraftResponse(BaseModel):
    source_type: str = "photo"
    image_path: str
    image_refs: list[str] = []
    preview_image_path: str = ""
    preview_image_url: str = ""
    raw_text: str
    question_text: str = ""
    child_answer: str = ""
    work_steps: str = ""
    confidence: float = 0.0
    needs_confirmation: bool = True
    items: list[SubmissionPhotoDraftItemResponse] = []
    ocr_plan: dict[str, Any] = {}
    detected_regions: list[dict[str, int]] = []
    quality_warnings: list[str] = []
    quality_message: str = ""
    preprocess_source: str = ""
    ocr_provider: str = ""
    ocr_model: str = ""
    ocr_source: str = ""


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


def _env_with_local(key: str, legacy_key: str | None = None) -> str:
    value = os.getenv(key)
    if value:
        return value.strip()
    value = _read_local_dotenv_value(key)
    if value:
        return value
    if not legacy_key:
        return ""
    return os.getenv(legacy_key) or _read_local_dotenv_value(legacy_key)


def _read_local_dotenv_value(key: str) -> str:
    root = Path(__file__).resolve().parents[3]
    for env_path in (root / "songguo" / ".env", root / ".env"):
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            current_key, value = line.split("=", 1)
            if current_key.strip() == key:
                return value.strip().strip("\"'")
    return ""


def get_photo_review_service() -> PhotoReviewService:
    global _PHOTO_REVIEW_SERVICE
    if _PHOTO_REVIEW_SERVICE is None:
        ocr_provider_name = (
            _env_with_local("SONGGUO_PHOTO_OCR_PROVIDER", "DEEPTUTOR_PHOTO_OCR_PROVIDER")
            or ""
        )
        vision_model = _env_with_local("SONGGUO_VISION_MODEL", "DEEPTUTOR_VISION_MODEL")
        if ocr_provider_name == "vision":
            ocr_provider = VisionOCRProvider(model=vision_model)
        elif ocr_provider_name in {"aliyun_edu", "aliyun_edu_ocr"}:
            ocr_provider = AliyunEduOCRProvider()
        else:
            ocr_provider = DeterministicOCRProvider()
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


def _photo_submission_raw_text_from_draft(draft) -> str:
    items = list(getattr(draft, "items", []) or [])
    if items:
        structured_raw_text = "\n\n".join(
            _build_photo_submission_raw_text(
                question_text=item.question_text,
                child_answer=item.child_answer,
                work_steps=item.work_steps,
            )
            for item in items
        )
        if structured_raw_text.strip():
            return structured_raw_text
    raw_text = str(getattr(draft, "raw_text", "") or "").strip()
    if raw_text:
        return raw_text
    return _build_photo_submission_raw_text(
        question_text=draft.question_text,
        child_answer=draft.child_answer,
        work_steps=draft.work_steps,
    )


def _photo_preview_url(preview_image_path: str) -> str:
    if not preview_image_path:
        return ""
    return f"/api/v1/learning/submissions/photo-preview/{Path(preview_image_path).name}"


def _photo_draft_item_display_status(item) -> str:
    if item.ocr_judgement in {"correct", "wrong"}:
        return item.ocr_judgement
    if not item.question_text or not item.child_answer:
        return "pending"
    if item.confidence < 0.75:
        return "pending"
    return "pending"


def _submission_response(submission_id: str) -> LearningSubmissionResponse:
    service = get_learning_service()
    snapshot = service.get_submission_snapshot(submission_id)
    submission = snapshot.submission
    fallback_counts = visual_fallback_counts(snapshot.items)
    pending_fallback_count = fallback_counts["pending_visual_fallback_count"]
    running_fallback_count = fallback_counts["running_visual_fallback_count"]
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
        detected_subject=submission.detected_subject,
        detected_task_type=submission.detected_task_type,
        detected_intent=submission.detected_intent,
        subject_confidence=submission.subject_confidence,
        route_to=submission.route_to,
        routing_evidence=submission.routing_evidence,
        guard_reason=submission.guard_reason,
        router_version=submission.router_version,
        needs_clarification=submission.needs_clarification,
        grade=submission.grade,
        source_type=submission.source_type,
        status=submission.status,
        item_count=submission.item_count,
        correct_count=submission.correct_count,
        wrong_count=submission.wrong_count,
        needs_manual_confirm_count=submission.needs_manual_confirm_count,
        pending_visual_fallback_count=pending_fallback_count,
        visual_fallback_active=bool(pending_fallback_count or running_fallback_count),
        active_queue_item_id=submission.active_queue_item_id,
        active_tutor_session=active_session,
        items=[_submission_item_response(item) for item in snapshot.items],
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
            **fallback_counts,
        },
    )


def _submission_item_response(item) -> LearningSubmissionItemResponse:
    rubric = item.data_json.get("basic_subject_rubric") if isinstance(item.data_json, dict) else None
    if not isinstance(rubric, dict):
        rubric = {}
    scores = rubric.get("rubric_scores") if isinstance(rubric.get("rubric_scores"), dict) else {}
    fallback = visual_fallback_state(item)
    return LearningSubmissionItemResponse(
        item_id=item.item_id,
        item_index=item.item_index,
        question_text=item.question_text,
        child_answer=item.child_answer,
        detected_subject=item.detected_subject,
        detected_task_type=item.detected_task_type,
        evaluation_mode=item.evaluation_mode,
        correct_answer=item.correct_answer,
        judge_result=item.judge_result,
        question_type_id=item.question_type_id,
        knowledge_point=item.knowledge_point,
        misconception_tag=item.misconception_tag,
        rubric_outcome=str(rubric.get("outcome") or ""),
        rubric_feedback=str(rubric.get("feedback_summary") or ""),
        rubric_scores=_normalize_rubric_scores(scores),
        evidence_points=_item_evidence_points(item),
        bbox=item.bbox_json,
        status=item.status,
        display_status=_item_display_status(item),
        visual_fallback_status=str(fallback.get("status") or ""),
        visual_fallback_reason=str(fallback.get("reason") or ""),
        visual_fallback_message=str(fallback.get("message") or ""),
        visual_fallback_attempts=int(fallback.get("attempts") or 0),
    )


def _item_evidence_points(item) -> list[str]:
    data_json = item.data_json if isinstance(item.data_json, dict) else {}
    points: list[str] = []
    trace = data_json.get("evidence_trace")
    if isinstance(trace, list):
        for entry in trace:
            if not isinstance(entry, dict):
                continue
            label = str(entry.get("label") or "").strip()
            text = str(entry.get("text") or "").strip()
            if not text:
                continue
            points.append(f"{label}：{text}" if label else text)
    if not points:
        route = data_json.get("question_type_route")
        if isinstance(route, dict):
            evidence = route.get("evidence")
            if isinstance(evidence, list):
                points.extend(str(item).strip() for item in evidence if str(item).strip())
        objective = data_json.get("objective_judge")
        if isinstance(objective, dict) and str(objective.get("evidence") or "").strip():
            points.append(str(objective.get("evidence")).strip())
        rubric = data_json.get("basic_subject_rubric")
        if isinstance(rubric, dict):
            criteria = rubric.get("criteria_evidence")
            if isinstance(criteria, list):
                points.extend(str(item).strip() for item in criteria if str(item).strip())
            feedback = str(rubric.get("feedback_summary") or "").strip()
            if feedback:
                points.append(feedback)
    deduped: list[str] = []
    seen: set[str] = set()
    for point in points:
        if point in seen:
            continue
        seen.add(point)
        deduped.append(point)
    return deduped[:4]


def _item_display_status(item) -> str:
    fallback = visual_fallback_state(item)
    if fallback.get("status") in {"pending", "running"}:
        return "fallback_running"
    judge_result = getattr(item.judge_result, "value", str(item.judge_result))
    if judge_result == "correct":
        return "correct"
    if judge_result == "wrong":
        return "wrong"
    return "pending"


def _normalize_rubric_scores(scores: dict) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for key, value in scores.items():
        try:
            normalized[str(key)] = int(value)
        except (TypeError, ValueError):
            normalized[str(key)] = 0
    return normalized


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
        child_id=child_id,
        filename=file.filename or "upload.bin",
        content=content,
        content_type=file.content_type or "application/octet-stream",
    )
    preview_image_path = draft.preview_image_path or ""
    return SubmissionPhotoDraftResponse(
        image_path=image_path,
        image_refs=[image_path],
        preview_image_path=preview_image_path,
        preview_image_url=_photo_preview_url(preview_image_path),
        raw_text=_photo_submission_raw_text_from_draft(draft),
        question_text=draft.question_text,
        child_answer=draft.child_answer,
        work_steps=draft.work_steps,
        confidence=draft.confidence,
        needs_confirmation=draft.needs_confirmation,
        items=[
            SubmissionPhotoDraftItemResponse(
                item_index=item.item_index,
                question_text=item.question_text,
                child_answer=item.child_answer,
                work_steps=item.work_steps,
                confidence=item.confidence,
                bbox=item.bbox.model_dump(mode="json") if item.bbox else None,
                ocr_action=item.source_action,
                ocr_source=draft.source,
                ocr_judgement=item.ocr_judgement,
                marking_source=item.marking_source,
                correct_answer=item.correct_answer,
                evidence_points=item.evidence_points,
                quality_warnings=item.quality_warnings,
                display_status=_photo_draft_item_display_status(item),
            )
            for item in draft.items
        ],
        ocr_plan=draft.data_json.get("ocr_plan", {}) if isinstance(draft.data_json, dict) else {},
        detected_regions=[region.model_dump(mode="json") for region in draft.detected_regions],
        quality_warnings=draft.quality_warnings,
        quality_message=draft.quality_message,
        preprocess_source=draft.preprocess_source,
        ocr_provider=draft.provider,
        ocr_model=draft.model,
        ocr_source=draft.source,
    )


@router.get("/submissions/photo-preview/{filename}")
def get_submission_photo_preview(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    if not safe_name or safe_name != filename:
        raise HTTPException(status_code=404, detail="Preview image not found")
    target = get_photo_review_service().artifact_root / safe_name
    try:
        resolved_root = get_photo_review_service().artifact_root.resolve()
        resolved_target = target.resolve()
    except OSError as exc:
        raise HTTPException(status_code=404, detail="Preview image not found") from exc
    if resolved_root not in resolved_target.parents or not resolved_target.exists():
        raise HTTPException(status_code=404, detail="Preview image not found")
    return FileResponse(resolved_target, media_type="image/jpeg")


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
            image_refs=request.image_refs,
            item_bboxes={
                item.item_index: item.bbox
                for item in request.draft_items
                if item.bbox
            },
            item_metadata={
                item.item_index: {
                    "ocr_action": item.ocr_action,
                    "ocr_source": item.ocr_source,
                    "ocr_judgement": item.ocr_judgement,
                    "marking_source": item.marking_source,
                    "correct_answer": item.correct_answer,
                    "evidence_points": item.evidence_points,
                    "display_status": item.display_status,
                    "quality_warnings": item.quality_warnings,
                }
                for item in request.draft_items
                if (
                    item.ocr_action
                    or item.ocr_source
                    or item.ocr_judgement
                    or item.marking_source
                    or item.correct_answer
                    or item.evidence_points
                    or item.display_status
                    or item.quality_warnings
                )
            },
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
    get_learning_service().confirm_submission(
        submission.submission_id,
        start_tutor=request.start_tutor,
    )
    return _submission_response(submission_id)


@router.post("/submissions/{submission_id}/visual-fallback/step", response_model=LearningSubmissionResponse)
async def run_submission_visual_fallback_step(
    submission_id: str,
    max_items: int = 1,
    x_session_token: str | None = Header(None, alias="X-Session-Token"),
) -> LearningSubmissionResponse:
    _authorize_submission_access(
        submission_id,
        child_id=None,
        session_token=x_session_token,
    )
    try:
        await get_learning_service().run_submission_visual_fallback_step(
            submission_id,
            max_items=max_items,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Learning submission not found") from None
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

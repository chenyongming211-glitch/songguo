from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Awaitable, Callable
from uuid import uuid4

from pydantic import BaseModel, Field

from songguo.backend.services.learning.input_safety import check_learning_input
from songguo.backend.services.learning.models import utc_now
from songguo.backend.services.learning.service import LearningService, _expected_answer
from songguo.backend.services.learning.store import InMemoryLearningStore


class OCRDraft(BaseModel):
    question_text: str = ""
    child_answer: str = ""
    work_steps: str = ""
    confidence: float = 0.0
    needs_confirmation: bool = True


class PhotoReview(BaseModel):
    review_id: str = Field(default_factory=lambda: f"pr_{uuid4().hex}")
    child_id: str
    subject: str = "math"
    grade: int = 3
    status: str
    image_path: str
    question_text: str = ""
    child_answer: str = ""
    work_steps: str = ""
    confidence: float = 0.0
    grading_result: str
    feedback: str = ""
    better_method: str = ""
    prerequisite_question: str = ""
    next_prompt: str = ""
    linked_session_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConfirmPhotoReviewRequest(BaseModel):
    question_text: str
    child_answer: str


class DeterministicOCRProvider:
    """Local deterministic OCR fallback for tests and offline development.

    It parses text fixtures shaped like:
    QUESTION: ...
    ANSWER: ...
    WORK: ...
    Real OCR / vision providers can replace this interface later.
    """

    def recognize(self, content: bytes, *, filename: str) -> OCRDraft:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            return OCRDraft()

        question = _extract_field(text, "QUESTION")
        answer = _extract_field(text, "ANSWER")
        work = _extract_field(text, "WORK")
        if not question:
            return OCRDraft()
        return OCRDraft(
            question_text=question,
            child_answer=answer,
            work_steps=work,
            confidence=0.92 if answer else 0.72,
            needs_confirmation=not answer,
        )


VisionFunc = Callable[..., Awaitable[str]]


class VisionOCRProvider:
    """Vision-model OCR provider behind an explicit opt-in seam."""

    def __init__(self, vision_func: VisionFunc | None = None, *, model: str | None = None) -> None:
        self.vision_func = vision_func
        self.model = model

    async def recognize_async(self, content: bytes, *, filename: str) -> OCRDraft:
        response = await self._call_vision_model(
            prompt=(
                "Extract a child's math homework from this image. "
                "Return strict JSON with keys: question_text, child_answer, "
                "work_steps, confidence. If uncertain, use empty strings and low confidence."
            ),
            image_data=_image_data_url(content, filename),
        )
        return _parse_ocr_response(response)

    async def _call_vision_model(self, *, prompt: str, image_data: str) -> str:
        if self.vision_func is not None:
            return await self.vision_func(prompt=prompt, image_data=image_data)

        from songguo.backend.services.learning.real_model_client import (
            OpenAICompatibleModelClient,
            get_llm_client,
        )

        llm_client = get_llm_client()
        if self.model:
            llm_client = OpenAICompatibleModelClient(
                llm_client.config.model_copy(update={"model": self.model})
            )
        vision_func = llm_client.get_vision_model_func()
        return await vision_func(prompt=prompt, image_data=image_data)


class PhotoReviewService:
    def __init__(
        self,
        *,
        store: InMemoryLearningStore,
        artifact_root: Path,
        ocr_provider: object | None = None,
        learning_service: LearningService | None = None,
    ) -> None:
        self.store = store
        self.artifact_root = artifact_root
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.ocr_provider = ocr_provider or DeterministicOCRProvider()
        self.learning_service = learning_service or LearningService(store=store)
        self._reviews: dict[str, PhotoReview] = {}

    def create_from_upload(
        self,
        *,
        child_id: str,
        subject: str = "math",
        grade: int = 3,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> PhotoReview:
        image_path = self._save_artifact(filename, content)
        draft = self._recognize(content, filename=filename)
        return self._create_from_draft(
            child_id=child_id,
            subject=subject,
            grade=grade,
            image_path=image_path,
            draft=draft,
        )

    async def create_from_upload_async(
        self,
        *,
        child_id: str,
        subject: str = "math",
        grade: int = 3,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> PhotoReview:
        image_path = self._save_artifact(filename, content)
        draft = await self._recognize_async(content, filename=filename)
        return self._create_from_draft(
            child_id=child_id,
            subject=subject,
            grade=grade,
            image_path=image_path,
            draft=draft,
        )

    async def recognize_submission_draft_async(
        self,
        *,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> tuple[str, OCRDraft]:
        image_path = self._save_artifact(filename, content)
        draft = await self._recognize_async(content, filename=filename)
        return image_path, draft

    async def _recognize_async(self, content: bytes, *, filename: str) -> OCRDraft:
        try:
            recognize_async = getattr(self.ocr_provider, "recognize_async", None)
            if recognize_async:
                return await recognize_async(content, filename=filename)
            return self._recognize(content, filename=filename)
        except Exception:
            return OCRDraft()

    def _recognize(self, content: bytes, *, filename: str) -> OCRDraft:
        try:
            return self.ocr_provider.recognize(content, filename=filename)
        except Exception:
            return OCRDraft()

    def _create_from_draft(
        self,
        *,
        child_id: str,
        subject: str,
        grade: int,
        image_path: str,
        draft: OCRDraft,
    ) -> PhotoReview:
        review = self._grade_or_request_confirmation(
            child_id=child_id,
            subject=subject,
            grade=grade,
            image_path=image_path,
            question_text=draft.question_text,
            child_answer=draft.child_answer,
            work_steps=draft.work_steps,
            confidence=draft.confidence,
            force_confirmation=draft.needs_confirmation,
        )
        self._reviews[review.review_id] = review
        self._persist_review(review)
        return review

    def confirm(self, review_id: str, *, question_text: str, child_answer: str) -> PhotoReview:
        existing = self.get(review_id)
        review = self._grade_or_request_confirmation(
            child_id=existing.child_id,
            subject=existing.subject,
            grade=existing.grade,
            image_path=existing.image_path,
            question_text=question_text,
            child_answer=child_answer,
            work_steps=existing.work_steps,
            confidence=1.0,
            force_confirmation=False,
        )
        review.review_id = review_id
        review.created_at = existing.created_at
        review.updated_at = utc_now()
        self._reviews[review_id] = review
        self._persist_review(review)
        return review

    def get(self, review_id: str) -> PhotoReview:
        review = self._reviews.get(review_id)
        if review is not None:
            return review
        loaded = self.store.get_photo_review(review_id)
        if loaded is None:
            raise KeyError(review_id)
        review = PhotoReview.model_validate(loaded)
        self._reviews[review_id] = review
        return review

    def _persist_review(self, review: PhotoReview) -> None:
        self.store.save_photo_review(review.model_dump(mode="json"))

    def _save_artifact(self, filename: str, content: bytes) -> str:
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename or "upload.bin")
        target = self.artifact_root / f"{uuid4().hex}_{safe_name}"
        target.write_bytes(content)
        return str(target)

    def _grade_or_request_confirmation(
        self,
        *,
        child_id: str,
        subject: str,
        grade: int,
        image_path: str,
        question_text: str,
        child_answer: str,
        work_steps: str,
        confidence: float,
        force_confirmation: bool,
    ) -> PhotoReview:
        if force_confirmation or not question_text or not child_answer:
            return PhotoReview(
                child_id=child_id,
                subject=subject,
                grade=grade,
                status="needs_confirmation",
                image_path=image_path,
                question_text=question_text,
                child_answer=child_answer,
                work_steps=work_steps,
                confidence=confidence,
                grading_result="needs_confirmation",
                feedback="图片识别不够确定，请先确认题目和孩子答案。",
            )

        for value in (question_text, child_answer):
            input_verdict = check_learning_input(value)
            if not input_verdict.allowed:
                raise ValueError(input_verdict.message)

        expected = _expected_answer(question_text)
        if expected is not None and child_answer.strip() == expected:
            return PhotoReview(
                child_id=child_id,
                subject=subject,
                grade=grade,
                status="graded_correct",
                image_path=image_path,
                question_text=question_text,
                child_answer=child_answer,
                work_steps=work_steps,
                confidence=confidence,
                grading_result="correct",
                feedback="做对了。孩子已经完成了这道题的关键计算。",
                better_method="更稳的方法是先说清楚每一步为什么这样算，再检查单位和抄写。",
            )

        created = self.learning_service.create_session(
            child_id=child_id,
            subject=subject,
            grade=grade,
            question_text=question_text,
        )
        attempt = self.learning_service.submit_attempt(created.session_id, child_answer=child_answer)
        return PhotoReview(
            child_id=child_id,
            subject=subject,
            grade=grade,
            status="remediation_ready",
            image_path=image_path,
            question_text=question_text,
            child_answer=child_answer,
            work_steps=work_steps,
            confidence=confidence,
            grading_result="incorrect" if expected is not None else "unknown",
            feedback="这道题需要引导，先从更简单的一步开始。",
            prerequisite_question=_prerequisite_question(question_text),
            next_prompt=attempt.message,
            linked_session_id=created.session_id,
        )


def _extract_field(text: str, field_name: str) -> str:
    pattern = re.compile(
        rf"^{field_name}\s*:\s*(.*?)(?=\n[A-Z_]+\s*:|\Z)",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def _prerequisite_question(question_text: str) -> str:
    return "先把这道题拆成一个更简单的小问题，再说第一步准备怎么算。"


def _image_data_url(content: bytes, filename: str) -> str:
    mime_type = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }.get(filename.rsplit(".", 1)[-1].lower() if "." in filename else "", "image/png")
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _parse_ocr_response(response: str) -> OCRDraft:
    text = response.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return OCRDraft()

    question = str(payload.get("question_text") or "").strip()
    answer = str(payload.get("child_answer") or "").strip()
    work = str(payload.get("work_steps") or "").strip()
    try:
        confidence = float(payload.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    return OCRDraft(
        question_text=question,
        child_answer=answer,
        work_steps=work,
        confidence=confidence,
        needs_confirmation=not question or not answer or confidence < 0.8,
    )

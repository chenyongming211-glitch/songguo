from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
import inspect
import json
import os
from pathlib import Path
import re
from time import perf_counter
from typing import TYPE_CHECKING, Any, Awaitable, Callable
from uuid import uuid4

import cv2
import numpy as np
from pydantic import BaseModel, Field

from songguo.backend.services.learning.answer_binding import bind_grouped_comparison_answers
from songguo.backend.services.learning.input_safety import check_learning_input
from songguo.backend.services.learning.models import utc_now
from songguo.backend.services.learning.photo_preprocess import (
    HomeworkPhotoAnalysis,
    analyze_homework_photo,
)
from songguo.backend.services.learning.education_ocr_router import (
    EducationOcrAction,
    EducationOcrPlan,
    EducationOcrRouter,
    OcrQualitySignal,
)
from songguo.backend.services.learning.store import InMemoryLearningStore
from songguo.backend.services.learning.submission_intake import parse_text_submission

if TYPE_CHECKING:
    from songguo.backend.services.learning.service import LearningService


class ImageBBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class OCRItemDraft(BaseModel):
    item_index: int
    question_text: str
    child_answer: str = ""
    work_steps: str = ""
    confidence: float = 0.0
    bbox: ImageBBox | None = None
    source_action: str = ""
    ocr_judgement: str = ""
    marking_source: str = ""
    correct_answer: str = ""
    evidence_points: list[str] = Field(default_factory=list)
    quality_warnings: list[str] = Field(default_factory=list)


class OCRDraft(BaseModel):
    raw_text: str = ""
    question_text: str = ""
    child_answer: str = ""
    work_steps: str = ""
    confidence: float = 0.0
    needs_confirmation: bool = True
    items: list[OCRItemDraft] = Field(default_factory=list)
    detected_regions: list[ImageBBox] = Field(default_factory=list)
    preview_image_path: str = ""
    quality_warnings: list[str] = Field(default_factory=list)
    quality_message: str = ""
    preprocess_source: str = ""
    provider: str = "deterministic_ocr"
    model: str = "local_fixture"
    source: str = "deterministic"
    data_json: dict[str, Any] = Field(default_factory=dict)


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
        raw_text = _build_raw_text(
            question_text=question,
            child_answer=answer,
            work_steps=work,
        )
        items = []
        if answer or work:
            items = [
                OCRItemDraft(
                    item_index=1,
                    question_text=question.strip(),
                    child_answer=answer.strip(),
                    work_steps=work.strip(),
                    confidence=0.92 if answer else 0.72,
                )
            ]
        else:
            items = _items_from_text(raw_text)
        if not items and question:
            items = [
                OCRItemDraft(
                    item_index=1,
                    question_text=question.strip(),
                    child_answer=answer.strip(),
                    work_steps=work.strip(),
                    confidence=0.92 if answer else 0.72,
                )
            ]
        items = _with_default_bboxes(items)
        confidence = _overall_confidence(items, default=0.92 if answer else 0.72)
        return OCRDraft(
            raw_text=raw_text,
            question_text=question,
            child_answer=answer,
            work_steps=work,
            confidence=confidence,
            needs_confirmation=not items or any(not item.child_answer for item in items) or confidence < 0.8,
            items=items,
        )


VisionFunc = Callable[..., Awaitable[str]]
AliyunEduOCRFunc = Callable[..., Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class _ImageSegment:
    content: bytes
    filename: str
    x: int
    y: int
    width: int
    height: int
    full_width: int
    full_height: int


class VisionOCRProvider:
    """Vision-model OCR provider behind an explicit opt-in seam."""

    def __init__(self, vision_func: VisionFunc | None = None, *, model: str | None = None) -> None:
        self.vision_func = vision_func
        self.model = model
        self._resolved_provider = ""
        self._resolved_model = ""

    async def recognize_async(
        self,
        content: bytes,
        *,
        filename: str,
        region_hints: list[ImageBBox] | None = None,
    ) -> OCRDraft:
        hint_text = _format_region_hints(region_hints or [])
        prompt = (
            "OCR this homework image. Return JSON only, no markdown: "
            '{"raw_text":"","confidence":0.0,'
            '"items":[{"item_index":1,"question_text":"","child_answer":"",'
            '"work_steps":"","confidence":0.0,'
            '"bbox":{"x":0,"y":0,"width":0,"height":0}}]}. '
            "Use 0-1000 image coordinates for bbox. Split every visible question. "
            f"{hint_text}"
            "Keep uncertainty as empty strings and low confidence."
        )
        segments = _wide_image_segments(content, filename=filename)
        if len(segments) > 1:
            try:
                segment_drafts = await asyncio.gather(
                    *[
                        self._recognize_segment_async(
                            segment,
                            prompt=prompt,
                        )
                        for segment in segments
                    ]
                )
                parsed = _merge_segment_drafts(segment_drafts, segments)
                if parsed.items or parsed.raw_text:
                    return parsed.model_copy(
                        update={
                            "provider": self._resolved_provider or "vision_ocr",
                            "model": self.model or self._resolved_model or "configured_vision_model",
                            "source": "vision_parallel_split",
                        }
                    )
            except Exception:
                pass
        response = await self._call_vision_model(
            prompt=prompt,
            image_data=_image_data_url(content, filename),
        )
        parsed = _parse_ocr_response(response)
        return parsed.model_copy(
            update={
                "provider": self._resolved_provider or "vision_ocr",
                "model": self.model or self._resolved_model or "configured_vision_model",
                "source": "vision",
            }
        )

    async def _recognize_segment_async(self, segment: _ImageSegment, *, prompt: str) -> OCRDraft:
        response = await self._call_vision_model(
            prompt=prompt,
            image_data=_image_data_url(segment.content, segment.filename),
        )
        return _parse_ocr_response(response)

    async def _call_vision_model(self, *, prompt: str, image_data: str) -> str:
        if self.vision_func is not None:
            return await self.vision_func(prompt=prompt, image_data=image_data)

        from songguo.backend.services.learning.real_model_client import (
            OpenAICompatibleModelClient,
            load_vision_model_config,
        )

        config = load_vision_model_config(model=self.model)
        self._resolved_provider = config.binding
        self._resolved_model = config.model
        llm_client = OpenAICompatibleModelClient(config)
        vision_func = llm_client.get_vision_model_func()
        return await vision_func(prompt=prompt, image_data=image_data)


class AliyunEduOCRProvider:
    """Aliyun OCR education-scenario provider behind an explicit opt-in seam."""

    use_preprocessed_content = True

    def __init__(
        self,
        *,
        config: Any | None = None,
        client_func: AliyunEduOCRFunc | None = None,
        fallback_provider: object | None = None,
    ) -> None:
        self.config = config
        self.client_func = client_func
        self.fallback_provider = fallback_provider

    async def recognize_async(
        self,
        content: bytes,
        *,
        filename: str,
        region_hints: list[ImageBBox] | None = None,
    ) -> OCRDraft:
        config = self._resolved_config()
        router = EducationOcrRouter(config)
        plan = router.initial_plan(filename=filename)
        action = plan.primary_action.value
        last_error: Exception | None = None
        try:
            draft = await self._run_ocr_action(
                config=config,
                action=action,
                content=content,
                filename=filename,
                region_hints=region_hints or [],
                plan=plan,
            )
            if draft.items or draft.raw_text:
                secondary_actions = self._secondary_actions_for_draft(
                    router=router,
                    config=config,
                    plan=plan,
                    draft=draft,
                )
                if secondary_actions and plan.primary_action == EducationOcrAction.PAPER_CUT:
                    if secondary_actions[0] == EducationOcrAction.ORAL_CALCULATION:
                        return await self._recognize_paper_cut_oral_judgement(
                            config=config,
                            content=content,
                            filename=filename,
                            region_hints=region_hints or [],
                            plan={
                                **plan.model_dump(),
                                "secondary_actions": [action.value for action in secondary_actions],
                            },
                        )
                    return await self._recognize_paper_cut_hybrid_text(
                        cut_draft=draft,
                        config=config,
                        content=content,
                        filename=filename,
                        region_hints=region_hints or [],
                        plan={
                            **plan.model_dump(),
                            "secondary_actions": [action.value for action in secondary_actions],
                        },
                    )
                if secondary_actions:
                    return _draft_with_ocr_plan(
                        draft,
                        {
                            **plan.model_dump(),
                            "secondary_actions": [action.value for action in secondary_actions],
                        },
                    )
                return _draft_with_ocr_plan(draft, plan.model_dump())

            for fallback_action in _aliyun_edu_attempt_actions(action)[1:]:
                fallback_plan = _ocr_plan_for_action(
                    fallback_action,
                    reason="empty_primary_fallback",
                    subject_hint=str(getattr(config, "subject", "default") or "default"),
                )
                fallback_draft = await self._run_ocr_action(
                    config=config,
                    action=fallback_action,
                    content=content,
                    filename=filename,
                    region_hints=region_hints or [],
                    plan=fallback_plan,
                )
                if fallback_draft.items or fallback_draft.raw_text:
                    return fallback_draft
        except Exception as exc:
            last_error = exc

        fallback = self._resolved_fallback_provider(config)
        if fallback is not None:
            return await _recognize_with_provider_async(
                fallback,
                content,
                filename=filename,
                region_hints=region_hints or [],
            )
        if last_error is not None:
            return _aliyun_edu_error_draft(action=action, error=last_error)
        return OCRDraft(
            provider="aliyun_edu_ocr",
            model=action,
            source="aliyun_edu_empty",
            data_json={"ocr_plan": plan.model_dump()},
        )

    async def _run_ocr_action(
        self,
        *,
        config: Any,
        action: str,
        content: bytes,
        filename: str,
        region_hints: list[ImageBBox],
        plan: EducationOcrPlan,
    ) -> OCRDraft:
        payload = await self._call_edu_ocr(
            config=config,
            action=action,
            content=content,
            filename=filename,
            region_hints=region_hints,
        )
        draft = _parse_aliyun_edu_ocr_response(payload, action=action)
        return _draft_with_ocr_plan(draft, plan.model_dump())

    def _secondary_actions_for_draft(
        self,
        *,
        router: EducationOcrRouter,
        config: Any,
        plan: EducationOcrPlan,
        draft: OCRDraft,
    ) -> list[EducationOcrAction]:
        max_actions = int(getattr(config, "max_secondary_actions", 1) or 0)
        if max_actions <= 0:
            return []
        if (
            plan.primary_action == EducationOcrAction.PAPER_CUT
            and plan.reason.startswith("auto")
            and _draft_looks_like_oral_calculation_page(draft)
        ):
            return [EducationOcrAction.ORAL_CALCULATION][:max_actions]
        if plan.reason.startswith("auto"):
            actions = router.secondary_actions(
                OcrQualitySignal(
                    primary_action=plan.primary_action,
                    item_count=len(draft.items),
                    answer_count=sum(1 for item in draft.items if item.child_answer.strip()),
                    raw_text_length=len(draft.raw_text),
                    confidence=draft.confidence,
                )
            )
        elif _should_run_paper_cut_text_fallback(
            draft=draft,
            config=config,
            action=plan.primary_action.value,
        ):
            actions = [EducationOcrAction.PAPER_OCR]
        else:
            actions = []
        return actions[:max_actions]

    async def _recognize_paper_cut_oral_judgement(
        self,
        *,
        config: Any,
        content: bytes,
        filename: str,
        region_hints: list[ImageBBox],
        plan: dict[str, Any],
    ) -> OCRDraft:
        payload = await self._call_edu_ocr(
            config=config,
            action="RecognizeEduOralCalculation",
            content=content,
            filename=filename,
            region_hints=region_hints,
        )
        oral_draft = _parse_aliyun_edu_ocr_response(payload, action="RecognizeEduOralCalculation")
        if not oral_draft.items:
            return OCRDraft(
                provider="aliyun_edu_ocr",
                model="RecognizeEduPaperCut+RecognizeEduOralCalculation",
                source="aliyun_edu_paper_cut_oral_judgement",
                data_json={"ocr_plan": plan},
            )
        return oral_draft.model_copy(
            update={
                "model": "RecognizeEduPaperCut+RecognizeEduOralCalculation",
                "source": "aliyun_edu_paper_cut_oral_judgement",
                "data_json": {
                    **oral_draft.data_json,
                    "ocr_plan": plan,
                },
            }
        )

    async def _recognize_paper_cut_hybrid_text(
        self,
        *,
        cut_draft: OCRDraft,
        config: Any,
        content: bytes,
        filename: str,
        region_hints: list[ImageBBox],
        plan: dict[str, Any] | None = None,
    ) -> OCRDraft:
        payload = await self._call_edu_ocr(
            config=config,
            action="RecognizeEduPaperOcr",
            content=content,
            filename=filename,
            region_hints=region_hints,
        )
        text_draft = _parse_aliyun_edu_ocr_response(payload, action="RecognizeEduPaperOcr")
        if not text_draft.items:
            return cut_draft
        merged_items = _merge_payload_items_with_raw_text_items(cut_draft.items, text_draft.items)
        raw_text = text_draft.raw_text or cut_draft.raw_text
        confidence = max(cut_draft.confidence, text_draft.confidence)
        return cut_draft.model_copy(
            update={
                "raw_text": raw_text,
                "question_text": merged_items[0].question_text if merged_items else cut_draft.question_text,
                "child_answer": merged_items[0].child_answer if merged_items else cut_draft.child_answer,
                "work_steps": merged_items[0].work_steps if merged_items else cut_draft.work_steps,
                "confidence": confidence,
                "needs_confirmation": (
                    not merged_items
                    or any(not item.question_text or not item.child_answer for item in merged_items)
                    or confidence < 0.75
                ),
                "items": merged_items,
                "model": "RecognizeEduPaperCut+RecognizeEduPaperOcr",
                "source": "aliyun_edu_paper_cut_hybrid_text",
                "data_json": {
                    **cut_draft.data_json,
                    "ocr_plan": plan or cut_draft.data_json.get("ocr_plan", {}),
                },
            }
        )

    async def _call_edu_ocr(
        self,
        *,
        config: Any,
        action: str,
        content: bytes,
        filename: str,
        region_hints: list[ImageBBox],
    ) -> dict[str, Any]:
        if self.client_func is not None:
            result = self.client_func(
                config=config,
                action=action,
                content=content,
                filename=filename,
                image_type=config.image_type,
                subject=config.subject,
                region_hints=region_hints,
            )
            if inspect.isawaitable(result):
                return await result
            return result
        return await asyncio.to_thread(
            _call_aliyun_edu_ocr_sync,
            config=config,
            action=action,
            content=content,
            filename=filename,
        )

    def _resolved_config(self):
        if self.config is not None:
            return self.config
        from songguo.backend.services.learning.real_model_client import load_aliyun_edu_ocr_config

        return load_aliyun_edu_ocr_config()

    def _resolved_fallback_provider(self, config: Any) -> object | None:
        if self.fallback_provider is not None:
            return self.fallback_provider
        fallback_name = str(getattr(config, "fallback_provider", "") or "").strip().lower()
        if fallback_name in {"", "none", "off", "false", "0"}:
            return None
        if fallback_name == "vision":
            return VisionOCRProvider()
        if fallback_name == "deterministic":
            return DeterministicOCRProvider()
        return None


class PhotoReviewService:
    def __init__(
        self,
        *,
        store: InMemoryLearningStore,
        artifact_root: Path,
        ocr_provider: object | None = None,
        learning_service: "LearningService" | None = None,
    ) -> None:
        self.store = store
        self.artifact_root = artifact_root
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.ocr_provider = ocr_provider or DeterministicOCRProvider()
        self.learning_service = learning_service or _build_learning_service(store)
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
        analysis = _analyze_upload(content, filename=filename, content_type=content_type)
        preview_image_path = self._save_processed_artifact(filename, analysis)
        started_at = perf_counter()
        draft = self._recognize(
            _ocr_input_content(self.ocr_provider, original_content=content, analysis=analysis),
            filename=filename,
            region_hints=_processed_region_hints(analysis),
        )
        draft = _attach_preprocess_analysis(draft, analysis, preview_image_path=preview_image_path)
        self._record_ocr_observation(
            child_id=child_id,
            image_path=image_path,
            filename=filename,
            content_type=content_type,
            draft=draft,
            latency_ms=int((perf_counter() - started_at) * 1000),
        )
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
        analysis = _analyze_upload(content, filename=filename, content_type=content_type)
        preview_image_path = self._save_processed_artifact(filename, analysis)
        started_at = perf_counter()
        draft = await self._recognize_async(
            _ocr_input_content(self.ocr_provider, original_content=content, analysis=analysis),
            filename=filename,
            region_hints=_processed_region_hints(analysis),
        )
        draft = _attach_preprocess_analysis(draft, analysis, preview_image_path=preview_image_path)
        self._record_ocr_observation(
            child_id=child_id,
            image_path=image_path,
            filename=filename,
            content_type=content_type,
            draft=draft,
            latency_ms=int((perf_counter() - started_at) * 1000),
        )
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
        child_id: str,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> tuple[str, OCRDraft]:
        image_path = self._save_artifact(filename, content)
        analysis = _analyze_upload(content, filename=filename, content_type=content_type)
        preview_image_path = self._save_processed_artifact(filename, analysis)
        started_at = perf_counter()
        draft = await self._recognize_async(
            _ocr_input_content(self.ocr_provider, original_content=content, analysis=analysis),
            filename=filename,
            region_hints=_processed_region_hints(analysis),
        )
        draft = _attach_preprocess_analysis(draft, analysis, preview_image_path=preview_image_path)
        self._record_ocr_observation(
            child_id=child_id,
            image_path=image_path,
            filename=filename,
            content_type=content_type,
            draft=draft,
            latency_ms=int((perf_counter() - started_at) * 1000),
        )
        return image_path, draft

    async def _recognize_async(
        self,
        content: bytes,
        *,
        filename: str,
        region_hints: list[ImageBBox] | None = None,
    ) -> OCRDraft:
        try:
            recognize_async = getattr(self.ocr_provider, "recognize_async", None)
            if recognize_async:
                if _call_accepts_region_hints(recognize_async):
                    return await recognize_async(content, filename=filename, region_hints=region_hints or [])
                return await recognize_async(content, filename=filename)
            return self._recognize(content, filename=filename, region_hints=region_hints)
        except Exception:
            return OCRDraft()

    def _recognize(
        self,
        content: bytes,
        *,
        filename: str,
        region_hints: list[ImageBBox] | None = None,
    ) -> OCRDraft:
        try:
            recognize = self.ocr_provider.recognize
            if _call_accepts_region_hints(recognize):
                return recognize(content, filename=filename, region_hints=region_hints or [])
            return recognize(content, filename=filename)
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

    def _record_ocr_observation(
        self,
        *,
        child_id: str,
        image_path: str,
        filename: str,
        content_type: str,
        draft: OCRDraft,
        latency_ms: int,
    ) -> None:
        has_content = bool(draft.items or draft.raw_text or draft.question_text)
        self.store.record_ai_call(
            child_id=child_id,
            session_id=image_path,
            provider=draft.provider,
            model=draft.model,
            operation="photo_ocr.recognize",
            token_estimate=0,
            status="success" if has_content else "empty",
            agent=type(self.ocr_provider).__name__,
            latency_ms=latency_ms,
            confidence=draft.confidence,
            metadata={
                "filename": filename,
                "content_type": content_type,
                "source": draft.source,
                "item_count": len(draft.items),
                "needs_confirmation": draft.needs_confirmation,
                "raw_text_length": len(draft.raw_text),
                "preprocess_source": draft.preprocess_source,
                "quality_warnings": draft.quality_warnings,
                "quality_message": draft.quality_message,
                "detected_region_count": len(draft.detected_regions),
            },
        )

    def _save_artifact(self, filename: str, content: bytes) -> str:
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename or "upload.bin")
        target = self.artifact_root / f"{uuid4().hex}_{safe_name}"
        target.write_bytes(content)
        return str(target)

    def _save_processed_artifact(self, filename: str, analysis: HomeworkPhotoAnalysis) -> str:
        preview_content = analysis.preview_content or analysis.processed_content
        if not preview_content or analysis.source == "non_image_fixture":
            return ""
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename or "upload.jpg")
        stem = safe_name.rsplit(".", 1)[0] if "." in safe_name else safe_name
        target = self.artifact_root / f"{uuid4().hex}_{stem}_opencv_preview.jpg"
        target.write_bytes(preview_content)
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

        expected = _expected_answer_lazy(question_text)
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


def _build_learning_service(store: InMemoryLearningStore):
    from songguo.backend.services.learning.service import LearningService

    return LearningService(store=store)


def _expected_answer_lazy(question_text: str) -> str | None:
    from songguo.backend.services.learning.service import _expected_answer

    return _expected_answer(question_text)


def _attach_preprocess_analysis(
    draft: OCRDraft,
    analysis: HomeworkPhotoAnalysis,
    *,
    preview_image_path: str = "",
) -> OCRDraft:
    regions = [
        ImageBBox(
            x=region.x,
            y=region.y,
            width=region.width,
            height=region.height,
        )
        for region in (analysis.processed_question_regions or analysis.question_regions)
    ]
    items = list(draft.items)
    if items:
        mapped_items = []
        for index, item in enumerate(items):
            bbox = item.bbox
            if bbox is None and regions and len(regions) == len(items):
                bbox = regions[index]
            mapped_items.append(item.model_copy(update={"bbox": bbox}))
        items = mapped_items
    quality_warnings = _append_unique(
        list(draft.quality_warnings),
        list(analysis.quality_warnings),
    )
    quality_message = draft.quality_message or analysis.quality_message
    if not items and (draft.raw_text or draft.question_text):
        quality_warnings = _append_unique(quality_warnings, ["no_structured_items"])
        quality_message = quality_message or (
            "识别到了部分文字，但没有拆出清晰题目和答案，请先核对识别内容；"
            "如果题目或答案缺失，建议重拍。"
        )
    return draft.model_copy(
        update={
            "items": items,
            "detected_regions": regions,
            "preview_image_path": preview_image_path,
            "quality_warnings": quality_warnings,
            "quality_message": quality_message,
            "preprocess_source": analysis.source,
        }
    )


def _analyze_upload(content: bytes, *, filename: str, content_type: str) -> HomeworkPhotoAnalysis:
    if not _looks_like_image_upload(filename=filename, content_type=content_type):
        return HomeworkPhotoAnalysis(
            processed_content=content,
            mime_type=content_type or _mime_type_for_filename(filename),
            source="non_image_fixture",
        )
    return analyze_homework_photo(content, filename=filename)


def _ocr_input_content(
    provider: object,
    *,
    original_content: bytes,
    analysis: HomeworkPhotoAnalysis,
) -> bytes:
    _ = provider
    return analysis.processed_content or original_content


def _looks_like_image_upload(*, filename: str, content_type: str) -> bool:
    normalized_content_type = (content_type or "").split(";", 1)[0].strip().lower()
    if normalized_content_type.startswith("image/"):
        return True
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return suffix in {"jpg", "jpeg", "png", "webp"}


def _processed_region_hints(analysis: HomeworkPhotoAnalysis) -> list[ImageBBox]:
    regions = analysis.processed_question_regions or analysis.question_regions
    return [
        ImageBBox(x=region.x, y=region.y, width=region.width, height=region.height)
        for region in regions
    ]


def _append_unique(values: list[str], additions: list[str]) -> list[str]:
    result = list(values)
    for value in additions:
        if value not in result:
            result.append(value)
    return result


def _call_accepts_region_hints(callable_obj) -> bool:
    try:
        return "region_hints" in inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False


async def _recognize_with_provider_async(
    provider: object,
    content: bytes,
    *,
    filename: str,
    region_hints: list[ImageBBox],
) -> OCRDraft:
    try:
        recognize_async = getattr(provider, "recognize_async", None)
        if recognize_async:
            if _call_accepts_region_hints(recognize_async):
                result = recognize_async(content, filename=filename, region_hints=region_hints)
            else:
                result = recognize_async(content, filename=filename)
            return await result if inspect.isawaitable(result) else result
        recognize = getattr(provider, "recognize", None)
        if recognize:
            if _call_accepts_region_hints(recognize):
                return recognize(content, filename=filename, region_hints=region_hints)
            return recognize(content, filename=filename)
    except Exception:
        return OCRDraft()
    return OCRDraft()


def _format_region_hints(region_hints: list[ImageBBox]) -> str:
    if not region_hints:
        return ""
    payload = [
        {
            "item_index": index,
            "bbox": hint.model_dump(mode="json"),
        }
        for index, hint in enumerate(region_hints[:12], start=1)
    ]
    return (
        "OpenCV candidate question regions are provided below; verify, correct, "
        "merge or split them before returning final items: "
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}. "
    )


def _map_processed_bbox_to_original(
    bbox: ImageBBox,
    analysis: HomeworkPhotoAnalysis,
) -> ImageBBox:
    if (
        analysis.original_width <= 0
        or analysis.original_height <= 0
        or analysis.processed_width <= 0
        or analysis.processed_height <= 0
    ):
        return bbox
    left = analysis.processed_x + (bbox.x / 1000.0) * analysis.processed_width
    top = analysis.processed_y + (bbox.y / 1000.0) * analysis.processed_height
    width = (bbox.width / 1000.0) * analysis.processed_width
    height = (bbox.height / 1000.0) * analysis.processed_height
    x = max(0, min(999, round((left / analysis.original_width) * 1000)))
    y = max(0, min(999, round((top / analysis.original_height) * 1000)))
    mapped_width = max(1, min(1000 - x, round((width / analysis.original_width) * 1000)))
    mapped_height = max(1, min(1000 - y, round((height / analysis.original_height) * 1000)))
    return ImageBBox(x=x, y=y, width=mapped_width, height=mapped_height)


def _image_data_url(content: bytes, filename: str) -> str:
    optimized, mime_type = _optimized_image_bytes(content, filename=filename)
    encoded = base64.b64encode(optimized).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _wide_image_segments(content: bytes, *, filename: str) -> list[_ImageSegment]:
    if not _vision_split_wide_images_enabled():
        return []
    image = _decode_cv_image(content)
    if image is None:
        return []

    height, width = image.shape[:2]
    if height <= 0 or width < _vision_split_min_width():
        return []
    if width / height < _vision_split_wide_ratio():
        return []

    overlap = max(0, min(width // 8, int(width * _vision_split_overlap_ratio())))
    midpoint = width // 2
    boxes = [
        (0, 0, min(width, midpoint + overlap), height),
        (max(0, midpoint - overlap), 0, width, height),
    ]
    segments: list[_ImageSegment] = []
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    for index, box in enumerate(boxes, start=1):
        left, top, right, bottom = box
        crop = image[top:bottom, left:right]
        encoded = _encode_cv_jpeg(crop, quality=72)
        if not encoded:
            return []
        segments.append(
            _ImageSegment(
                content=encoded,
                filename=f"{stem}_segment_{index}.jpg",
                x=left,
                y=top,
                width=max(1, right - left),
                height=max(1, bottom - top),
                full_width=width,
                full_height=height,
            )
        )
    return segments


def _vision_split_wide_images_enabled() -> bool:
    value = os.getenv("SONGGUO_VISION_SPLIT_WIDE_IMAGES", "0").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _vision_split_wide_ratio() -> float:
    raw = os.getenv("SONGGUO_VISION_SPLIT_WIDE_RATIO", "").strip()
    try:
        return max(1.1, min(2.4, float(raw or "1.25")))
    except ValueError:
        return 1.25


def _vision_split_min_width() -> int:
    raw = os.getenv("SONGGUO_VISION_SPLIT_MIN_WIDTH", "").strip()
    try:
        return max(900, min(2600, int(raw or "1200")))
    except ValueError:
        return 1200


def _vision_split_overlap_ratio() -> float:
    raw = os.getenv("SONGGUO_VISION_SPLIT_OVERLAP_RATIO", "").strip()
    try:
        return max(0.0, min(0.08, float(raw or "0.03")))
    except ValueError:
        return 0.03


def _optimized_image_bytes(content: bytes, *, filename: str) -> tuple[bytes, str]:
    original_mime = _mime_type_for_filename(filename)
    image = _decode_cv_image(content)
    if image is None:
        return content, original_mime

    max_side = _vision_max_image_side()
    height, width = image.shape[:2]
    longest_side = max(width, height)
    if longest_side > max_side:
        scale = max_side / float(longest_side)
        target_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    optimized = _encode_cv_jpeg(image, quality=72)
    if not optimized:
        return content, original_mime
    if len(optimized) >= len(content):
        return content, original_mime
    return optimized, "image/jpeg"


def _decode_cv_image(content: bytes):
    if not content:
        return None
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None
    return image


def _encode_cv_jpeg(image, *, quality: int) -> bytes:
    if image is None or image.size == 0:
        return b""
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return encoded.tobytes() if ok else b""


def _vision_max_image_side() -> int:
    raw = os.getenv("SONGGUO_VISION_MAX_IMAGE_SIDE", "").strip()
    try:
        return max(480, min(1800, int(raw or "960")))
    except ValueError:
        return 960


def _mime_type_for_filename(filename: str) -> str:
    mime_type = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }.get(filename.rsplit(".", 1)[-1].lower() if "." in filename else "", "image/png")
    return mime_type


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
    raw_text = str(payload.get("raw_text") or "").strip()
    if not raw_text:
        raw_text = _build_raw_text(question_text=question, child_answer=answer, work_steps=work)
    confidence = _safe_confidence(payload.get("confidence"))
    items = _items_from_payload(payload.get("items"))
    raw_text_items = _items_from_text(raw_text) if raw_text else []
    if items and raw_text_items:
        items = _merge_payload_items_with_raw_text_items(items, raw_text_items)
    elif not items:
        items = raw_text_items
    if not items and question:
        items = [
            OCRItemDraft(
                item_index=1,
                question_text=question,
                child_answer=answer,
                work_steps=work,
                confidence=confidence,
            )
        ]
    if items and not question:
        question = items[0].question_text
    if items and not answer:
        answer = items[0].child_answer
    if items and not work:
        work = items[0].work_steps
    if confidence <= 0:
        confidence = _overall_confidence(items)
    if _should_enrich_raw_text_from_items(
        raw_text=raw_text,
        items=items,
        raw_text_items=raw_text_items,
    ):
        raw_text = _raw_text_from_items(items)
    items = _with_default_bboxes(items)

    return OCRDraft(
        raw_text=raw_text,
        question_text=question,
        child_answer=answer,
        work_steps=work,
        confidence=confidence,
        needs_confirmation=not items or any(not item.question_text or not item.child_answer for item in items) or confidence < 0.8,
        items=items,
    )


def _parse_aliyun_edu_ocr_response(payload: object, *, action: str) -> OCRDraft:
    data = _aliyun_data_payload(payload)
    if not data:
        return OCRDraft(
            provider="aliyun_edu_ocr",
            model=action,
            source=_aliyun_source_for_action(action),
        )
    if action == "RecognizeEduOralCalculation":
        return _parse_aliyun_oral_calculation(data, action=action)
    if action == "RecognizeEduPaperStructed":
        return _parse_aliyun_paper_structed(data, action=action)
    return _parse_aliyun_paper_or_question(data, action=action)


def _parse_aliyun_oral_calculation(data: dict[str, Any], *, action: str) -> OCRDraft:
    image_width = _safe_int(data.get("width") or data.get("orgWidth"), default=1000)
    image_height = _safe_int(data.get("height") or data.get("orgHeight"), default=1000)
    raw_items = data.get("mathsInfo") or data.get("mathInfos") or data.get("math_info") or []
    items: list[OCRItemDraft] = []
    raw_parts: list[str] = []
    for raw_item in raw_items if isinstance(raw_items, list) else []:
        if not isinstance(raw_item, dict):
            continue
        title = str(raw_item.get("title") or raw_item.get("content") or "").strip()
        if not title:
            continue
        raw_parts.append(title)
        question, answer = _split_oral_calculation_title(title)
        result = str(raw_item.get("result") or "").strip().lower()
        ocr_judgement = _normalize_aliyun_oral_result(result)
        correct_answer = _oral_correct_answer(raw_item, fallback=answer if ocr_judgement == "correct" else "")
        evidence_points = [_oral_judgement_evidence(ocr_judgement)] if ocr_judgement else []
        confidence = 0.95 if result in {"right", "wrong"} else 0.65
        items.append(
            OCRItemDraft(
                item_index=len(items) + 1,
                question_text=question,
                child_answer=answer,
                work_steps=_first_string(raw_item, "process", "formula", "solution", "analysis"),
                confidence=confidence,
                bbox=_bbox_from_aliyun_position(
                    raw_item.get("pos") or raw_item.get("points"),
                    image_width=image_width,
                    image_height=image_height,
                ),
                source_action=action,
                ocr_judgement=ocr_judgement,
                marking_source="aliyun_edu_oral_calculation" if ocr_judgement else "",
                correct_answer=correct_answer,
                evidence_points=evidence_points,
            )
        )
    items = _with_default_bboxes(items)
    confidence = _overall_confidence(items, default=0.0)
    raw_text = "\n".join(raw_parts).strip() or _raw_text_from_items(items)
    return OCRDraft(
        raw_text=raw_text,
        question_text=items[0].question_text if items else "",
        child_answer=items[0].child_answer if items else "",
        confidence=confidence,
        needs_confirmation=(
            not items
            or any(not item.question_text or not item.child_answer for item in items)
            or confidence < 0.75
        ),
        items=items,
        provider="aliyun_edu_ocr",
        model=action,
        source="aliyun_edu_oral_calculation",
    )


def _normalize_aliyun_oral_result(result: str) -> str:
    normalized = (result or "").strip().lower()
    if normalized in {"right", "correct", "true", "1", "yes"}:
        return "correct"
    if normalized in {"wrong", "incorrect", "false", "0", "no"}:
        return "wrong"
    return ""


def _oral_correct_answer(raw_item: dict[str, Any], *, fallback: str = "") -> str:
    return (
        _first_string(
            raw_item,
            "correct_answer",
            "correctAnswer",
            "right_answer",
            "rightAnswer",
            "standard_answer",
            "standardAnswer",
            "answer",
        )
        or fallback
    )


def _oral_judgement_evidence(judgement: str) -> str:
    if judgement == "correct":
        return "教育OCR口算判题：正确"
    if judgement == "wrong":
        return "教育OCR口算判题：错误"
    return "教育OCR口算判题：待确认"


def _parse_aliyun_paper_or_question(data: dict[str, Any], *, action: str) -> OCRDraft:
    explicit_raw_text = str(data.get("content") or data.get("Content") or "").strip()
    raw_text = explicit_raw_text
    raw_text_from_page_list = False
    word_infos = _aliyun_words_info(data)
    if not raw_text and word_infos:
        raw_text = "\n".join(
            str(word.get("word") or word.get("text") or "").strip()
            for word in word_infos
            if isinstance(word, dict) and str(word.get("word") or word.get("text") or "").strip()
        )
    if not raw_text:
        raw_text = _aliyun_page_list_text(data)
        raw_text_from_page_list = bool(raw_text)
    raw_text = _normalize_inline_answer_markers(raw_text)
    items = _items_from_aliyun_question_lists(data, action=action)
    should_merge_raw_text = bool(explicit_raw_text) or not items or (raw_text_from_page_list and len(items) == 1)
    raw_text_items = (
        _items_from_text(raw_text, assign_default_bboxes=False)
        if raw_text and should_merge_raw_text
        else []
    )
    if items and raw_text_items:
        items = _merge_payload_items_with_raw_text_items(items, raw_text_items)
    elif raw_text_items:
        items = raw_text_items
    items = _assign_text_item_bboxes_from_word_infos(items, word_infos, data)
    if items and len(items) == 1 and items[0].bbox is None:
        bbox = _first_word_bbox(word_infos, data)
        if bbox is not None:
            items = [items[0].model_copy(update={"bbox": bbox})]
    if action == "RecognizeEduPaperCut":
        items = _sort_ocr_items_by_bbox_layout(items)
    items = _with_default_bboxes(items)
    confidence = _aliyun_words_confidence(word_infos)
    if confidence <= 0:
        confidence = _overall_confidence(items, default=0.82 if raw_text else 0.0)
    if not raw_text and items:
        raw_text = _raw_text_from_items(items)
    return OCRDraft(
        raw_text=raw_text,
        question_text=items[0].question_text if items else "",
        child_answer=items[0].child_answer if items else "",
        work_steps=items[0].work_steps if items else "",
        confidence=confidence,
        needs_confirmation=(
            not items
            or any(not item.question_text or not item.child_answer for item in items)
            or confidence < 0.75
        ),
        items=items,
        provider="aliyun_edu_ocr",
        model=action,
        source=_aliyun_source_for_action(action),
    )


def _parse_aliyun_paper_structed(data: dict[str, Any], *, action: str) -> OCRDraft:
    items: list[OCRItemDraft] = []
    for part in data.get("part_info") or data.get("partInfo") or []:
        if not isinstance(part, dict):
            continue
        for subject in part.get("subject_list") or part.get("subjectList") or []:
            if not isinstance(subject, dict):
                continue
            question = _first_string(subject, "text", "content", "question", "stem")
            if not question:
                continue
            item_confidence = _safe_aliyun_probability(
                subject.get("prob") or subject.get("confidence")
            )
            if item_confidence <= 0:
                item_confidence = 0.82
            items.append(
                OCRItemDraft(
                    item_index=len(items) + 1,
                    question_text=question,
                    child_answer=_first_answer_from_structed_subject(subject),
                    confidence=item_confidence,
                    bbox=_bbox_from_structed_subject(subject, data=data),
                    source_action=action,
                )
            )
    items = _sort_ocr_items_by_number_and_bbox(_with_default_bboxes(items))
    raw_text = _raw_text_from_items(items)
    confidence = _overall_confidence(items, default=0.82 if items else 0.0)
    return OCRDraft(
        raw_text=raw_text,
        question_text=items[0].question_text if items else "",
        child_answer=items[0].child_answer if items else "",
        confidence=confidence,
        needs_confirmation=(
            not items
            or any(not item.question_text or not item.child_answer for item in items)
            or confidence < 0.75
        ),
        items=items,
        provider="aliyun_edu_ocr",
        model=action,
        source=_aliyun_source_for_action(action),
    )


def _assign_text_item_bboxes_from_word_infos(
    items: list[OCRItemDraft],
    word_infos: list[dict[str, Any]],
    data: dict[str, Any],
) -> list[OCRItemDraft]:
    if not items or not word_infos:
        return items
    records = _ocr_word_bbox_records(word_infos, data)
    if not records:
        return items

    starts: list[int | None] = []
    cursor = 0
    for item in items:
        start = _find_word_record_for_item(records, item, start_at=cursor)
        starts.append(start)
        if start is not None:
            cursor = start + 1

    ordered_starts = [start for start in starts if start is not None]
    starts_are_monotonic = ordered_starts == sorted(ordered_starts)
    mapped: list[OCRItemDraft] = []
    for index, item in enumerate(items):
        if item.bbox is not None:
            mapped.append(item)
            continue
        start = starts[index]
        if start is None:
            mapped.append(item)
            continue
        if not starts_are_monotonic:
            mapped.append(item.model_copy(update={"bbox": records[start]["bbox"]}))
            continue
        next_start = next(
            (value for value in starts[index + 1 :] if value is not None and value > start),
            None,
        )
        end = next_start if next_start is not None else len(records)
        bbox = _union_bboxes([record["bbox"] for record in records[start:end]])
        mapped.append(item.model_copy(update={"bbox": bbox or item.bbox}))
    return mapped


def _ocr_word_bbox_records(
    word_infos: list[dict[str, Any]],
    data: dict[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for word in word_infos:
        if not isinstance(word, dict):
            continue
        text = str(word.get("word") or word.get("text") or "").strip()
        if not text:
            continue
        bbox = _bbox_from_aliyun_payload(word, data=data)
        if bbox is None:
            continue
        records.append({"text": text, "compact": _compact_ocr_text_for_bbox_match(text), "bbox": bbox})
    return records


def _find_word_record_for_item(
    records: list[dict[str, Any]],
    item: OCRItemDraft,
    *,
    start_at: int,
) -> int | None:
    compact_question = _compact_ocr_text_for_bbox_match(item.question_text)
    if not compact_question:
        return None
    probe = compact_question[: max(6, min(18, len(compact_question)))]
    for index in range(start_at, len(records)):
        if _word_record_matches_item(records[index]["compact"], compact_question, probe):
            return index
    for index in range(0, start_at):
        if _word_record_matches_item(records[index]["compact"], compact_question, probe):
            return index
    return None


def _word_record_matches_item(record_text: str, item_text: str, probe: str) -> bool:
    if not record_text or not item_text:
        return False
    similarity = SequenceMatcher(None, record_text, item_text).ratio()
    return (
        item_text in record_text
        or record_text in item_text
        or probe in record_text
        or record_text[: max(6, min(14, len(record_text)))] in item_text
        or (len(item_text) >= 6 and similarity >= 0.62)
    )


def _compact_ocr_text_for_bbox_match(value: str) -> str:
    text = (value or "").replace("＝", "=").replace("×", "x").replace("÷", "/")
    return re.sub(r"[\s，。,.!?！？；;：:、\"'“”‘’（）()\[\]【】]+", "", text).lower()


def _union_bboxes(bboxes: list[ImageBBox]) -> ImageBBox | None:
    values = [bbox for bbox in bboxes if bbox is not None]
    if not values:
        return None
    left = min(bbox.x for bbox in values)
    top = min(bbox.y for bbox in values)
    right = max(bbox.x + bbox.width for bbox in values)
    bottom = max(bbox.y + bbox.height for bbox in values)
    return ImageBBox(
        x=left,
        y=top,
        width=max(1, min(1000 - left, right - left)),
        height=max(1, min(1000 - top, bottom - top)),
    )


def _first_answer_from_structed_subject(subject: dict[str, Any]) -> str:
    answers = subject.get("answer_list") or subject.get("answerList") or []
    if isinstance(answers, list):
        values = [
            _first_string(answer, "text", "content", "value", "answer")
            for answer in answers
            if isinstance(answer, dict)
        ]
        return "；".join(value for value in values if value)
    return _first_string(subject, "answer", "child_answer", "student_answer")


def _bbox_from_structed_subject(subject: dict[str, Any], *, data: dict[str, Any]) -> ImageBBox | None:
    bbox = _bbox_from_aliyun_payload(subject, data=data)
    if bbox is not None:
        return bbox
    elements = subject.get("element_list") or subject.get("elementList") or []
    if isinstance(elements, list):
        for element in elements:
            if isinstance(element, dict):
                bbox = _bbox_from_aliyun_payload(element, data=data)
                if bbox is not None:
                    return bbox
    return None


def _aliyun_data_payload(payload: object) -> dict[str, Any]:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return {}
    if not isinstance(payload, dict):
        body = getattr(payload, "body", None)
        if body is not None:
            return _aliyun_data_payload(body)
        return {}
    body = payload.get("body")
    if body is not None and not any(key in payload for key in ("Data", "data", "content", "mathsInfo")):
        return _aliyun_data_payload(body)
    data = payload.get("Data")
    if data is None:
        data = payload.get("data")
    if data is None:
        data = payload
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return {"content": data}
    return data if isinstance(data, dict) else {}


def _normalize_inline_answer_markers(raw_text: str) -> str:
    text = raw_text.strip()
    if not text:
        return ""
    text = re.sub(
        r"\s+((?:孩子|学生|child|student)?\s*(?:答案|answer))\s*[:：]",
        r"\n\1：",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(
        r"(?<=[^\s（(=＝])\s+((?:\d+\s*[\.\uff0e、)]|[（(]\s*\d+\s*[）)])\s*)",
        r"\n\n\1",
        text,
    )


def _aliyun_edu_ocr_action(scene: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", (scene or "").strip().lower()).strip("_")
    return {
        "oral": "RecognizeEduOralCalculation",
        "oral_calculation": "RecognizeEduOralCalculation",
        "kousuan": "RecognizeEduOralCalculation",
        "paper": "RecognizeEduPaperOcr",
        "paper_ocr": "RecognizeEduPaperOcr",
        "page": "RecognizeEduPaperOcr",
        "paper_cut": "RecognizeEduPaperCut",
        "cut": "RecognizeEduPaperCut",
        "question": "RecognizeEduQuestionOcr",
        "question_ocr": "RecognizeEduQuestionOcr",
        "paper_structed": "RecognizeEduPaperStructed",
        "paper_structured": "RecognizeEduPaperStructed",
        "formula": "RecognizeEduFormula",
        "auto": "RecognizeEduPaperCut",
        "": "RecognizeEduPaperCut",
    }.get(normalized, "RecognizeEduPaperCut")


def _aliyun_edu_attempt_actions(primary_action: str) -> list[str]:
    fallback_map = {
        "RecognizeEduPaperCut": [
            "RecognizeEduPaperCut",
            "RecognizeEduPaperOcr",
            "RecognizeEduQuestionOcr",
        ],
        "RecognizeEduPaperOcr": [
            "RecognizeEduPaperOcr",
            "RecognizeEduQuestionOcr",
        ],
        "RecognizeEduQuestionOcr": [
            "RecognizeEduQuestionOcr",
            "RecognizeEduPaperOcr",
        ],
    }
    return fallback_map.get(primary_action, [primary_action])


def _draft_with_ocr_plan(draft: OCRDraft, plan: dict[str, Any]) -> OCRDraft:
    return draft.model_copy(update={"data_json": {**draft.data_json, "ocr_plan": plan}})


def _ocr_plan_for_action(
    action: str,
    *,
    reason: str,
    subject_hint: str,
) -> EducationOcrPlan:
    ocr_action = EducationOcrAction(action)
    expected_output = "full_page_text" if ocr_action == EducationOcrAction.PAPER_OCR else "question_boxes"
    return EducationOcrPlan(
        primary_action=ocr_action,
        reason=reason,
        expected_output=expected_output,
        subject_hint=subject_hint,
    )


def _aliyun_source_for_action(action: str) -> str:
    return {
        "RecognizeEduOralCalculation": "aliyun_edu_oral_calculation",
        "RecognizeEduPaperOcr": "aliyun_edu_paper_ocr",
        "RecognizeEduPaperCut": "aliyun_edu_paper_cut",
        "RecognizeEduQuestionOcr": "aliyun_edu_question_ocr",
        "RecognizeEduPaperStructed": "aliyun_edu_paper_structed",
        "RecognizeEduFormula": "aliyun_edu_formula",
    }.get(action, "aliyun_edu")


def _should_run_paper_cut_text_fallback(
    *,
    draft: OCRDraft,
    config: Any,
    action: str,
) -> bool:
    if action != "RecognizeEduPaperCut":
        return False
    if not bool(getattr(config, "hybrid_text_fallback", False)):
        return False
    if not draft.items:
        return False
    min_answer_rate = float(getattr(config, "hybrid_text_fallback_min_answer_rate", 0.6) or 0.0)
    min_answer_rate = max(0.0, min(1.0, min_answer_rate))
    answer_rate = sum(1 for item in draft.items if item.child_answer.strip()) / len(draft.items)
    return answer_rate < min_answer_rate


def _draft_looks_like_oral_calculation_page(draft: OCRDraft) -> bool:
    items = list(draft.items or [])
    if len(items) < 3:
        return False
    direct_count = sum(1 for item in items if _looks_like_oral_calculation_text(item.question_text))
    if direct_count >= 3 and direct_count / len(items) >= 0.55:
        return True
    raw_text = draft.raw_text or _raw_text_from_items(items)
    return direct_count >= 2 and any(token in raw_text for token in ("口算", "直接写得数", "直接写得数。"))


def _looks_like_oral_calculation_text(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return False
    compact = re.sub(r"\s+", "", value)
    if len(compact) > 36:
        return False
    return bool(
        re.search(
            r"\d+(?:\.\d+)?\s*[+\-＋－×xX*÷/]\s*\d+(?:\.\d+)?\s*[=＝]?\s*(?:\d+(?:\.\d+)?)?",
            compact,
        )
    )


def _aliyun_edu_error_draft(*, action: str, error: Exception) -> OCRDraft:
    warnings = ["ocr_provider_error"]
    code = _aliyun_edu_error_warning_code(error)
    if code:
        warnings.append(code)
    return OCRDraft(
        provider="aliyun_edu_ocr",
        model=action,
        source="aliyun_edu_error",
        quality_warnings=warnings,
        quality_message=_aliyun_edu_error_message(code),
    )


def _aliyun_edu_error_warning_code(error: Exception) -> str:
    text = str(error)
    if "OcrServiceExpired" in text:
        return "ocr_service_expired"
    if "InvalidAccessKeyId" in text or "InvalidAccessKey" in text:
        return "ocr_invalid_access_key"
    if "Forbidden" in text or "Unauthorized" in text or "code: 401" in text:
        return "ocr_auth_failed"
    if "Throttl" in text or "QPS" in text or "TooManyRequests" in text:
        return "ocr_rate_limited"
    return ""


def _aliyun_edu_error_message(code: str) -> str:
    if code == "ocr_service_expired":
        return "教育 OCR 服务当前不可用，请检查阿里云 OCR 套餐是否开通或到期。"
    if code == "ocr_rate_limited":
        return "教育 OCR 当前请求过快，请稍后再试。"
    if code in {"ocr_invalid_access_key", "ocr_auth_failed"}:
        return "教育 OCR 服务当前不可用，请检查阿里云 AccessKey 权限和套餐状态。"
    return "教育 OCR 服务当前不可用，请稍后再试。"


def _split_oral_calculation_title(title: str) -> tuple[str, str]:
    value = title.strip()
    if "=" not in value:
        return value, ""
    left, right = value.rsplit("=", 1)
    question = f"{left.strip()} =".strip()
    return question, right.strip()


def _aliyun_words_info(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw_words = data.get("prism_wordsInfo") or data.get("wordsInfo") or data.get("words") or []
    words = [word for word in raw_words if isinstance(word, dict)] if isinstance(raw_words, list) else []
    for subject in _aliyun_page_subject_items(data):
        subject_words = subject.get("prism_wordsInfo") or []
        if isinstance(subject_words, list):
            words.extend(word for word in subject_words if isinstance(word, dict))
    return words


def _items_from_aliyun_question_lists(data: dict[str, Any], *, action: str = "") -> list[OCRItemDraft]:
    items: list[OCRItemDraft] = []
    for raw_item in _iter_aliyun_question_items(data):
        question = _first_string(
            raw_item,
            "question_text",
            "questionText",
            "question",
            "stem",
            "title",
            "content",
            "text",
        )
        if not question:
            continue
        handwritten_tokens = _handwritten_aliyun_word_tokens(raw_item, data=data)
        use_coordinate_answer = _should_use_coordinate_child_answer(handwritten_tokens)
        question = _question_with_coordinate_bound_answers(question, raw_item, data=data)
        if use_coordinate_answer:
            question = _question_without_coordinate_answer_artifacts(question, handwritten_tokens)
        child_answer = _first_string(
            raw_item,
            "child_answer",
            "childAnswer",
            "student_answer",
            "studentAnswer",
            "userAnswer",
            "answer",
        )
        if not child_answer and use_coordinate_answer:
            child_answer = _handwritten_answer_from_tokens(handwritten_tokens)
        items.append(
            OCRItemDraft(
                item_index=len(items) + 1,
                question_text=question,
                child_answer=child_answer,
                work_steps=_first_string(raw_item, "work_steps", "workSteps", "solution", "analysis"),
                confidence=_safe_aliyun_probability(raw_item.get("prob") or raw_item.get("confidence")),
                bbox=_bbox_from_aliyun_payload(raw_item, data=data),
                source_action=action,
            )
        )
    return _sort_ocr_items_by_number_and_bbox(items, assign_default_bboxes=False)


def _iter_aliyun_question_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for key in ("questionList", "questions", "items", "questionInfo", "questionsInfo"):
        value = data.get(key)
        if isinstance(value, list):
            collected.extend(item for item in value if isinstance(item, dict))
        elif isinstance(value, dict):
            collected.append(value)
    collected.extend(_aliyun_page_subject_items(data))
    return collected


def _aliyun_page_subject_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    page_list = data.get("page_list") or data.get("pageList") or []
    if not isinstance(page_list, list):
        return []
    subjects: list[dict[str, Any]] = []
    for page in page_list:
        if not isinstance(page, dict):
            continue
        subject_list = page.get("subject_list") or page.get("subjectList") or []
        if not isinstance(subject_list, list):
            continue
        page_width = _safe_int(page.get("width") or page.get("orgWidth"), default=1000)
        page_height = _safe_int(page.get("height") or page.get("orgHeight"), default=1000)
        for raw_subject in subject_list:
            if not isinstance(raw_subject, dict):
                continue
            subject = dict(raw_subject)
            subject["_image_width"] = page_width
            subject["_image_height"] = page_height
            subjects.append(subject)
    return subjects


def _aliyun_page_list_text(data: dict[str, Any]) -> str:
    return "\n\n".join(
        str(subject.get("text") or subject.get("content") or "").strip()
        for subject in _aliyun_page_subject_items(data)
        if str(subject.get("text") or subject.get("content") or "").strip()
    )


def _handwritten_answer_from_tokens(tokens: list[dict[str, Any]]) -> str:
    values = [_normalize_handwritten_answer_token(str(token.get("text") or "")) for token in tokens]
    return "；".join(value for value in values if value)


def _should_use_coordinate_child_answer(tokens: list[dict[str, Any]]) -> bool:
    if not tokens:
        return False
    if len(tokens) == 1:
        return True
    return all(_is_short_objective_answer_token(str(token.get("text") or "")) for token in tokens)


def _question_with_coordinate_bound_answers(
    question: str,
    raw_item: dict[str, Any],
    *,
    data: dict[str, Any],
) -> str:
    answer_tokens = [
        token
        for token in _handwritten_aliyun_word_tokens(raw_item, data=data)
        if _is_short_objective_answer_token(token["text"])
    ]
    if not answer_tokens:
        return question
    gaps = _ocr_gap_markers_from_words(question, raw_item, data=data)
    if not gaps:
        return question
    assignments: dict[int, str] = {}
    used_gaps: set[int] = set()
    for token in answer_tokens:
        if _answer_token_already_present_nearby(token, raw_item, data=data):
            continue
        best_index = _nearest_gap_index(token, gaps, used_gaps=used_gaps)
        if best_index is None:
            continue
        normalized = _normalize_handwritten_answer_token(token["text"])
        if not normalized:
            continue
        assignments[best_index] = normalized
        used_gaps.add(best_index)
    if not assignments:
        return question
    updated = question
    for gap_index, answer in sorted(assignments.items(), key=lambda item: gaps[item[0]]["start"], reverse=True):
        gap = gaps[gap_index]
        replacement = _filled_gap_text(str(gap["text"]), answer)
        updated = f"{updated[:gap['start']]}{replacement}{updated[gap['end']:]}"
    return updated


def _question_without_coordinate_answer_artifacts(
    question: str,
    tokens: list[dict[str, Any]],
) -> str:
    cleaned = question
    for token in tokens:
        answer = _normalize_handwritten_answer_token(str(token.get("text") or ""))
        if not answer or _is_comparison_answer_token(answer):
            continue
        cleaned = _remove_bracketed_coordinate_answer(cleaned, answer)
    return cleaned


def _remove_bracketed_coordinate_answer(question: str, answer: str) -> str:
    escaped = re.escape(answer)
    cleaned = re.sub(
        rf"[（(]\s*[^（）()]{{0,12}}{escaped}[^（）()]{{0,12}}\s*[）)]",
        "( )",
        question,
        count=1,
    )
    if cleaned != question:
        return cleaned
    cleaned = re.sub(
        rf"\s*[-－]\s*{escaped}\s*[）)](?=\s*[\u4e00-\u9fffA-Za-z])",
        "( )",
        question,
        count=1,
    )
    if cleaned != question:
        return cleaned
    return re.sub(
        rf"[（(]\s*[（(]\s*{escaped}(?=\s*(?:个|字|位|元|米|厘米|分|秒|时|千克|克|本|张|条|道|题|倍))",
        "( )",
        question,
        count=1,
    )


def _answer_token_already_present_nearby(
    token: dict[str, Any],
    raw_item: dict[str, Any],
    *,
    data: dict[str, Any],
) -> bool:
    answer = _normalize_handwritten_answer_token(str(token.get("text") or ""))
    if not answer:
        return False
    token_bbox = token.get("bbox")
    if not isinstance(token_bbox, ImageBBox):
        return False
    words = raw_item.get("prism_wordsInfo") or raw_item.get("wordsInfo") or []
    if not isinstance(words, list):
        return False
    token_center_x = token_bbox.x + token_bbox.width / 2
    token_center_y = token_bbox.y + token_bbox.height / 2
    for word in words:
        if not isinstance(word, dict) or _safe_int(word.get("recClassify"), default=0) == 2:
            continue
        if answer not in str(word.get("word") or ""):
            continue
        bbox = _bbox_from_aliyun_word(word, raw_item=raw_item, data=data)
        if bbox is None:
            continue
        word_center_x = bbox.x + bbox.width / 2
        word_center_y = bbox.y + bbox.height / 2
        if abs(token_center_y - word_center_y) <= 45:
            if abs(token_center_x - word_center_x) <= max(80, bbox.width * 0.75):
                return True
    return False


def _handwritten_aliyun_word_tokens(
    raw_item: dict[str, Any],
    *,
    data: dict[str, Any],
) -> list[dict[str, Any]]:
    words = raw_item.get("prism_wordsInfo") or raw_item.get("wordsInfo") or []
    if not isinstance(words, list):
        return []
    tokens: list[dict[str, Any]] = []
    for word in words:
        if not isinstance(word, dict):
            continue
        if _safe_int(word.get("recClassify"), default=0) != 2:
            continue
        text = _normalize_handwritten_answer_token(str(word.get("word") or ""))
        if not text:
            continue
        bbox = _bbox_from_aliyun_word(word, raw_item=raw_item, data=data)
        if bbox is None:
            continue
        tokens.append({"text": text, "bbox": bbox})
    return tokens


def _ocr_gap_markers_from_words(
    question: str,
    raw_item: dict[str, Any],
    *,
    data: dict[str, Any],
) -> list[dict[str, Any]]:
    words = raw_item.get("prism_wordsInfo") or raw_item.get("wordsInfo") or []
    if not isinstance(words, list):
        return []
    gaps: list[dict[str, Any]] = []
    cursor = 0
    marker_re = re.compile(r"[（(]\s*[.．]?\s*[）)]|○")
    for word in words:
        if not isinstance(word, dict) or _safe_int(word.get("recClassify"), default=0) == 2:
            continue
        text = str(word.get("word") or "")
        if not text:
            continue
        start = question.find(text, cursor)
        if start < 0:
            start = question.find(text)
        if start < 0:
            continue
        cursor = start + len(text)
        bbox = _bbox_from_aliyun_word(word, raw_item=raw_item, data=data)
        if bbox is None:
            continue
        for match in marker_re.finditer(text):
            marker = match.group(0)
            if any(sign in marker for sign in "<>＝=≤≥√✓Vv×xX对错"):
                continue
            center = _inline_marker_center(bbox, text=text, start=match.start(), end=match.end())
            gaps.append(
                {
                    "start": start + match.start(),
                    "end": start + match.end(),
                    "text": marker,
                    "center_x": center[0],
                    "center_y": center[1],
                    "height": bbox.height,
                }
            )
    return gaps


def _nearest_gap_index(
    token: dict[str, Any],
    gaps: list[dict[str, Any]],
    *,
    used_gaps: set[int],
) -> int | None:
    bbox = token.get("bbox")
    if not isinstance(bbox, ImageBBox):
        return None
    token_center_x = bbox.x + bbox.width / 2
    token_center_y = bbox.y + bbox.height / 2
    best_index: int | None = None
    best_score = float("inf")
    for index, gap in enumerate(gaps):
        if index in used_gaps:
            continue
        row_threshold = max(45.0, float(gap["height"]) * 1.8, bbox.height * 1.8)
        dy = abs(token_center_y - float(gap["center_y"]))
        if dy > row_threshold:
            continue
        dx = abs(token_center_x - float(gap["center_x"]))
        score = dx + dy * 2
        if score < best_score:
            best_index = index
            best_score = score
    return best_index


def _inline_marker_center(bbox: ImageBBox, *, text: str, start: int, end: int) -> tuple[float, float]:
    length = max(1, len(text))
    marker_middle = (start + end) / 2
    return (
        bbox.x + bbox.width * (marker_middle / length),
        bbox.y + bbox.height / 2,
    )


def _filled_gap_text(marker: str, answer: str) -> str:
    if marker.startswith("（"):
        return f"（{answer}）"
    if marker == "○":
        return f"({answer})"
    return f"({answer})"


def _bbox_from_aliyun_word(
    word: dict[str, Any],
    *,
    raw_item: dict[str, Any],
    data: dict[str, Any],
) -> ImageBBox | None:
    image_width = _safe_int(
        raw_item.get("_image_width") or data.get("width") or data.get("orgWidth"),
        default=1000,
    )
    image_height = _safe_int(
        raw_item.get("_image_height") or data.get("height") or data.get("orgHeight"),
        default=1000,
    )
    return _bbox_from_aliyun_position(
        word.get("pos") or word.get("points"),
        image_width=image_width,
        image_height=image_height,
    )


def _normalize_handwritten_answer_token(value: str) -> str:
    text = re.sub(r"\s+", "", value or "").strip()
    text = text.strip("()（）")
    if text in {"", ".", "．", "。", "，", ","}:
        return ""
    if text == "✓":
        return "√"
    return text


def _is_short_objective_answer_token(value: str) -> bool:
    text = _normalize_handwritten_answer_token(value)
    return bool(re.fullmatch(r"(?:[<>＝=≤≥]|[A-Da-d]|[√✓Vv×xX对错])", text))


def _is_comparison_answer_token(value: str) -> bool:
    return bool(re.fullmatch(r"[<>＝=≤≥]", _normalize_handwritten_answer_token(value)))


def _first_string(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, dict):
            nested = _first_string(value, "text", "content", "value")
            if nested:
                return nested
            continue
        if isinstance(value, list):
            joined = "\n".join(str(item).strip() for item in value if str(item).strip())
            if joined:
                return joined
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _bbox_from_aliyun_payload(raw_item: dict[str, Any], *, data: dict[str, Any]) -> ImageBBox | None:
    image_width = _safe_int(
        raw_item.get("_image_width") or data.get("width") or data.get("orgWidth"),
        default=1000,
    )
    image_height = _safe_int(
        raw_item.get("_image_height") or data.get("height") or data.get("orgHeight"),
        default=1000,
    )
    content_list = raw_item.get("content_list_info") or raw_item.get("contentListInfo") or []
    if isinstance(content_list, list) and content_list:
        first_content = content_list[0] if isinstance(content_list[0], dict) else {}
    else:
        first_content = {}
    bbox = _bbox_from_aliyun_position(
        raw_item.get("pos") or raw_item.get("points") or first_content.get("pos"),
        image_width=image_width,
        image_height=image_height,
    )
    if bbox is not None:
        return bbox
    box = raw_item.get("box") or raw_item.get("bbox") or raw_item.get("boundingBox")
    if isinstance(box, dict):
        return _bbox_from_payload(
            {
                "x": box.get("x"),
                "y": box.get("y"),
                "width": box.get("w") or box.get("width"),
                "height": box.get("h") or box.get("height"),
            }
        )
    return None


def _first_word_bbox(word_infos: list[dict[str, Any]], data: dict[str, Any]) -> ImageBBox | None:
    if not word_infos:
        return None
    return _bbox_from_aliyun_payload(word_infos[0], data=data)


def _bbox_from_aliyun_position(
    value: object,
    *,
    image_width: int,
    image_height: int,
) -> ImageBBox | None:
    if not isinstance(value, list) or not value:
        return None
    points = []
    for point in value:
        if not isinstance(point, dict):
            continue
        x = _safe_int(point.get("x"), default=-1)
        y = _safe_int(point.get("y"), default=-1)
        if x >= 0 and y >= 0:
            points.append((x, y))
    if not points or image_width <= 0 or image_height <= 0:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    left = min(xs)
    top = min(ys)
    width = max(xs) - left
    height = max(ys) - top
    if width <= 0 or height <= 0:
        return None
    x = max(0, min(1000, round((left / image_width) * 1000)))
    y = max(0, min(1000, round((top / image_height) * 1000)))
    normalized_width = max(1, min(1000 - x, round((width / image_width) * 1000)))
    normalized_height = max(1, min(1000 - y, round((height / image_height) * 1000)))
    return ImageBBox(x=x, y=y, width=normalized_width, height=normalized_height)


def _aliyun_words_confidence(word_infos: list[dict[str, Any]]) -> float:
    scores = [
        _safe_aliyun_probability(word.get("prob") or word.get("confidence"))
        for word in word_infos
        if isinstance(word, dict)
    ]
    scores = [score for score in scores if score > 0]
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 4)


def _safe_aliyun_probability(value: object) -> float:
    try:
        score = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if score > 1:
        score = score / 100.0
    return max(0.0, min(1.0, score))


def _call_aliyun_edu_ocr_sync(
    *,
    config: Any,
    action: str,
    content: bytes,
    filename: str,
) -> dict[str, Any]:
    if not getattr(config, "access_key_id", "") or not getattr(config, "access_key_secret", ""):
        raise RuntimeError("Aliyun Education OCR requires AccessKey ID and AccessKey Secret")
    try:
        from io import BytesIO

        from alibabacloud_ocr_api20210707.client import Client as AliyunOCRClient
        from alibabacloud_ocr_api20210707 import models as ocr_models
        from alibabacloud_tea_openapi import models as open_api_models
        from alibabacloud_tea_util import models as util_models
    except ImportError as exc:
        raise RuntimeError(
            "Aliyun Education OCR SDK is not installed; install requirements.txt first"
        ) from exc

    timeout_ms = max(100, int(float(getattr(config, "timeout_seconds", 12.0)) * 1000))
    client = AliyunOCRClient(
        open_api_models.Config(
            access_key_id=config.access_key_id,
            access_key_secret=config.access_key_secret,
            endpoint=_aliyun_edu_sdk_endpoint(config.endpoint),
            region_id=getattr(config, "region", "") or "cn-hangzhou",
            connect_timeout=timeout_ms,
            read_timeout=timeout_ms,
        )
    )
    request = _build_aliyun_edu_sdk_request(
        ocr_models,
        action=action,
        config=config,
        body=BytesIO(content),
    )
    runtime = util_models.RuntimeOptions(
        autoretry=True,
        max_attempts=max(1, int(getattr(config, "retry_attempts", 2))),
        read_timeout=timeout_ms,
        connect_timeout=timeout_ms,
    )
    method = getattr(client, _aliyun_edu_sdk_method(action))
    return _sdk_response_to_payload(method(request, runtime))


def _build_aliyun_edu_sdk_request(
    ocr_models: Any,
    *,
    action: str,
    config: Any,
    body: Any,
) -> object:
    if action == "RecognizeEduPaperCut":
        return ocr_models.RecognizeEduPaperCutRequest(
            cut_type=getattr(config, "cut_type", "question") or "question",
            image_type=getattr(config, "image_type", "photo") or "photo",
            output_oricoord=bool(getattr(config, "output_oricoord", True)),
            subject=getattr(config, "subject", "default") or "default",
            body=body,
        )
    if action == "RecognizeEduPaperOcr":
        return ocr_models.RecognizeEduPaperOcrRequest(
            image_type=getattr(config, "image_type", "photo") or "photo",
            output_oricoord=bool(getattr(config, "output_oricoord", True)),
            subject=getattr(config, "subject", "default") or "default",
            body=body,
        )
    if action == "RecognizeEduQuestionOcr":
        return ocr_models.RecognizeEduQuestionOcrRequest(
            need_rotate=bool(getattr(config, "need_rotate", True)),
            body=body,
        )
    if action == "RecognizeEduOralCalculation":
        return ocr_models.RecognizeEduOralCalculationRequest(body=body)
    if action == "RecognizeEduFormula":
        return ocr_models.RecognizeEduFormulaRequest(body=body)
    if action == "RecognizeEduPaperStructed":
        return ocr_models.RecognizeEduPaperStructedRequest(
            need_rotate=bool(getattr(config, "need_rotate", True)),
            output_oricoord=bool(getattr(config, "output_oricoord", True)),
            subject=getattr(config, "subject", "default") or "default",
            body=body,
        )
    raise RuntimeError(f"Unsupported Aliyun Education OCR action: {action}")


def _aliyun_edu_sdk_method(action: str) -> str:
    return {
        "RecognizeEduPaperCut": "recognize_edu_paper_cut_with_options",
        "RecognizeEduPaperOcr": "recognize_edu_paper_ocr_with_options",
        "RecognizeEduQuestionOcr": "recognize_edu_question_ocr_with_options",
        "RecognizeEduOralCalculation": "recognize_edu_oral_calculation_with_options",
        "RecognizeEduFormula": "recognize_edu_formula_with_options",
        "RecognizeEduPaperStructed": "recognize_edu_paper_structed_with_options",
    }[action]


def _aliyun_edu_sdk_endpoint(endpoint: str) -> str:
    value = (endpoint or "https://ocr-api.cn-hangzhou.aliyuncs.com").strip()
    value = re.sub(r"^https?://", "", value)
    return value.split("/", 1)[0]


def _sdk_response_to_payload(response: object) -> dict[str, Any]:
    if hasattr(response, "to_map"):
        mapped = response.to_map()
        return mapped if isinstance(mapped, dict) else {}
    body = getattr(response, "body", None)
    if hasattr(body, "to_map"):
        mapped_body = body.to_map()
        return {"body": mapped_body} if isinstance(mapped_body, dict) else {}
    if isinstance(body, dict):
        return {"body": body}
    return {}


def _merge_segment_drafts(
    drafts: list[OCRDraft],
    segments: list[_ImageSegment],
) -> OCRDraft:
    merged_items: list[OCRItemDraft] = []
    seen_questions: list[str] = []
    raw_parts: list[str] = []
    confidences: list[float] = []
    for draft, segment in zip(drafts, segments, strict=False):
        if draft.raw_text.strip():
            raw_parts.append(draft.raw_text.strip())
        if draft.confidence > 0:
            confidences.append(draft.confidence)
        for item in draft.items:
            compact_question = _compact(item.question_text)
            if compact_question and any(
                compact_question == existing
                or compact_question in existing
                or existing in compact_question
                for existing in seen_questions
            ):
                continue
            if compact_question:
                seen_questions.append(compact_question)
            merged_items.append(
                item.model_copy(
                    update={
                        "item_index": len(merged_items) + 1,
                        "bbox": _map_segment_bbox_to_full_image(item.bbox, segment),
                    }
                )
            )
    merged_items = _with_default_bboxes(
        [
            item.model_copy(update={"item_index": index})
            for index, item in enumerate(merged_items, start=1)
        ]
    )
    raw_text = "\n\n".join(raw_parts).strip() or _raw_text_from_items(merged_items)
    confidence = round(sum(confidences) / len(confidences), 4) if confidences else _overall_confidence(merged_items)
    question = merged_items[0].question_text if merged_items else ""
    answer = merged_items[0].child_answer if merged_items else ""
    return OCRDraft(
        raw_text=raw_text,
        question_text=question,
        child_answer=answer,
        confidence=confidence,
        needs_confirmation=(
            not merged_items
            or any(not item.question_text or not item.child_answer for item in merged_items)
            or confidence < 0.8
        ),
        items=merged_items,
    )


def _map_segment_bbox_to_full_image(
    bbox: ImageBBox | None,
    segment: _ImageSegment,
) -> ImageBBox | None:
    if bbox is None:
        return None
    x = segment.x + (bbox.x / 1000.0) * segment.width
    y = segment.y + (bbox.y / 1000.0) * segment.height
    width = (bbox.width / 1000.0) * segment.width
    height = (bbox.height / 1000.0) * segment.height
    mapped_x = max(0, min(999, round((x / segment.full_width) * 1000)))
    mapped_y = max(0, min(999, round((y / segment.full_height) * 1000)))
    mapped_width = max(1, min(1000 - mapped_x, round((width / segment.full_width) * 1000)))
    mapped_height = max(1, min(1000 - mapped_y, round((height / segment.full_height) * 1000)))
    return ImageBBox(x=mapped_x, y=mapped_y, width=mapped_width, height=mapped_height)


def _merge_payload_items_with_raw_text_items(
    payload_items: list[OCRItemDraft],
    raw_text_items: list[OCRItemDraft],
) -> list[OCRItemDraft]:
    if len(payload_items) != len(raw_text_items):
        return _merge_payload_items_by_similarity(payload_items, raw_text_items)
    if not _same_order_payload_raw_match(payload_items, raw_text_items):
        return _merge_payload_items_by_similarity(payload_items, raw_text_items)
    merged: list[OCRItemDraft] = []
    for payload_item, raw_item in zip(payload_items, raw_text_items, strict=True):
        merged.append(_merge_payload_item_with_raw_item(payload_item, raw_item))
    return _renumber_ocr_items(merged)


def _same_order_payload_raw_match(
    payload_items: list[OCRItemDraft],
    raw_text_items: list[OCRItemDraft],
) -> bool:
    if len(payload_items) != len(raw_text_items):
        return False
    if not payload_items:
        return True
    scores = [
        _ocr_question_match_score(payload_item.question_text, raw_item.question_text)
        for payload_item, raw_item in zip(payload_items, raw_text_items, strict=True)
    ]
    return min(scores) >= 0.58


def _merge_payload_items_by_similarity(
    payload_items: list[OCRItemDraft],
    raw_text_items: list[OCRItemDraft],
) -> list[OCRItemDraft]:
    if not raw_text_items:
        return _with_default_bboxes(payload_items)
    merged: list[OCRItemDraft] = []
    used_raw_indexes: set[int] = set()
    for payload_item in payload_items:
        best_index = -1
        best_score = 0.0
        for raw_index, raw_item in enumerate(raw_text_items):
            if raw_index in used_raw_indexes:
                continue
            score = _ocr_question_match_score(payload_item.question_text, raw_item.question_text)
            if score > best_score:
                best_index = raw_index
                best_score = score
        if best_index >= 0 and best_score >= 0.58:
            used_raw_indexes.add(best_index)
            merged.append(_merge_payload_item_with_raw_item(payload_item, raw_text_items[best_index]))
        else:
            merged.append(payload_item)
    return _renumber_ocr_items(merged)


def _merge_payload_item_with_raw_item(
    payload_item: OCRItemDraft,
    raw_item: OCRItemDraft,
) -> OCRItemDraft:
    update: dict[str, object] = {
        "confidence": max(payload_item.confidence, raw_item.confidence),
    }
    if _should_preserve_payload_question_for_answer_binding(payload_item, raw_item):
        pass
    elif _should_prefer_raw_question(payload_item, raw_item):
        update["question_text"] = raw_item.question_text
    if not payload_item.child_answer and raw_item.child_answer:
        update["child_answer"] = raw_item.child_answer
    if not payload_item.work_steps and raw_item.work_steps:
        update["work_steps"] = raw_item.work_steps
    return payload_item.model_copy(update=update)


def _should_preserve_payload_question_for_answer_binding(
    payload_item: OCRItemDraft,
    raw_item: OCRItemDraft,
) -> bool:
    if not raw_item.child_answer:
        return False
    if not re.search(r"[<>＝=≤≥]", payload_item.question_text or ""):
        return False
    bindings = bind_grouped_comparison_answers(
        question_text=payload_item.question_text,
        child_answer=raw_item.child_answer,
    )
    return len(bindings) >= 2 and any(binding.child_answer for binding in bindings)


def _renumber_ocr_items(items: list[OCRItemDraft]) -> list[OCRItemDraft]:
    return _with_default_bboxes([
        item.model_copy(update={"item_index": index})
        for index, item in enumerate(items, start=1)
    ])


def _ocr_question_match_score(left: str, right: str) -> float:
    compact_left = _compact_ocr_question_for_match(left)
    compact_right = _compact_ocr_question_for_match(right)
    if not compact_left or not compact_right:
        return 0.0
    if compact_left in compact_right or compact_right in compact_left:
        return 1.0
    return SequenceMatcher(None, compact_left, compact_right).ratio()


def _compact_ocr_question_for_match(value: str) -> str:
    text = re.sub(r"^\s*\d+[\.\uff0e、)]\s*", "", value or "")
    text = re.sub(r"[（(]\s*[^（）()]{0,16}\s*[）)]", "()", text)
    text = text.replace("＝", "=").replace("×", "x").replace("÷", "/")
    return re.sub(r"[\s，。,.!?！？；;：:、]+", "", text).lower()


def _sort_ocr_items_by_number_and_bbox(
    items: list[OCRItemDraft],
    *,
    assign_default_bboxes: bool = True,
) -> list[OCRItemDraft]:
    if len(items) < 2:
        return _with_default_bboxes(items) if assign_default_bboxes else items
    numbered_count = sum(1 for item in items if _ocr_question_number(item.question_text) is not None)
    has_real_bboxes = sum(1 for item in items if item.bbox is not None) >= 2
    if numbered_count >= 2 and numbered_count / len(items) >= 0.6:
        sorted_items = sorted(
            enumerate(items),
            key=lambda pair: (
                _ocr_question_number(pair[1].question_text) or 10_000,
                *_ocr_bbox_sort_key(pair[1].bbox),
                pair[0],
            ),
        )
    elif has_real_bboxes:
        sorted_items = sorted(
            enumerate(items),
            key=lambda pair: (*_ocr_bbox_sort_key(pair[1].bbox), pair[0]),
        )
    else:
        sorted_items = list(enumerate(items))
    renumbered_items = [
        item.model_copy(update={"item_index": index})
        for index, (_, item) in enumerate(sorted_items, start=1)
    ]
    return _with_default_bboxes(renumbered_items) if assign_default_bboxes else renumbered_items


def _sort_ocr_items_by_bbox_layout(items: list[OCRItemDraft]) -> list[OCRItemDraft]:
    if len(items) < 2:
        return items
    if sum(1 for item in items if item.bbox is not None) < 2:
        return [
            item.model_copy(update={"item_index": index})
            for index, item in enumerate(items, start=1)
        ]
    numbers = [_ocr_question_number(item.question_text) for item in items]
    concrete_numbers = [number for number in numbers if number is not None]
    if (
        len(concrete_numbers) >= 2
        and len(concrete_numbers) / len(items) >= 0.6
        and len(set(concrete_numbers)) == len(concrete_numbers)
    ):
        return _sort_ocr_items_by_number_and_bbox(items, assign_default_bboxes=False)
    sorted_items = sorted(
        enumerate(items),
        key=lambda pair: (*_ocr_bbox_top_left_sort_key(pair[1].bbox), pair[0]),
    )
    return [
        item.model_copy(update={"item_index": index})
        for index, (_, item) in enumerate(sorted_items, start=1)
    ]


def _ocr_question_number(question_text: str) -> int | None:
    match = re.match(r"^\s*(\d+)[\.\uff0e、)]", question_text or "")
    return int(match.group(1)) if match else None


def _ocr_bbox_sort_key(bbox: ImageBBox | None) -> tuple[int, int, int]:
    if bbox is None:
        return (10_000, 10_000, 10_000)
    row_bucket = max(0, bbox.y) // 70
    return (row_bucket, bbox.x, bbox.y)


def _ocr_bbox_top_left_sort_key(bbox: ImageBBox | None) -> tuple[int, int]:
    if bbox is None:
        return (10_000, 10_000)
    return (bbox.y, bbox.x)


def _should_prefer_raw_question(payload_item: OCRItemDraft, raw_item: OCRItemDraft) -> bool:
    payload_question = payload_item.question_text.strip()
    raw_question = raw_item.question_text.strip()
    if raw_item.child_answer and _contains_answer_marker(payload_question):
        return True
    if raw_item.child_answer and _question_contains_extracted_answer(payload_question, raw_item.child_answer):
        return True
    if len(raw_question) <= len(payload_question):
        return False
    if payload_item.child_answer and raw_item.child_answer:
        if _compact(payload_item.child_answer) != _compact(raw_item.child_answer):
            return False
    compact_payload = _compact(payload_question)
    compact_raw = _compact(raw_question)
    return bool(compact_payload and compact_payload in compact_raw)


def _contains_answer_marker(value: str) -> bool:
    return bool(
        re.search(
            r"(?:孩子|学生|child|student)?\s*(?:答案|answer)\s*[:：]",
            value or "",
            flags=re.IGNORECASE,
        )
    )


def _question_contains_extracted_answer(question_text: str, child_answer: str) -> bool:
    compact_question = _compact(question_text)
    if not compact_question:
        return False
    answer_parts = [
        _compact(part)
        for part in re.split(r"[；;,\n]+", child_answer or "")
        if _compact(part)
    ]
    return any(part and part in compact_question for part in answer_parts)


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value or "").lower()


def _should_enrich_raw_text_from_items(
    *,
    raw_text: str,
    items: list[OCRItemDraft],
    raw_text_items: list[OCRItemDraft],
) -> bool:
    if not items:
        return False
    if not raw_text.strip():
        return True
    if len(items) > len(raw_text_items):
        return True
    item_answer_count = sum(1 for item in items if item.child_answer.strip())
    raw_answer_count = sum(1 for item in raw_text_items if item.child_answer.strip())
    return item_answer_count > raw_answer_count


def _raw_text_from_items(items: list[OCRItemDraft]) -> str:
    return "\n\n".join(
        _build_raw_text(
            question_text=item.question_text,
            child_answer=item.child_answer,
            work_steps=item.work_steps,
        )
        for item in items
        if item.question_text.strip()
    )


def _build_raw_text(*, question_text: str, child_answer: str, work_steps: str) -> str:
    return "\n".join(
        value
        for value in [
            question_text.strip(),
            f"孩子答案：{child_answer.strip()}" if child_answer.strip() else "",
            f"解题过程：{work_steps.strip()}" if work_steps.strip() else "",
        ]
        if value
    )


def _items_from_text(raw_text: str, *, assign_default_bboxes: bool = True) -> list[OCRItemDraft]:
    try:
        draft = parse_text_submission(
            child_id="ocr_draft",
            subject="auto",
            grade=3,
            raw_text=raw_text,
        )
    except ValueError:
        return []
    items = [
        OCRItemDraft(
            item_index=item.item_index,
            question_text=item.question_text,
            child_answer=item.child_answer or "",
            work_steps=item.work_steps,
            confidence=item.confidence,
        )
        for item in draft.items
    ]
    return _with_default_bboxes(items) if assign_default_bboxes else items


def _items_from_payload(value: object) -> list[OCRItemDraft]:
    if not isinstance(value, list):
        return []
    items: list[OCRItemDraft] = []
    for index, raw_item in enumerate(value, start=1):
        if not isinstance(raw_item, dict):
            continue
        question = str(raw_item.get("question_text") or "").strip()
        if not question:
            continue
        items.append(
            OCRItemDraft(
                item_index=int(raw_item.get("item_index") or len(items) + 1),
                question_text=question,
                child_answer=str(raw_item.get("child_answer") or "").strip(),
                work_steps=str(raw_item.get("work_steps") or "").strip(),
                confidence=_safe_confidence(raw_item.get("confidence")),
                bbox=_bbox_from_payload(raw_item.get("bbox") or raw_item.get("bounding_box")),
            )
        )
    return _sort_ocr_items_by_number_and_bbox(items)


def _bbox_from_payload(value: object) -> ImageBBox | None:
    if not isinstance(value, dict):
        return None
    x = _safe_int(value.get("x"), default=-1)
    y = _safe_int(value.get("y"), default=-1)
    width = _safe_int(value.get("width"), default=-1)
    height = _safe_int(value.get("height"), default=-1)
    if width <= 0 or height <= 0:
        return None
    x = max(0, min(1000, x))
    y = max(0, min(1000, y))
    width = max(1, min(1000 - x, width))
    height = max(1, min(1000 - y, height))
    return ImageBBox(x=x, y=y, width=width, height=height)


def _safe_int(value: object, *, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _with_default_bboxes(items: list[OCRItemDraft]) -> list[OCRItemDraft]:
    if not items:
        return []
    box_height = max(120, 720 // len(items))
    normalized: list[OCRItemDraft] = []
    for index, item in enumerate(items, start=1):
        y = min(920, 80 + (index - 1) * box_height)
        bbox = item.bbox or ImageBBox(
            x=60,
            y=y,
            width=880,
            height=max(1, min(box_height, 1000 - y)),
        )
        normalized.append(item.model_copy(update={"item_index": index, "bbox": bbox}))
    return normalized


def _safe_confidence(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _overall_confidence(items: list[OCRItemDraft], *, default: float = 0.0) -> float:
    if not items:
        return default
    scored = [item.confidence for item in items if item.confidence > 0]
    if not scored:
        return default
    return round(sum(scored) / len(scored), 4)

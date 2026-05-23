from __future__ import annotations

from difflib import SequenceMatcher
import inspect
from pathlib import Path
import re
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from songguo.backend.services.learning.basic_subject_rubric import (
    BasicSubjectRubricContext,
    BasicSubjectRubricEvaluator,
)
from songguo.backend.services.learning.math_structuring import (
    MathProblemStructuringGateway,
    ProblemAnalysis,
    reliable_final_answer,
)
from songguo.backend.services.learning.photo_review import (
    ImageBBox,
    OCRDraft,
    VisionOCRProvider,
)
from songguo.backend.services.learning.submission_evaluator import (
    _fallback_misconception_tag,
    _judge_item,
)
from songguo.backend.services.learning.submission_intake import parse_text_submission
from songguo.backend.services.learning.submission_models import (
    EvidenceType,
    JudgeResult,
    LearningItem,
    LearningItemStatus,
    LearningSubmissionSnapshot,
    LearningSubmissionStatus,
    MasteryState,
)
from songguo.backend.services.learning.tutor_graph.basic_subject_graph import (
    is_basic_subject_route,
)


VISUAL_FALLBACK_KEY = "visual_fallback"
ACTIVE_VISUAL_FALLBACK_STATUSES = {"pending", "running"}
TERMINAL_VISUAL_FALLBACK_STATUSES = {"done", "failed", "needs_manual_confirm", "skipped"}


def visual_fallback_state(item: Any) -> dict[str, Any]:
    data_json = item.data_json if isinstance(item.data_json, dict) else {}
    state = data_json.get(VISUAL_FALLBACK_KEY)
    if not isinstance(state, dict):
        return {}
    return state


def visual_fallback_counts(items: list[Any]) -> dict[str, int]:
    counts = {
        "pending_visual_fallback_count": 0,
        "running_visual_fallback_count": 0,
        "done_visual_fallback_count": 0,
        "failed_visual_fallback_count": 0,
    }
    for item in items:
        status = str(visual_fallback_state(item).get("status") or "")
        if status == "pending":
            counts["pending_visual_fallback_count"] += 1
        elif status == "running":
            counts["running_visual_fallback_count"] += 1
        elif status == "done":
            counts["done_visual_fallback_count"] += 1
        elif status in {"failed", "needs_manual_confirm"}:
            counts["failed_visual_fallback_count"] += 1
    return counts


def mark_visual_fallback_pending_for_submission(
    *,
    store: Any,
    submission_id: str,
) -> LearningSubmissionSnapshot:
    submission = store.require_submission(submission_id)
    if str(submission.source_type) != "photo" or not submission.image_refs:
        return store.get_submission_snapshot(submission_id)

    items = store.list_submission_items(submission_id)
    if not items:
        store.add_submission_item(
            LearningItem(
                submission_id=submission.submission_id,
                child_id=submission.child_id,
                family_id=submission.family_id,
                item_index=1,
                question_text="",
                child_answer=None,
                detected_subject=submission.detected_subject or submission.subject,
                detected_task_type=submission.detected_task_type,
                evaluation_mode="visual_fallback_whole_page",
                judge_result=JudgeResult.NEEDS_MANUAL_CONFIRM,
                status=LearningItemStatus.NEEDS_MANUAL_CONFIRM,
                data_json={
                    "reason": "no_structured_items",
                    "whole_page_visual_fallback": True,
                    "evidence_trace": [
                        {
                            "stage": "visual_fallback",
                            "label": "视觉复核",
                            "text": "整页照片没有拆出题目，先排队做整页视觉复核。",
                            "confidence": 0.0,
                            "source": "no_structured_items",
                            "outcome": "pending",
                        }
                    ],
                    VISUAL_FALLBACK_KEY: {
                        "status": "pending",
                        "reason": "no_structured_items",
                        "message": "整页照片没有拆出题目，先排队做整页视觉复核。",
                        "attempts": 0,
                    },
                },
            )
        )
        return store.get_submission_snapshot(submission_id)

    for item in items:
        if not _item_needs_visual_fallback(item):
            continue
        state = visual_fallback_state(item)
        if str(state.get("status") or "") in ACTIVE_VISUAL_FALLBACK_STATUSES | TERMINAL_VISUAL_FALLBACK_STATUSES:
            continue
        reason = _fallback_reason_for_item(item)
        _set_visual_fallback_state(
            store=store,
            item=item,
            status="pending",
            reason=reason,
            message=_fallback_message_for_reason(reason),
        )

    return store.get_submission_snapshot(submission_id)


async def run_visual_fallback_step_async(
    *,
    store: Any,
    submission_id: str,
    ocr_provider: Any | None = None,
    math_gateway: MathProblemStructuringGateway | None = None,
    basic_rubric_evaluator: BasicSubjectRubricEvaluator | None = None,
    max_items: int = 1,
) -> LearningSubmissionSnapshot:
    submission = store.require_submission(submission_id)
    provider = ocr_provider or VisionOCRProvider()
    processed = 0
    for item in store.list_submission_items(submission_id):
        if processed >= max(1, int(max_items or 1)):
            break
        state = visual_fallback_state(item)
        if state.get("status") != "pending":
            continue
        processed += 1
        running_item = _set_visual_fallback_state(
            store=store,
            item=item,
            status="running",
            reason=str(state.get("reason") or _fallback_reason_for_item(item)),
            message="正在用视觉模型复核这道题。",
            attempts=int(state.get("attempts") or 0) + 1,
        )
        if str(state.get("reason") or "") == "deferred_judgement":
            _run_deferred_judgement_step(
                store=store,
                submission_id=submission_id,
                item=running_item,
                math_gateway=math_gateway,
                basic_rubric_evaluator=basic_rubric_evaluator,
            )
            continue
        started_at = perf_counter()
        try:
            content, filename = _visual_fallback_content(submission, running_item)
            draft = await _recognize_with_provider(
                provider,
                content,
                filename=filename,
                region_hints=_region_hints_for_item(running_item),
            )
            latency_ms = int((perf_counter() - started_at) * 1000)
            _record_visual_ocr_call(
                store=store,
                submission=submission,
                item=running_item,
                draft=draft,
                provider=provider,
                latency_ms=latency_ms,
                status="success" if _draft_has_useful_item(draft) else "empty",
            )
        except Exception as exc:
            store.record_ai_call(
                child_id=running_item.child_id,
                session_id=submission.submission_id,
                provider=_label_or_default(getattr(provider, "provider", ""), "visual_fallback"),
                model=_label_or_default(getattr(provider, "model", ""), "configured_vision_model"),
                operation="submission_visual_fallback.ocr",
                token_estimate=0,
                status="error",
                agent=type(provider).__name__,
                submission_id=submission.submission_id,
                item_id=running_item.item_id,
                latency_ms=int((perf_counter() - started_at) * 1000),
                failure_reason=exc.__class__.__name__,
                metadata={"item_index": running_item.item_index},
            )
            _set_visual_fallback_state(
                store=store,
                item=running_item,
                status="failed",
                reason="visual_provider_error",
                message="视觉复核失败，请手动确认或重新拍照。",
                failure_reason=exc.__class__.__name__,
            )
            continue

        recognized_items = _recognized_items_from_draft(draft=draft, submission=submission)
        if _is_whole_page_visual_fallback_item(running_item) and len(recognized_items) > 1:
            _apply_whole_page_visual_fallback_items(
                store=store,
                submission=submission,
                placeholder_item=running_item,
                recognized_items=recognized_items,
                draft=draft,
                original_reason=str(state.get("reason") or _fallback_reason_for_item(running_item)),
                math_gateway=math_gateway,
                basic_rubric_evaluator=basic_rubric_evaluator,
            )
            continue

        recognized = recognized_items[0] if recognized_items else _empty_recognized_item()
        if not recognized["question_text"] or not recognized["child_answer"]:
            _set_visual_fallback_state(
                store=store,
                item=running_item,
                status="needs_manual_confirm",
                reason="visual_result_incomplete",
                message="视觉复核仍没有拿到完整题目和孩子答案，请手动确认或重新拍照。",
                provider=draft.provider,
                model=draft.model,
                confidence=draft.confidence,
            )
            continue
        if not _visual_fallback_result_matches_item(running_item, recognized):
            _set_visual_fallback_state(
                store=store,
                item=running_item,
                status="needs_manual_confirm",
                reason="visual_result_mismatch",
                message="视觉复核识别到的题目和原题不一致，请手动确认或重新拍照。",
                provider=draft.provider,
                model=draft.model,
                confidence=draft.confidence,
            )
            continue

        updated_item = store.update_submission_item(
            running_item.item_id,
            question_text=recognized["question_text"],
            child_answer=recognized["child_answer"],
            confidence=max(running_item.confidence, float(recognized["confidence"] or 0.0)),
            data_json={
                **(running_item.data_json if isinstance(running_item.data_json, dict) else {}),
                VISUAL_FALLBACK_KEY: {
                    **visual_fallback_state(running_item),
                    "status": "done",
                    "reason": str(state.get("reason") or _fallback_reason_for_item(running_item)),
                    "message": "视觉复核已更新这道题。",
                    "provider": draft.provider,
                    "model": draft.model,
                    "confidence": float(recognized["confidence"] or draft.confidence or 0.0),
                },
            },
        )
        _evaluate_visual_fallback_item(
            store=store,
            submission_id=submission_id,
            item_id=updated_item.item_id,
            math_gateway=math_gateway,
            basic_rubric_evaluator=basic_rubric_evaluator,
        )

    _refresh_submission_status_after_visual_fallback(store=store, submission_id=submission_id)
    return store.get_submission_snapshot(submission_id)


def _item_needs_visual_fallback(item: Any) -> bool:
    if item.judge_result in {JudgeResult.CORRECT, JudgeResult.WRONG}:
        return False
    if item.judge_result in {JudgeResult.UNKNOWN, JudgeResult.NEEDS_MANUAL_CONFIRM}:
        return True
    return item.status == LearningItemStatus.NEEDS_MANUAL_CONFIRM


def _fallback_reason_for_item(item: Any) -> str:
    data_json = item.data_json if isinstance(item.data_json, dict) else {}
    quality_warnings = {str(value) for value in data_json.get("quality_warnings") or []}
    ocr_action = str(data_json.get("ocr_action") or "")
    confidence = float(getattr(item, "confidence", 0.0) or 0.0)
    has_ocr_metadata = bool(ocr_action or data_json.get("ocr_source") or quality_warnings)
    if not str(item.child_answer or "").strip() and has_ocr_metadata:
        return "ocr_answer_missing"
    if "low_confidence" in quality_warnings or "ocr_low_confidence" in quality_warnings:
        return "ocr_low_confidence"
    if 0 < confidence < 0.75 and has_ocr_metadata:
        return "ocr_low_confidence"
    if ocr_action == "RecognizeEduFormula" or "formula_recognition_uncertain" in quality_warnings:
        return "formula_recognition_uncertain"
    if not str(item.question_text or "").strip() or not str(item.child_answer or "").strip():
        return "missing_question_or_answer"
    return str(data_json.get("reason") or "needs_manual_confirm")


def _fallback_message_for_reason(reason: str) -> str:
    if reason == "ocr_answer_missing":
        return "这道题缺少孩子答案，正在单题复核，其他题可以先看结果。"
    if reason == "ocr_low_confidence":
        return "这道题识别置信度偏低，正在复核，其他题可以先看结果。"
    if reason == "formula_recognition_uncertain":
        return "这道题公式识别不稳定，正在复核，其他题可以先看结果。"
    if reason == "missing_question_or_answer":
        return "这道题题目或孩子答案没有识别完整，先排队做视觉复核，其他题可以先看结果。"
    if reason == "math_gateway_error":
        return "判题模型没有稳定解析这道题，先排队做视觉复核。"
    return "这道题需要更稳的视觉复核，其他题可以先看结果。"


def _set_visual_fallback_state(
    *,
    store: Any,
    item: Any,
    status: str,
    reason: str,
    message: str,
    attempts: int | None = None,
    failure_reason: str = "",
    provider: str = "",
    model: str = "",
    confidence: float | None = None,
) -> Any:
    existing = visual_fallback_state(item)
    next_state = {
        **existing,
        "status": status,
        "reason": reason,
        "message": message,
    }
    if attempts is not None:
        next_state["attempts"] = attempts
    elif "attempts" not in next_state:
        next_state["attempts"] = 0
    if failure_reason:
        next_state["failure_reason"] = failure_reason
    if provider:
        next_state["provider"] = provider
    if model:
        next_state["model"] = model
    if confidence is not None:
        next_state["confidence"] = confidence
    data_json = item.data_json if isinstance(item.data_json, dict) else {}
    return store.update_submission_item(
        item.item_id,
        data_json={
            **_append_visual_fallback_evidence(
                data_json,
                status=status,
                reason=reason,
                message=message,
                confidence=float(confidence or 0.0),
            ),
            VISUAL_FALLBACK_KEY: next_state,
        },
    )


def _append_visual_fallback_evidence(
    data_json: dict[str, Any],
    *,
    status: str,
    reason: str,
    message: str,
    confidence: float = 0.0,
) -> dict[str, Any]:
    trace = [entry for entry in data_json.get("evidence_trace", []) if isinstance(entry, dict)]
    entry = {
        "stage": "visual_fallback",
        "label": "视觉复核",
        "text": str(message or "").strip(),
        "confidence": max(0.0, min(1.0, confidence)),
        "source": str(reason or "").strip(),
        "outcome": str(status or "").strip(),
    }
    key = (entry["stage"], entry["label"], entry["text"], entry["outcome"])
    seen = {
        (
            str(item.get("stage") or ""),
            str(item.get("label") or ""),
            str(item.get("text") or ""),
            str(item.get("outcome") or ""),
        )
        for item in trace
    }
    if entry["text"] and key not in seen:
        trace.append(entry)
    return {**data_json, "evidence_trace": trace[-6:]}


def _visual_fallback_content(submission: Any, item: Any) -> tuple[bytes, str]:
    image_ref = str((submission.image_refs or [""])[0])
    if not image_ref or image_ref.startswith("artifact://"):
        raise FileNotFoundError("visual fallback requires a local saved image path")
    path = Path(image_ref)
    content = path.read_bytes()
    cropped = _crop_image_content(content, item.bbox_json)
    return cropped or content, path.name


def _region_hints_for_item(item: Any) -> list[ImageBBox]:
    bbox = item.bbox_json if isinstance(item.bbox_json, dict) else None
    if not bbox:
        return []
    try:
        return [ImageBBox(**bbox)]
    except Exception:
        return []


def _crop_image_content(content: bytes, bbox: dict[str, Any] | None) -> bytes | None:
    if not bbox:
        return None
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    image_height, image_width = image.shape[:2]
    x = float(bbox.get("x") or 0)
    y = float(bbox.get("y") or 0)
    width = float(bbox.get("width") or 0)
    height = float(bbox.get("height") or 0)
    if not width or not height:
        return None
    if max(x, y, width, height) <= 1000 and (x + width > image_width or y + height > image_height):
        x = x / 1000 * image_width
        y = y / 1000 * image_height
        width = width / 1000 * image_width
        height = height / 1000 * image_height
    pad_x = max(8, int(width * 0.04))
    pad_y = max(8, int(height * 0.04))
    left = max(0, int(x) - pad_x)
    top = max(0, int(y) - pad_y)
    right = min(image_width, int(x + width) + pad_x)
    bottom = min(image_height, int(y + height) + pad_y)
    if right <= left or bottom <= top:
        return None
    ok, encoded = cv2.imencode(".jpg", image[top:bottom, left:right], [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return encoded.tobytes() if ok else None


def _is_whole_page_visual_fallback_item(item: Any) -> bool:
    data_json = item.data_json if isinstance(item.data_json, dict) else {}
    return bool(data_json.get("whole_page_visual_fallback"))


async def _recognize_with_provider(
    provider: Any,
    content: bytes,
    *,
    filename: str,
    region_hints: list[ImageBBox],
) -> OCRDraft:
    recognize_async = getattr(provider, "recognize_async", None)
    if recognize_async:
        if _accepts_region_hints(recognize_async):
            return await recognize_async(content, filename=filename, region_hints=region_hints)
        return await recognize_async(content, filename=filename)
    recognize = getattr(provider, "recognize", None)
    if recognize is None:
        raise AttributeError("visual fallback OCR provider must implement recognize_async or recognize")
    if _accepts_region_hints(recognize):
        return recognize(content, filename=filename, region_hints=region_hints)
    return recognize(content, filename=filename)


def _accepts_region_hints(func: Any) -> bool:
    try:
        return "region_hints" in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def _label_or_default(value: Any, default: str) -> str:
    text = str(value or "").strip()
    return text or default


def _record_visual_ocr_call(
    *,
    store: Any,
    submission: Any,
    item: Any,
    draft: OCRDraft,
    provider: Any,
    latency_ms: int,
    status: str,
) -> None:
    store.record_ai_call(
        child_id=item.child_id,
        session_id=submission.submission_id,
        provider=_label_or_default(draft.provider or getattr(provider, "provider", ""), "visual_fallback"),
        model=_label_or_default(draft.model or getattr(provider, "model", ""), "configured_vision_model"),
        operation="submission_visual_fallback.ocr",
        token_estimate=0,
        status=status,
        agent=type(provider).__name__,
        submission_id=submission.submission_id,
        item_id=item.item_id,
        latency_ms=latency_ms,
        confidence=draft.confidence,
        metadata={
            "item_index": item.item_index,
            "source": draft.source,
            "item_count": len(draft.items),
            "raw_text_length": len(draft.raw_text),
        },
    )


def _draft_has_useful_item(draft: OCRDraft) -> bool:
    return any(
        item["question_text"] or item["child_answer"]
        for item in _recognized_items_from_draft(draft=draft, submission=None)
    )


def _best_recognized_item(*, draft: OCRDraft, submission: Any | None) -> dict[str, Any]:
    items = _recognized_items_from_draft(draft=draft, submission=submission)
    if items:
        return items[0]
    return _empty_recognized_item()


def _recognized_items_from_draft(*, draft: OCRDraft, submission: Any | None) -> list[dict[str, Any]]:
    if draft.items:
        return [
            {
                "question_text": str(item.question_text or "").strip(),
                "child_answer": str(item.child_answer or "").strip(),
                "confidence": float(item.confidence or draft.confidence or 0.0),
                "bbox": item.bbox.model_dump(mode="json") if item.bbox is not None else None,
            }
            for item in draft.items
        ]
    if draft.raw_text and submission is not None:
        try:
            parsed = parse_text_submission(
                child_id=submission.child_id,
                subject=submission.subject,
                grade=submission.grade,
                raw_text=draft.raw_text,
            )
            if parsed.items:
                item = parsed.items[0]
                return [
                    {
                        "question_text": str(item.question_text or "").strip(),
                        "child_answer": str(item.child_answer or "").strip(),
                        "confidence": float(item.confidence or draft.confidence or 0.0),
                        "bbox": None,
                    }
                    for item in parsed.items
                ]
        except ValueError:
            pass
    if draft.question_text or draft.child_answer:
        return [
            {
                "question_text": str(draft.question_text or "").strip(),
                "child_answer": str(draft.child_answer or "").strip(),
                "confidence": float(draft.confidence or 0.0),
                "bbox": None,
            }
        ]
    return []


def _empty_recognized_item() -> dict[str, Any]:
    return {
        "question_text": "",
        "child_answer": "",
        "confidence": 0.0,
        "bbox": None,
    }


def _visual_fallback_result_matches_item(item: Any, recognized: dict[str, Any]) -> bool:
    existing = _compact_question_for_match(str(getattr(item, "question_text", "") or ""))
    candidate = _compact_question_for_match(str(recognized.get("question_text") or ""))
    if not existing or not candidate:
        return True
    if len(existing) < 8 or len(candidate) < 8:
        return True
    if existing in candidate or candidate in existing:
        return True
    return SequenceMatcher(None, existing, candidate).ratio() >= 0.45


def _compact_question_for_match(value: str) -> str:
    text = (value or "").replace("×", "x").replace("＝", "=")
    return re.sub(r"[\s，。,.!?！？；;：:、\"'“”‘’（）()\[\]【】$^_{}=\\]+", "", text).lower()


def _apply_whole_page_visual_fallback_items(
    *,
    store: Any,
    submission: Any,
    placeholder_item: Any,
    recognized_items: list[dict[str, Any]],
    draft: OCRDraft,
    original_reason: str,
    math_gateway: MathProblemStructuringGateway | None,
    basic_rubric_evaluator: BasicSubjectRubricEvaluator | None,
) -> None:
    applied_items = []
    for index, recognized in enumerate(recognized_items, start=1):
        fallback_data = {
            **(placeholder_item.data_json if index == 1 and isinstance(placeholder_item.data_json, dict) else {}),
            VISUAL_FALLBACK_KEY: {
                "status": "done",
                "reason": original_reason,
                "message": "整页视觉复核已拆出这道题。",
                "provider": draft.provider,
                "model": draft.model,
                "confidence": float(recognized["confidence"] or draft.confidence or 0.0),
                "attempts": int(visual_fallback_state(placeholder_item).get("attempts") or 1),
            },
        }
        if index == 1:
            updated = store.update_submission_item(
                placeholder_item.item_id,
                item_index=index,
                question_text=recognized["question_text"],
                child_answer=recognized["child_answer"],
                confidence=max(placeholder_item.confidence, float(recognized["confidence"] or 0.0)),
                bbox_json=recognized.get("bbox") or placeholder_item.bbox_json,
                data_json=fallback_data,
            )
            applied_items.append(updated)
            continue
        added = store.add_submission_item(
            LearningItem(
                submission_id=submission.submission_id,
                child_id=placeholder_item.child_id,
                family_id=placeholder_item.family_id,
                item_index=index,
                question_text=recognized["question_text"],
                child_answer=recognized["child_answer"],
                detected_subject=placeholder_item.detected_subject,
                detected_task_type=placeholder_item.detected_task_type,
                evaluation_mode=placeholder_item.evaluation_mode,
                confidence=float(recognized["confidence"] or 0.0),
                bbox_json=recognized.get("bbox"),
                data_json=fallback_data,
            )
        )
        applied_items.append(added)

    for item in applied_items:
        _evaluate_visual_fallback_item(
            store=store,
            submission_id=submission.submission_id,
            item_id=item.item_id,
            math_gateway=math_gateway,
            basic_rubric_evaluator=basic_rubric_evaluator,
        )


def _evaluate_visual_fallback_item(
    *,
    store: Any,
    submission_id: str,
    item_id: str,
    math_gateway: MathProblemStructuringGateway | None,
    basic_rubric_evaluator: BasicSubjectRubricEvaluator | None,
) -> None:
    submission = store.require_submission(submission_id)
    if submission.route_to and is_basic_subject_route(submission.route_to):
        _evaluate_basic_visual_fallback_item(
            store=store,
            submission_id=submission_id,
            item_id=item_id,
            evaluator=basic_rubric_evaluator or BasicSubjectRubricEvaluator(),
        )
        return
    _evaluate_math_visual_fallback_item(
        store=store,
        submission_id=submission_id,
        item_id=item_id,
        gateway=math_gateway or MathProblemStructuringGateway(),
    )


def _run_deferred_judgement_step(
    *,
    store: Any,
    submission_id: str,
    item: Any,
    math_gateway: MathProblemStructuringGateway | None,
    basic_rubric_evaluator: BasicSubjectRubricEvaluator | None,
) -> None:
    _evaluate_visual_fallback_item(
        store=store,
        submission_id=submission_id,
        item_id=item.item_id,
        math_gateway=math_gateway,
        basic_rubric_evaluator=basic_rubric_evaluator,
    )
    latest = _require_submission_item(store, submission_id, item.item_id)
    current_state = visual_fallback_state(latest)
    store.update_submission_item(
        latest.item_id,
        data_json={
            **(latest.data_json if isinstance(latest.data_json, dict) else {}),
            VISUAL_FALLBACK_KEY: {
                **current_state,
                "status": "done",
                "reason": "deferred_judgement",
                "message": "这道题已完成补判。",
            },
        },
    )


def _evaluate_math_visual_fallback_item(
    *,
    store: Any,
    submission_id: str,
    item_id: str,
    gateway: MathProblemStructuringGateway,
) -> None:
    submission = store.require_submission(submission_id)
    item = _require_submission_item(store, submission_id, item_id)
    started_at = perf_counter()
    try:
        analysis = gateway.analyze(
            question_text=item.question_text,
            grade=submission.grade,
            subject=submission.subject,
        )
        latency_ms = int((perf_counter() - started_at) * 1000)
    except Exception as exc:
        store.record_ai_call(
            child_id=item.child_id,
            session_id=submission.submission_id,
            provider=str(getattr(gateway, "provider", "") or "math_gateway"),
            model=str(getattr(gateway, "model", "") or getattr(gateway, "model_name", "") or "configured"),
            operation="visual_fallback.math_gateway.analyze",
            token_estimate=0,
            status="error",
            agent="MathProblemStructuringGateway",
            submission_id=submission.submission_id,
            item_id=item.item_id,
            latency_ms=int((perf_counter() - started_at) * 1000),
            failure_reason=exc.__class__.__name__,
            metadata={"question_type_id": item.question_type_id},
        )
        _set_item_manual_after_visual_fallback(
            store=store,
            item=item,
            reason="math_gateway_error",
            message="视觉复核后仍无法稳定解析这道数学题，请手动确认。",
        )
        return

    _record_math_gateway_after_visual_fallback(
        store=store,
        gateway=gateway,
        submission=submission,
        item=item,
        analysis=analysis,
        latency_ms=latency_ms,
    )
    if analysis is None or not item.child_answer:
        _set_item_manual_after_visual_fallback(
            store=store,
            item=item,
            reason="missing_analysis_or_answer",
            message="视觉复核后仍缺少判题需要的信息，请手动确认。",
        )
        return

    judged = _judge_item(analysis=analysis, child_answer=item.child_answer)
    question_type_id = analysis.problem_type
    knowledge_point = analysis.knowledge_point
    correct_answer = reliable_final_answer(analysis)
    if judged.correct:
        updated_item = store.update_submission_item(
            item.item_id,
            judge_result=JudgeResult.CORRECT,
            status=LearningItemStatus.JUDGED,
            correct_answer=correct_answer,
            question_type_id=question_type_id,
            knowledge_point=knowledge_point,
            confidence=max(item.confidence, analysis.confidence),
            data_json={
                **(item.data_json if isinstance(item.data_json, dict) else {}),
                "problem_analysis": analysis.model_dump(mode="json"),
            },
        )
        store.save_mastery_evidence(
            item_id=updated_item.item_id,
            child_id=updated_item.child_id,
            question_type_id=question_type_id,
            evidence_type=EvidenceType.SUBMISSION_CORRECT,
            is_correct=True,
            mastery_state_after=MasteryState.OBSERVED,
            review_due=False,
            solved_without_help=True,
            confidence=analysis.confidence,
        )
        return

    misconception_tag = judged.misconception_tag or _fallback_misconception_tag(analysis)
    updated_item = store.update_submission_item(
        item.item_id,
        judge_result=JudgeResult.WRONG,
        status=LearningItemStatus.QUEUED_FOR_TUTORING,
        correct_answer=correct_answer,
        question_type_id=question_type_id,
        knowledge_point=knowledge_point,
        misconception_tag=misconception_tag,
        confidence=max(item.confidence, analysis.confidence),
        data_json={
            **(item.data_json if isinstance(item.data_json, dict) else {}),
            "problem_analysis": analysis.model_dump(mode="json"),
            "judge_evidence": judged.evidence,
        },
    )
    store.record_wrong_question(
        session_id=submission.submission_id,
        child_id=updated_item.child_id,
        normalized_question=updated_item.question_text,
        knowledge_point=knowledge_point,
        mistake_summary=judged.evidence or "视觉复核后判为错题，需要进入陪练。",
        last_misconception=misconception_tag,
        highest_hint_level=1,
    )
    store.save_mastery_evidence(
        item_id=updated_item.item_id,
        child_id=updated_item.child_id,
        question_type_id=question_type_id,
        evidence_type=EvidenceType.WRONG_UNRESOLVED,
        is_correct=False,
        mastery_state_after=MasteryState.NEEDS_REVIEW,
        review_due=True,
        misconception_tag=misconception_tag,
        confidence=analysis.confidence,
    )
    store.enqueue_tutor_item(
        submission_id=submission.submission_id,
        item_id=updated_item.item_id,
        child_id=updated_item.child_id,
        question_type_id=question_type_id,
        priority=9,
    )


def _record_math_gateway_after_visual_fallback(
    *,
    store: Any,
    gateway: Any,
    submission: Any,
    item: Any,
    analysis: ProblemAnalysis | None,
    latency_ms: int,
) -> None:
    store.record_ai_call(
        child_id=item.child_id,
        session_id=submission.submission_id,
        provider=str(getattr(gateway, "provider", "") or "math_gateway"),
        model=str(getattr(gateway, "model", "") or getattr(gateway, "model_name", "") or "configured"),
        operation="visual_fallback.math_gateway.analyze",
        token_estimate=0,
        status="success" if analysis is not None else "empty",
        agent="MathProblemStructuringGateway",
        submission_id=submission.submission_id,
        item_id=item.item_id,
        latency_ms=latency_ms,
        confidence=analysis.confidence if analysis is not None else 0.0,
        metadata={
            "question_type_id": analysis.problem_type if analysis is not None else item.question_type_id,
            "knowledge_point": analysis.knowledge_point if analysis is not None else item.knowledge_point,
            "source": analysis.source if analysis is not None else "",
        },
    )


def _evaluate_basic_visual_fallback_item(
    *,
    store: Any,
    submission_id: str,
    item_id: str,
    evaluator: BasicSubjectRubricEvaluator,
) -> None:
    submission = store.require_submission(submission_id)
    item = _require_submission_item(store, submission_id, item_id)
    subject = (submission.detected_subject or submission.subject or "general").lower()
    if not item.question_text.strip() or not (item.child_answer or "").strip():
        _set_item_manual_after_visual_fallback(
            store=store,
            item=item,
            reason="missing_question_or_answer",
            message="视觉复核后仍缺少题目或孩子答案，请手动确认。",
        )
        return

    started_at = perf_counter()
    rubric = evaluator.evaluate(
        BasicSubjectRubricContext(
            subject=subject,
            task_type=submission.detected_task_type or item.detected_task_type or "unknown",
            grade=submission.grade,
            question_text=item.question_text,
            child_answer=item.child_answer or "",
            route_to=submission.route_to,
        )
    )
    rubric_latency_ms = int((perf_counter() - started_at) * 1000)
    store.record_ai_call(
        child_id=item.child_id,
        session_id=submission.submission_id,
        provider=rubric.provider,
        model=rubric.model,
        operation="visual_fallback.basic_subject_rubric.evaluate",
        token_estimate=0,
        status=rubric.outcome,
        agent="BasicSubjectRubricEvaluator",
        submission_id=submission.submission_id,
        item_id=item.item_id,
        latency_ms=rubric_latency_ms,
        confidence=rubric.confidence,
        route_to=submission.route_to,
        metadata={
            "rubric_outcome": rubric.outcome,
            "question_type_id": rubric.question_type_id,
            "knowledge_point": rubric.knowledge_point,
            "source": rubric.source,
        },
    )
    rubric_data = rubric.model_dump(mode="json")
    if rubric.outcome == "needs_manual_confirm":
        store.update_submission_item(
            item.item_id,
            judge_result=JudgeResult.NEEDS_MANUAL_CONFIRM,
            status=LearningItemStatus.NEEDS_MANUAL_CONFIRM,
            question_type_id=rubric.question_type_id,
            knowledge_point=rubric.knowledge_point,
            misconception_tag=rubric.misconception_tag,
            confidence=max(item.confidence, rubric.confidence),
            data_json={
                **(item.data_json if isinstance(item.data_json, dict) else {}),
                "basic_subject_rubric": rubric_data,
                VISUAL_FALLBACK_KEY: {
                    **visual_fallback_state(item),
                    "status": "needs_manual_confirm",
                    "reason": "basic_subject_rubric_uncertain",
                    "message": "视觉复核后仍需要人工确认。",
                },
            },
        )
        return

    if rubric.is_correct:
        updated_item = store.update_submission_item(
            item.item_id,
            judge_result=JudgeResult.CORRECT,
            status=LearningItemStatus.JUDGED,
            question_type_id=rubric.question_type_id,
            knowledge_point=rubric.knowledge_point,
            misconception_tag=None,
            confidence=max(item.confidence, rubric.confidence),
            data_json={
                **(item.data_json if isinstance(item.data_json, dict) else {}),
                "basic_subject_tutor": True,
                "route_to": submission.route_to,
                "basic_subject_rubric": rubric_data,
            },
        )
        store.save_mastery_evidence(
            item_id=updated_item.item_id,
            child_id=updated_item.child_id,
            question_type_id=rubric.question_type_id,
            evidence_type=EvidenceType.SUBMISSION_CORRECT,
            is_correct=True,
            mastery_state_after=MasteryState.OBSERVED,
            review_due=False,
            solved_without_help=True,
            confidence=updated_item.confidence,
            data_json={"basic_subject_rubric": rubric_data},
        )
        return

    updated_item = store.update_submission_item(
        item.item_id,
        judge_result=JudgeResult.WRONG,
        status=LearningItemStatus.QUEUED_FOR_TUTORING,
        question_type_id=rubric.question_type_id,
        knowledge_point=rubric.knowledge_point,
        misconception_tag=rubric.misconception_tag,
        confidence=max(item.confidence, rubric.confidence, submission.subject_confidence),
        data_json={
            **(item.data_json if isinstance(item.data_json, dict) else {}),
            "basic_subject_tutor": True,
            "route_to": submission.route_to,
            "basic_subject_rubric": rubric_data,
        },
    )
    store.record_wrong_question(
        session_id=submission.submission_id,
        child_id=updated_item.child_id,
        normalized_question=updated_item.question_text,
        knowledge_point=rubric.knowledge_point,
        mistake_summary=rubric.feedback_summary or "视觉复核后判为需要基础陪练。",
        last_misconception=updated_item.misconception_tag,
        highest_hint_level=1,
    )
    store.save_mastery_evidence(
        item_id=updated_item.item_id,
        child_id=updated_item.child_id,
        question_type_id=rubric.question_type_id,
        evidence_type=EvidenceType.WRONG_UNRESOLVED,
        is_correct=False,
        mastery_state_after=MasteryState.NEEDS_REVIEW,
        review_due=True,
        misconception_tag=updated_item.misconception_tag,
        confidence=updated_item.confidence,
        data_json={"basic_subject_rubric": rubric_data},
    )
    store.enqueue_tutor_item(
        submission_id=submission.submission_id,
        item_id=updated_item.item_id,
        child_id=updated_item.child_id,
        question_type_id=rubric.question_type_id,
        priority=5,
    )


def _set_item_manual_after_visual_fallback(
    *,
    store: Any,
    item: Any,
    reason: str,
    message: str,
) -> None:
    store.update_submission_item(
        item.item_id,
        judge_result=JudgeResult.NEEDS_MANUAL_CONFIRM,
        status=LearningItemStatus.NEEDS_MANUAL_CONFIRM,
        data_json={
            **(item.data_json if isinstance(item.data_json, dict) else {}),
            VISUAL_FALLBACK_KEY: {
                **visual_fallback_state(item),
                "status": "needs_manual_confirm",
                "reason": reason,
                "message": message,
            },
        },
    )


def _require_submission_item(store: Any, submission_id: str, item_id: str) -> Any:
    for item in store.list_submission_items(submission_id):
        if item.item_id == item_id:
            return item
    raise KeyError(item_id)


def _refresh_submission_status_after_visual_fallback(*, store: Any, submission_id: str) -> None:
    snapshot = store.get_submission_snapshot(submission_id)
    if any(
        visual_fallback_state(item).get("status") in ACTIVE_VISUAL_FALLBACK_STATUSES
        for item in snapshot.items
    ):
        return
    if snapshot.tutor_queue:
        store.update_submission(submission_id, status=LearningSubmissionStatus.TUTORING)
        return
    if snapshot.submission.needs_manual_confirm_count:
        store.update_submission(submission_id, status=LearningSubmissionStatus.NEEDS_MANUAL_CONFIRM)
        return
    store.complete_submission_if_queue_done(submission_id)

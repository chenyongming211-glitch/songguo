from __future__ import annotations

from collections import Counter
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from songguo.backend.evaluation.ocr_eval import (
    OCRSample,
    _match_items,
    _ratio,
    _valid_bbox,
)
from songguo.backend.services.learning.photo_review import DeterministicOCRProvider
from songguo.backend.services.learning.service import LearningService
from songguo.backend.services.learning.store import InMemoryLearningStore
from songguo.backend.services.learning.submission_visual_fallback import visual_fallback_state


SUBJECT_ROUTES = {
    "math": "math_mistake_tutor",
    "chinese": "chinese_basic_tutor",
    "english": "english_basic_tutor",
}


class PhotoSubmissionItemEvalResult(BaseModel):
    item_index: int
    question_text: str = ""
    child_answer: str = ""
    judge_result: str = ""
    display_status: str = "pending"
    rubric_outcome: str = ""
    misconception_tag: str | None = None


class PhotoSubmissionE2EResult(BaseModel):
    sample_id: str
    status: str
    retry_count: int = 0
    elapsed_ms: int = 0
    ocr_elapsed_ms: int = 0
    submission_elapsed_ms: int = 0
    ocr_confidence: float = 0.0
    expected_subject: str = ""
    detected_subject: str = ""
    expected_route: str = ""
    route_to: str = ""
    route_confidence: float = 0.0
    submission_status: str = ""
    expected_item_count: int = 0
    item_count: int = 0
    usable_bbox_count: int = 0
    valid_bbox_count: int = 0
    actual_answer_count: int = 0
    answer_coverage_rate: float = 0.0
    ocr_action_counts: dict[str, int] = Field(default_factory=dict)
    question_match_count: int = 0
    answer_match_count: int = 0
    final_judged_item_count: int = 0
    fallback_running_count: int = 0
    pending_item_count: int = 0
    correct_count: int = 0
    wrong_count: int = 0
    needs_manual_confirm_count: int = 0
    errors: list[str] = Field(default_factory=list)
    raw_text: str = ""
    item_results: list[PhotoSubmissionItemEvalResult] = Field(default_factory=list)


class PhotoSubmissionE2EReport(BaseModel):
    total: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: float = 0.0
    subject_match_rate: float = 0.0
    route_match_rate: float = 0.0
    no_manual_confirm_rate: float = 0.0
    item_count_match_rate: float = 0.0
    usable_bbox_rate: float = 0.0
    valid_bbox_rate: float = 0.0
    answer_coverage_rate: float = 0.0
    fallback_running_rate: float = 0.0
    pending_rate: float = 0.0
    ocr_action_counts: dict[str, int] = Field(default_factory=dict)
    question_match_rate: float = 0.0
    answer_match_rate: float = 0.0
    final_judgement_rate: float = 0.0
    average_latency_ms: int = 0
    average_ocr_latency_ms: int = 0
    average_submission_latency_ms: int = 0
    average_ocr_confidence: float = 0.0
    results: list[PhotoSubmissionE2EResult] = Field(default_factory=list)


def evaluate_photo_submission_samples(
    *,
    samples: list[OCRSample],
    ocr_provider: Any | None = None,
    service: LearningService | None = None,
    limit: int | None = None,
    sample_retries: int = 0,
    start_tutor: bool = False,
) -> PhotoSubmissionE2EReport:
    provider = ocr_provider or DeterministicOCRProvider()
    selected = samples[: max(0, limit)] if limit is not None else samples
    results = [
        _evaluate_one(
            sample=sample,
            ocr_provider=provider,
            service=service or _build_service(),
            sample_retries=max(0, sample_retries),
            start_tutor=start_tutor,
        )
        for sample in selected
    ]
    return _build_report(results)


def _evaluate_one(
    *,
    sample: OCRSample,
    ocr_provider: Any,
    service: LearningService,
    sample_retries: int,
    start_tutor: bool,
) -> PhotoSubmissionE2EResult:
    started = time.perf_counter()
    expected_subject = _normalize_expected_subject(sample.subject)
    expected_route = SUBJECT_ROUTES.get(expected_subject, "")
    expected_count = len(sample.expected_items)
    errors: list[str] = []
    retry_count = 0
    ocr_elapsed_ms = 0
    draft = None
    ocr_error: Exception | None = None
    try:
        content = sample.image_path.read_bytes()
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return PhotoSubmissionE2EResult(
            sample_id=sample.sample_id,
            status="error",
            elapsed_ms=elapsed_ms,
            expected_subject=expected_subject,
            expected_route=expected_route,
            expected_item_count=expected_count,
            errors=[f"image_read_error:{exc.__class__.__name__}"],
        )
    for attempt_index in range(sample_retries + 1):
        ocr_started = time.perf_counter()
        try:
            draft = ocr_provider.recognize(content, filename=sample.image_path.name)
            ocr_elapsed_ms = int((time.perf_counter() - ocr_started) * 1000)
            retry_count = attempt_index
            _record_ocr_observation(
                service=service,
                sample=sample,
                provider=ocr_provider,
                draft=draft,
                latency_ms=ocr_elapsed_ms,
                retry_count=retry_count,
            )
            break
        except Exception as exc:
            error_elapsed_ms = int((time.perf_counter() - ocr_started) * 1000)
            if attempt_index < sample_retries:
                _record_ocr_error_observation(
                    service=service,
                    sample=sample,
                    provider=ocr_provider,
                    retry_count=attempt_index,
                    error=exc,
                    latency_ms=error_elapsed_ms,
                    will_retry=True,
                )
                continue
            ocr_error = exc
            ocr_elapsed_ms = error_elapsed_ms
    if draft is None:
        ocr_error = ocr_error or RuntimeError("OCR failed")
        _record_ocr_error_observation(
            service=service,
            sample=sample,
            provider=ocr_provider,
            retry_count=sample_retries,
            error=ocr_error,
            latency_ms=ocr_elapsed_ms,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return PhotoSubmissionE2EResult(
            sample_id=sample.sample_id,
            status="error",
            retry_count=sample_retries,
            elapsed_ms=elapsed_ms,
            expected_subject=expected_subject,
            expected_route=expected_route,
            expected_item_count=expected_count,
            errors=[f"ocr_error:{ocr_error.__class__.__name__}"],
        )

    ocr_matches = _match_items(sample.expected_items, draft.items)
    actual_count = len(draft.items)
    usable_bbox_count = sum(1 for item in draft.items if item.bbox is not None)
    valid_bbox_count = sum(1 for item in draft.items if _valid_bbox(getattr(item, "bbox", None)))
    actual_answer_count = sum(1 for item in draft.items if str(item.child_answer or "").strip())
    answer_coverage_denominator = expected_count or actual_count
    ocr_action_counts = _ocr_action_counts(draft)
    question_match_count = sum(1 for match in ocr_matches if match.question_matched)
    answer_match_count = sum(1 for match in ocr_matches if match.answer_matched)
    if expected_count != actual_count:
        errors.append(f"ocr_item_count_mismatch:expected_{expected_count}:actual_{actual_count}")
    for match in ocr_matches:
        if not match.question_matched:
            errors.append(f"ocr_question_mismatch:item_{match.item_index}")
        if match.expected_answer and not match.answer_matched:
            errors.append(f"ocr_answer_mismatch:item_{match.item_index}")

    try:
        submission_started = time.perf_counter()
        created = service.create_submission(
            child_id="child_photo_e2e",
            subject="auto",
            grade=sample.grade,
            source_type="photo",
            raw_text=draft.raw_text,
            image_refs=[str(sample.image_path)],
            item_bboxes=_item_bboxes_from_draft(draft),
            item_metadata=_item_metadata_from_draft(draft),
        )
        service.confirm_submission(created.submission_id, start_tutor=start_tutor)
        submission_elapsed_ms = int((time.perf_counter() - submission_started) * 1000)
        snapshot = service.get_submission_snapshot(created.submission_id)
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return PhotoSubmissionE2EResult(
            sample_id=sample.sample_id,
            status="error",
            retry_count=retry_count,
            elapsed_ms=elapsed_ms,
            ocr_elapsed_ms=ocr_elapsed_ms,
            ocr_confidence=draft.confidence,
            expected_subject=expected_subject,
            expected_route=expected_route,
            expected_item_count=expected_count,
            actual_answer_count=actual_answer_count,
            answer_coverage_rate=_ratio(actual_answer_count, answer_coverage_denominator),
            ocr_action_counts=ocr_action_counts,
            question_match_count=question_match_count,
            answer_match_count=answer_match_count,
            errors=[*errors, f"submission_error:{exc.__class__.__name__}"],
            raw_text=draft.raw_text,
        )

    submission = snapshot.submission
    detected_subject = (submission.detected_subject or submission.subject or "").lower()
    route_to = submission.route_to or ""
    submission_status = str(submission.status)
    if expected_subject and detected_subject != expected_subject:
        errors.append(f"subject_mismatch:expected_{expected_subject}:actual_{detected_subject}")
    if expected_route and route_to != expected_route:
        errors.append(f"route_mismatch:expected_{expected_route}:actual_{route_to}")
    if submission_status == "needs_manual_confirm" or submission.needs_manual_confirm_count:
        errors.append("needs_manual_confirm")
    if expected_count and len(snapshot.items) != expected_count:
        errors.append(f"submission_item_count_mismatch:expected_{expected_count}:actual_{len(snapshot.items)}")
    for item in snapshot.items:
        if str(item.judge_result) in {"", "unknown"} and not item.data_json.get("basic_subject_rubric"):
            errors.append(f"item_not_judged:item_{item.item_index}")
    final_judged_item_count = sum(
        1
        for item in snapshot.items
        if str(item.judge_result) not in {"", "unknown"}
        or bool(item.data_json.get("basic_subject_rubric"))
    )
    fallback_running_count = sum(
        1
        for item in snapshot.items
        if str(visual_fallback_state(item).get("status") or "") in {"pending", "running"}
    )
    pending_item_count = sum(1 for item in snapshot.items if _item_is_pending(item))

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return PhotoSubmissionE2EResult(
        sample_id=sample.sample_id,
        status="passed" if not errors else "failed",
        retry_count=retry_count,
        elapsed_ms=elapsed_ms,
        ocr_elapsed_ms=ocr_elapsed_ms,
        submission_elapsed_ms=submission_elapsed_ms,
        ocr_confidence=draft.confidence,
        expected_subject=expected_subject,
        detected_subject=detected_subject,
        expected_route=expected_route,
        route_to=route_to,
        route_confidence=submission.subject_confidence,
        submission_status=submission_status,
        expected_item_count=expected_count,
        item_count=submission.item_count,
        usable_bbox_count=usable_bbox_count,
        valid_bbox_count=valid_bbox_count,
        actual_answer_count=actual_answer_count,
        answer_coverage_rate=_ratio(actual_answer_count, answer_coverage_denominator),
        ocr_action_counts=ocr_action_counts,
        question_match_count=question_match_count,
        answer_match_count=answer_match_count,
        final_judged_item_count=final_judged_item_count,
        fallback_running_count=fallback_running_count,
        pending_item_count=pending_item_count,
        correct_count=submission.correct_count,
        wrong_count=submission.wrong_count,
        needs_manual_confirm_count=submission.needs_manual_confirm_count,
        errors=errors,
        raw_text=draft.raw_text,
        item_results=[
            PhotoSubmissionItemEvalResult(
                item_index=item.item_index,
                question_text=item.question_text,
                child_answer=item.child_answer or "",
                judge_result=str(item.judge_result),
                display_status=_display_status_for_item(item),
                rubric_outcome=str(
                    (item.data_json.get("basic_subject_rubric") or {}).get("outcome") or ""
                ),
                misconception_tag=item.misconception_tag,
            )
            for item in snapshot.items
        ],
    )


def _record_ocr_observation(
    *,
    service: LearningService,
    sample: OCRSample,
    provider: Any,
    draft: Any,
    latency_ms: int,
    retry_count: int,
) -> None:
    store = getattr(service, "store", None)
    record_ai_call = getattr(store, "record_ai_call", None)
    if record_ai_call is None:
        return
    record_ai_call(
        child_id="child_photo_e2e",
        session_id=f"ocr:{sample.sample_id}",
        provider=str(getattr(draft, "provider", "") or "photo_ocr"),
        model=str(getattr(draft, "model", "") or "configured"),
        operation="photo_ocr.recognize",
        token_estimate=0,
        status="success",
        agent=type(provider).__name__,
        latency_ms=latency_ms,
        confidence=float(getattr(draft, "confidence", 0.0) or 0.0),
        metadata={
            "sample_id": sample.sample_id,
            "filename": sample.image_path.name,
            "item_count": len(getattr(draft, "items", []) or []),
            "answer_count": sum(
                1
                for item in getattr(draft, "items", []) or []
                if str(getattr(item, "child_answer", "") or "").strip()
            ),
            "ocr_action_counts": _ocr_action_counts(draft),
            "needs_confirmation": bool(getattr(draft, "needs_confirmation", False)),
            "retry_count": retry_count,
            "source": str(getattr(draft, "source", "") or ""),
        },
    )


def _record_ocr_error_observation(
    *,
    service: LearningService,
    sample: OCRSample,
    provider: Any,
    retry_count: int,
    error: Exception,
    latency_ms: int = 0,
    will_retry: bool = False,
) -> None:
    store = getattr(service, "store", None)
    record_ai_call = getattr(store, "record_ai_call", None)
    if record_ai_call is None:
        return
    record_ai_call(
        child_id="child_photo_e2e",
        session_id=f"ocr:{sample.sample_id}",
        provider="photo_ocr",
        model="configured",
        operation="photo_ocr.recognize",
        token_estimate=0,
        status="error",
        agent=type(provider).__name__,
        latency_ms=latency_ms,
        failure_reason=error.__class__.__name__,
        metadata={
            "sample_id": sample.sample_id,
            "filename": sample.image_path.name,
            "retry_count": retry_count,
            "will_retry": will_retry,
        },
    )


def _build_report(results: list[PhotoSubmissionE2EResult]) -> PhotoSubmissionE2EReport:
    total = len(results)
    passed = sum(1 for result in results if result.status == "passed")
    expected_item_total = sum(result.expected_item_count for result in results)
    actual_item_total = sum(result.item_count for result in results)
    answer_coverage_denominator = expected_item_total or actual_item_total
    confidence_values = [result.ocr_confidence for result in results if result.ocr_confidence > 0]
    action_counts = Counter()
    for result in results:
        action_counts.update(result.ocr_action_counts)
    return PhotoSubmissionE2EReport(
        total=total,
        passed=passed,
        failed=total - passed,
        pass_rate=_ratio(passed, total),
        subject_match_rate=_ratio(
            sum(
                1
                for result in results
                if result.expected_subject and result.detected_subject == result.expected_subject
            ),
            total,
        ),
        route_match_rate=_ratio(
            sum(
                1
                for result in results
                if result.expected_route and result.route_to == result.expected_route
            ),
            total,
        ),
        no_manual_confirm_rate=_ratio(
            sum(1 for result in results if "needs_manual_confirm" not in result.errors),
            total,
        ),
        item_count_match_rate=_ratio(
            sum(
                1
                for result in results
                if not any(error.startswith("submission_item_count_mismatch") for error in result.errors)
            ),
            total,
        ),
        usable_bbox_rate=_ratio(
            sum(result.usable_bbox_count for result in results),
            expected_item_total,
        ),
        valid_bbox_rate=_ratio(
            sum(result.valid_bbox_count for result in results),
            expected_item_total,
        ),
        answer_coverage_rate=_ratio(
            sum(result.actual_answer_count for result in results),
            answer_coverage_denominator,
        ),
        fallback_running_rate=_ratio(
            sum(result.fallback_running_count for result in results),
            expected_item_total or actual_item_total,
        ),
        pending_rate=_ratio(
            sum(result.pending_item_count for result in results),
            expected_item_total or actual_item_total,
        ),
        ocr_action_counts=dict(sorted(action_counts.items())),
        question_match_rate=_ratio(
            sum(result.question_match_count for result in results),
            expected_item_total,
        ),
        answer_match_rate=_ratio(
            sum(result.answer_match_count for result in results),
            expected_item_total,
        ),
        final_judgement_rate=_ratio(
            sum(result.final_judged_item_count for result in results),
            expected_item_total,
        ),
        average_latency_ms=round(sum(result.elapsed_ms for result in results) / total) if total else 0,
        average_ocr_latency_ms=round(sum(result.ocr_elapsed_ms for result in results) / total) if total else 0,
        average_submission_latency_ms=round(sum(result.submission_elapsed_ms for result in results) / total)
        if total
        else 0,
        average_ocr_confidence=round(sum(confidence_values) / len(confidence_values), 4)
        if confidence_values
        else 0.0,
        results=results,
    )


def _build_service() -> LearningService:
    return LearningService(store=InMemoryLearningStore(), agent_runtime="langgraph")


def _normalize_expected_subject(subject: str) -> str:
    normalized = (subject or "").strip().lower()
    return "" if normalized in {"", "auto", "unknown"} else normalized


def _item_bboxes_from_draft(draft: Any) -> dict[int, dict[str, int]]:
    bboxes: dict[int, dict[str, int]] = {}
    for item in getattr(draft, "items", []) or []:
        bbox = getattr(item, "bbox", None)
        if bbox is None:
            continue
        dump = bbox.model_dump(mode="json") if hasattr(bbox, "model_dump") else dict(bbox)
        bboxes[int(getattr(item, "item_index", 0) or 0)] = dump
    return bboxes


def _item_metadata_from_draft(draft: Any) -> dict[int, dict[str, object]]:
    metadata: dict[int, dict[str, object]] = {}
    for item in getattr(draft, "items", []) or []:
        item_index = int(getattr(item, "item_index", 0) or 0)
        if not item_index:
            continue
        values = {
            "ocr_action": str(getattr(item, "source_action", "") or ""),
            "ocr_source": str(getattr(draft, "source", "") or ""),
            "display_status": "pending",
            "quality_warnings": list(getattr(item, "quality_warnings", []) or []),
        }
        if values["ocr_action"] or values["ocr_source"] or values["quality_warnings"]:
            metadata[item_index] = values
    return metadata


def _ocr_action_counts(draft: Any) -> dict[str, int]:
    counts: Counter[str] = Counter()
    data_json = getattr(draft, "data_json", {})
    plan = data_json.get("ocr_plan") if isinstance(data_json, dict) else None
    if isinstance(plan, dict):
        primary_action = str(plan.get("primary_action") or "")
        if primary_action:
            counts[primary_action] += 1
        for action in plan.get("secondary_actions") or []:
            action_name = str(action or "")
            if action_name:
                counts[action_name] += 1
    if not counts:
        for item in getattr(draft, "items", []) or []:
            action_name = str(getattr(item, "source_action", "") or "")
            if action_name:
                counts[action_name] += 1
    source = str(getattr(draft, "source", "") or "")
    if not counts and source:
        counts[source] += 1
    return dict(sorted(counts.items()))


def _display_status_for_item(item: Any) -> str:
    if str(visual_fallback_state(item).get("status") or "") in {"pending", "running"}:
        return "fallback_running"
    judge_result = str(getattr(item, "judge_result", "") or "")
    if judge_result == "correct":
        return "correct"
    if judge_result == "wrong":
        return "wrong"
    return "pending"


def _item_is_pending(item: Any) -> bool:
    if str(visual_fallback_state(item).get("status") or "") in {"pending", "running"}:
        return True
    judge_result = str(getattr(item, "judge_result", "") or "")
    return judge_result in {"", "unknown", "needs_manual_confirm"}

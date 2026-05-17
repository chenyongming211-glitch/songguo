from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
import json
import re
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from songguo.backend.services.learning.photo_review import (
    DeterministicOCRProvider,
    OCRDraft,
    OCRItemDraft,
)


QUESTION_MATCH_THRESHOLD = 0.86
ANSWER_MATCH_THRESHOLD = 0.92


class OCRExpectedItem(BaseModel):
    question_text: str
    child_answer: str = ""


class OCRSample(BaseModel):
    sample_id: str
    image_path: Path
    subject: str = "auto"
    grade: int = 3
    expected_items: list[OCRExpectedItem] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class OCRItemMatch(BaseModel):
    item_index: int
    expected_question: str = ""
    actual_question: str = ""
    expected_answer: str = ""
    actual_answer: str = ""
    question_score: float = 0.0
    answer_score: float = 0.0
    question_matched: bool = False
    answer_matched: bool = False


class OCRSampleResult(BaseModel):
    sample_id: str
    status: str
    elapsed_ms: int = 0
    confidence: float = 0.0
    needs_confirmation: bool = True
    expected_item_count: int = 0
    actual_item_count: int = 0
    item_count_matched: bool = False
    usable_bbox_count: int = 0
    valid_bbox_count: int = 0
    actual_answer_count: int = 0
    answer_coverage_rate: float = 0.0
    ocr_action_counts: dict[str, int] = Field(default_factory=dict)
    question_match_count: int = 0
    answer_match_count: int = 0
    errors: list[str] = Field(default_factory=list)
    raw_text: str = ""
    actual_items: list[OCRItemDraft] = Field(default_factory=list)
    matches: list[OCRItemMatch] = Field(default_factory=list)


class OCREvaluationReport(BaseModel):
    total: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: float = 0.0
    item_count_match_rate: float = 0.0
    usable_bbox_rate: float = 0.0
    valid_bbox_rate: float = 0.0
    answer_coverage_rate: float = 0.0
    ocr_action_counts: dict[str, int] = Field(default_factory=dict)
    question_match_rate: float = 0.0
    answer_match_rate: float = 0.0
    average_latency_ms: int = 0
    average_confidence: float = 0.0
    results: list[OCRSampleResult] = Field(default_factory=list)


def load_ocr_samples(manifest_path: Path | str) -> list[OCRSample]:
    manifest = Path(manifest_path)
    base_dir = manifest.parent
    samples: list[OCRSample] = []
    for line_number, raw_line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        payload = json.loads(line)
        image_path = Path(str(payload.get("image_path") or ""))
        if not image_path.is_absolute():
            image_path = base_dir / image_path
        payload["image_path"] = image_path
        payload.setdefault("sample_id", f"{manifest.stem}_{line_number}")
        samples.append(OCRSample.model_validate(payload))
    return samples


def evaluate_ocr_samples(
    *,
    samples: list[OCRSample],
    ocr_provider: Any | None = None,
    limit: int | None = None,
) -> OCREvaluationReport:
    provider = ocr_provider or DeterministicOCRProvider()
    selected = samples[: max(0, limit)] if limit is not None else samples
    results = [_evaluate_one(sample, provider) for sample in selected]
    return _build_report(results)


def _evaluate_one(sample: OCRSample, provider: Any) -> OCRSampleResult:
    started = time.perf_counter()
    errors: list[str] = []
    try:
        content = sample.image_path.read_bytes()
        draft = provider.recognize(content, filename=sample.image_path.name)
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return OCRSampleResult(
            sample_id=sample.sample_id,
            status="error",
            elapsed_ms=elapsed_ms,
            expected_item_count=len(sample.expected_items),
            errors=[f"provider_error:{exc.__class__.__name__}"],
        )

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    matches = _match_items(sample.expected_items, draft.items)
    expected_count = len(sample.expected_items)
    actual_count = len(draft.items)
    usable_bbox_count = sum(1 for item in draft.items if item.bbox is not None)
    valid_bbox_count = sum(1 for item in draft.items if _valid_bbox(getattr(item, "bbox", None)))
    actual_answer_count = sum(1 for item in draft.items if str(item.child_answer or "").strip())
    answer_coverage_denominator = expected_count or actual_count
    item_count_matched = expected_count == actual_count
    if not item_count_matched:
        errors.append(f"item_count_mismatch:expected_{expected_count}:actual_{actual_count}")
    for match in matches:
        if not match.question_matched:
            errors.append(f"question_mismatch:item_{match.item_index}")
        if match.expected_answer and not match.answer_matched:
            errors.append(f"answer_mismatch:item_{match.item_index}")

    status = "passed" if not errors else "failed"
    return OCRSampleResult(
        sample_id=sample.sample_id,
        status=status,
        elapsed_ms=elapsed_ms,
        confidence=draft.confidence,
        needs_confirmation=draft.needs_confirmation,
        expected_item_count=expected_count,
        actual_item_count=actual_count,
        item_count_matched=item_count_matched,
        usable_bbox_count=usable_bbox_count,
        valid_bbox_count=valid_bbox_count,
        actual_answer_count=actual_answer_count,
        answer_coverage_rate=_ratio(actual_answer_count, answer_coverage_denominator),
        ocr_action_counts=_ocr_action_counts(draft),
        question_match_count=sum(1 for match in matches if match.question_matched),
        answer_match_count=sum(1 for match in matches if match.answer_matched),
        errors=errors,
        raw_text=draft.raw_text,
        actual_items=draft.items,
        matches=matches,
    )


def _match_items(
    expected_items: list[OCRExpectedItem],
    actual_items: list[OCRItemDraft],
) -> list[OCRItemMatch]:
    matches: list[OCRItemMatch] = []
    for index, expected in enumerate(expected_items, start=1):
        actual = actual_items[index - 1] if index - 1 < len(actual_items) else OCRItemDraft(
            item_index=index,
            question_text="",
        )
        question_score = _similarity(expected.question_text, actual.question_text)
        answer_score = _similarity(expected.child_answer, actual.child_answer)
        matches.append(
            OCRItemMatch(
                item_index=index,
                expected_question=expected.question_text,
                actual_question=actual.question_text,
                expected_answer=expected.child_answer,
                actual_answer=actual.child_answer,
                question_score=question_score,
                answer_score=answer_score,
                question_matched=question_score >= QUESTION_MATCH_THRESHOLD,
                answer_matched=not expected.child_answer or answer_score >= ANSWER_MATCH_THRESHOLD,
            )
        )
    return matches


def _build_report(results: list[OCRSampleResult]) -> OCREvaluationReport:
    total = len(results)
    passed = sum(1 for result in results if result.status == "passed")
    item_count_matches = sum(1 for result in results if result.item_count_matched)
    expected_item_total = sum(result.expected_item_count for result in results)
    actual_item_total = sum(result.actual_item_count for result in results)
    answer_coverage_denominator = expected_item_total or actual_item_total
    question_matches = sum(result.question_match_count for result in results)
    answer_matches = sum(result.answer_match_count for result in results)
    confidence_values = [result.confidence for result in results if result.confidence > 0]
    action_counts = Counter()
    for result in results:
        action_counts.update(result.ocr_action_counts)
    return OCREvaluationReport(
        total=total,
        passed=passed,
        failed=total - passed,
        pass_rate=_ratio(passed, total),
        item_count_match_rate=_ratio(item_count_matches, total),
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
        ocr_action_counts=dict(sorted(action_counts.items())),
        question_match_rate=_ratio(question_matches, expected_item_total),
        answer_match_rate=_ratio(answer_matches, expected_item_total),
        average_latency_ms=round(
            sum(result.elapsed_ms for result in results) / total
        )
        if total
        else 0,
        average_confidence=round(sum(confidence_values) / len(confidence_values), 4)
        if confidence_values
        else 0.0,
        results=results,
    )


def _similarity(expected: str, actual: str) -> float:
    normalized_expected = _normalize_text(expected)
    normalized_actual = _normalize_text(actual)
    if not normalized_expected and not normalized_actual:
        return 1.0
    if not normalized_expected or not normalized_actual:
        return 0.0
    if normalized_expected == normalized_actual:
        return 1.0
    return round(SequenceMatcher(None, normalized_expected, normalized_actual).ratio(), 4)


def _normalize_text(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", "", text)
    text = text.replace("：", ":").replace("？", "?").replace("＝", "=")
    text = text.replace("×", "*").replace("÷", "/")
    text = re.sub(r"[\"'“”‘’]", "", text)
    text = re.sub(r"[，。,.!！;；]", "", text)
    return text


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _ocr_action_counts(draft: OCRDraft) -> dict[str, int]:
    counts: Counter[str] = Counter()
    data_json = draft.data_json if isinstance(draft.data_json, dict) else {}
    plan = data_json.get("ocr_plan")
    if isinstance(plan, dict):
        primary_action = str(plan.get("primary_action") or "")
        if primary_action:
            counts[primary_action] += 1
        for action in plan.get("secondary_actions") or []:
            action_name = str(action or "")
            if action_name:
                counts[action_name] += 1
    if not counts:
        for item in draft.items:
            action_name = str(getattr(item, "source_action", "") or "")
            if action_name:
                counts[action_name] += 1
    if not counts:
        source = str(getattr(draft, "source", "") or "")
        if source:
            counts[source] += 1
    return dict(sorted(counts.items()))


def _valid_bbox(bbox: Any | None) -> bool:
    if bbox is None:
        return False
    x = int(getattr(bbox, "x", 0) or 0)
    y = int(getattr(bbox, "y", 0) or 0)
    width = int(getattr(bbox, "width", 0) or 0)
    height = int(getattr(bbox, "height", 0) or 0)
    return (
        0 <= x <= 1000
        and 0 <= y <= 1000
        and width > 0
        and height > 0
        and x + width <= 1000
        and y + height <= 1000
    )

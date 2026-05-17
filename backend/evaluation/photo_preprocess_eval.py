from __future__ import annotations

import statistics
import time

from pydantic import BaseModel, Field

from songguo.backend.evaluation.ocr_eval import OCRSample, _ratio, _valid_bbox
from songguo.backend.services.learning.photo_preprocess import analyze_homework_photo


class PhotoPreprocessSampleResult(BaseModel):
    sample_id: str
    status: str
    elapsed_ms: int = 0
    expected_item_count: int = 0
    region_count: int = 0
    usable_region: bool = False
    strict_region_count_matched: bool = False
    valid_box_count: int = 0
    ordered_regions: bool = True
    quality_warnings: list[str] = Field(default_factory=list)
    preprocess_source: str = ""
    errors: list[str] = Field(default_factory=list)


class PhotoPreprocessEvaluationReport(BaseModel):
    total: int = 0
    failure_count: int = 0
    usable_region_rate: float = 0.0
    strict_region_rate: float = 0.0
    valid_box_rate: float = 0.0
    ordered_region_rate: float = 0.0
    quality_warning_rate: float = 0.0
    average_preprocess_ms: int = 0
    p95_preprocess_ms: int = 0
    max_preprocess_ms: int = 0
    by_subject: dict[str, dict[str, float | int]] = Field(default_factory=dict)
    results: list[PhotoPreprocessSampleResult] = Field(default_factory=list)


def evaluate_photo_preprocess_samples(
    *,
    samples: list[OCRSample],
    limit: int | None = None,
) -> PhotoPreprocessEvaluationReport:
    selected = samples[: max(0, limit)] if limit is not None else samples
    results = [_evaluate_one(sample) for sample in selected]
    return _build_report(results, selected)


def _evaluate_one(sample: OCRSample) -> PhotoPreprocessSampleResult:
    started = time.perf_counter()
    expected_count = len(sample.expected_items)
    try:
        content = sample.image_path.read_bytes()
        analysis = analyze_homework_photo(content, filename=sample.image_path.name)
    except Exception as exc:
        return PhotoPreprocessSampleResult(
            sample_id=sample.sample_id,
            status="error",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            expected_item_count=expected_count,
            errors=[f"preprocess_error:{exc.__class__.__name__}"],
        )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    region_bboxes = analysis.question_regions
    y_values = [region.y for region in region_bboxes]
    valid_box_count = sum(1 for region in region_bboxes if _valid_bbox(region))
    errors: list[str] = []
    if not region_bboxes:
        errors.append("no_usable_region")
    if expected_count and len(region_bboxes) != expected_count:
        errors.append(f"region_count_mismatch:expected_{expected_count}:actual_{len(region_bboxes)}")
    if valid_box_count != len(region_bboxes):
        errors.append("invalid_region_bbox")
    if y_values != sorted(y_values):
        errors.append("region_order_mismatch")
    return PhotoPreprocessSampleResult(
        sample_id=sample.sample_id,
        status="passed" if not errors else "failed",
        elapsed_ms=elapsed_ms,
        expected_item_count=expected_count,
        region_count=len(region_bboxes),
        usable_region=bool(region_bboxes),
        strict_region_count_matched=bool(expected_count and len(region_bboxes) == expected_count),
        valid_box_count=valid_box_count,
        ordered_regions=y_values == sorted(y_values),
        quality_warnings=analysis.quality_warnings,
        preprocess_source=analysis.source,
        errors=errors,
    )


def _build_report(
    results: list[PhotoPreprocessSampleResult],
    samples: list[OCRSample],
) -> PhotoPreprocessEvaluationReport:
    total = len(results)
    expected_item_total = sum(result.expected_item_count for result in results)
    region_box_denominator = sum(
        max(result.expected_item_count, result.region_count)
        for result in results
    )
    elapsed_values = [result.elapsed_ms for result in results]
    by_subject: dict[str, dict[str, float | int]] = {}
    samples_by_id = {sample.sample_id: sample for sample in samples}
    for result in results:
        subject = samples_by_id.get(result.sample_id, OCRSample(sample_id=result.sample_id, image_path="")).subject
        bucket = by_subject.setdefault(
            subject,
            {
                "total": 0,
                "usable_count": 0,
                "strict_count": 0,
                "valid_box_count": 0,
                "expected_item_count": 0,
                "elapsed_values": [],
            },
        )
        bucket["total"] = int(bucket["total"]) + 1
        bucket["usable_count"] = int(bucket["usable_count"]) + int(result.usable_region)
        bucket["strict_count"] = int(bucket["strict_count"]) + int(result.strict_region_count_matched)
        bucket["valid_box_count"] = int(bucket["valid_box_count"]) + result.valid_box_count
        bucket["expected_item_count"] = int(bucket["expected_item_count"]) + result.expected_item_count
        bucket["elapsed_values"] = [*bucket["elapsed_values"], result.elapsed_ms]
    compact_by_subject = {}
    for subject, raw in by_subject.items():
        total_count = int(raw["total"])
        subject_elapsed = list(raw["elapsed_values"])
        subject_region_denominator = sum(
            max(result.expected_item_count, result.region_count)
            for result in results
            if samples_by_id.get(result.sample_id, OCRSample(sample_id=result.sample_id, image_path="")).subject
            == subject
        )
        compact_by_subject[subject] = {
            "total": total_count,
            "usable_region_rate": _ratio(int(raw["usable_count"]), total_count),
            "strict_region_rate": _ratio(int(raw["strict_count"]), total_count),
            "valid_box_rate": _ratio(int(raw["valid_box_count"]), subject_region_denominator),
            "average_preprocess_ms": round(statistics.mean(subject_elapsed)) if subject_elapsed else 0,
        }
    return PhotoPreprocessEvaluationReport(
        total=total,
        failure_count=sum(1 for result in results if result.status == "error"),
        usable_region_rate=_ratio(sum(1 for result in results if result.usable_region), total),
        strict_region_rate=_ratio(
            sum(1 for result in results if result.strict_region_count_matched),
            total,
        ),
        valid_box_rate=_ratio(sum(result.valid_box_count for result in results), region_box_denominator),
        ordered_region_rate=_ratio(sum(1 for result in results if result.ordered_regions), total),
        quality_warning_rate=_ratio(sum(1 for result in results if result.quality_warnings), total),
        average_preprocess_ms=round(statistics.mean(elapsed_values)) if elapsed_values else 0,
        p95_preprocess_ms=_percentile(elapsed_values, 0.95),
        max_preprocess_ms=max(elapsed_values) if elapsed_values else 0,
        by_subject=compact_by_subject,
        results=results,
    )


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]

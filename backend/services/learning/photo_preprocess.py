from __future__ import annotations

import cv2
import numpy as np
from pydantic import BaseModel, Field


class ImageRegion(BaseModel):
    x: int
    y: int
    width: int
    height: int


class HomeworkPhotoAnalysis(BaseModel):
    original_width: int = 0
    original_height: int = 0
    processed_x: int = 0
    processed_y: int = 0
    processed_width: int = 0
    processed_height: int = 0
    processed_content: bytes = b""
    mime_type: str = "image/jpeg"
    question_regions: list[ImageRegion] = Field(default_factory=list)
    processed_question_regions: list[ImageRegion] = Field(default_factory=list)
    quality_warnings: list[str] = Field(default_factory=list)
    quality_message: str = ""
    source: str = "opencv_projection_v0.2"

    @property
    def region_count(self) -> int:
        return len(self.question_regions)


def analyze_homework_photo(content: bytes, *, filename: str = "upload.jpg") -> HomeworkPhotoAnalysis:
    image = _decode_image(content)
    if image is None:
        return HomeworkPhotoAnalysis(
            processed_content=b"",
            mime_type=_mime_type_for_filename(filename),
            quality_warnings=["photo_decode_failed", "retake_required"],
            quality_message="照片没有读取成功，请重新拍照后再提交。",
            source="opencv_decode_failed",
        )

    original_height, original_width = image.shape[:2]
    normalized = _normalize_orientation(image)
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    low, high = _percentiles(gray, 0.005, 0.995)
    quality_warnings = _quality_warnings(gray, width=original_width, height=original_height, low=low, high=high)
    ink = _ink_mask(gray, high=high)
    ink = _clean_ink_mask(ink)
    content_box = _content_box(ink, image_shape=gray.shape)

    if content_box is None:
        warnings = _append_unique(quality_warnings, ["no_homework_content", "retake_required"])
        return HomeworkPhotoAnalysis(
            original_width=original_width,
            original_height=original_height,
            processed_x=0,
            processed_y=0,
            processed_width=original_width,
            processed_height=original_height,
            processed_content=_encode_jpeg(_enhance_for_ocr(normalized)),
            quality_warnings=warnings,
            quality_message=_quality_message(warnings),
            source="opencv_no_content_v0.2",
        )

    crop_box = _expand_box(content_box, image_shape=gray.shape, padding=max(24, min(gray.shape) // 40))
    left, top, right, bottom = crop_box
    cropped = normalized[top:bottom, left:right]
    crop_ink = ink[top:bottom, left:right]
    question_regions = _detect_question_regions(ink, image_size=(original_width, original_height))
    processed_regions = _regions_to_processed(
        question_regions,
        crop_box=crop_box,
        original_size=(original_width, original_height),
        processed_size=(max(1, right - left), max(1, bottom - top)),
    )

    if not question_regions and "retake_required" not in quality_warnings:
        quality_warnings = _append_unique(quality_warnings, ["no_homework_content"])
    if not _has_meaningful_ink(crop_ink):
        quality_warnings = _append_unique(quality_warnings, ["no_homework_content"])

    return HomeworkPhotoAnalysis(
        original_width=original_width,
        original_height=original_height,
        processed_x=left,
        processed_y=top,
        processed_width=max(1, right - left),
        processed_height=max(1, bottom - top),
        processed_content=_encode_jpeg(_enhance_for_ocr(cropped)),
        question_regions=question_regions,
        processed_question_regions=processed_regions,
        quality_warnings=quality_warnings,
        quality_message=_quality_message(quality_warnings),
        source="opencv_document_projection_v0.2",
    )


def _decode_image(content: bytes):
    if not content:
        return None
    array = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None
    return image


def _normalize_orientation(image):
    return image


def _percentiles(gray, low_percent: float, high_percent: float) -> tuple[int, int]:
    histogram = np.bincount(gray.reshape(-1), minlength=256)
    total = int(histogram.sum())
    if total <= 0:
        return 0, 255
    cumulative = np.cumsum(histogram)
    low = int(np.searchsorted(cumulative, total * low_percent, side="left"))
    high = int(np.searchsorted(cumulative, total * high_percent, side="left"))
    return low, high


def _quality_warnings(gray, *, width: int, height: int, low: int, high: int) -> list[str]:
    warnings: list[str] = []
    if min(width, height) < 600:
        warnings.append("small_image")
    if high - low < 32:
        warnings.append("low_contrast")
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_score < 18.0 and min(width, height) >= 600:
        warnings.append("blurry_image")
    if _has_glare_or_overexposed_area(gray):
        warnings.append("glare_or_overexposed_area")
    return warnings


def _has_glare_or_overexposed_area(gray) -> bool:
    height, width = gray.shape[:2]
    total_area = max(1, height * width)
    bright_mask = np.where(gray >= 252, 255, 0).astype(np.uint8)
    bright_mask = cv2.morphologyEx(
        bright_mask,
        cv2.MORPH_OPEN,
        np.ones((9, 9), np.uint8),
        iterations=1,
    )
    component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(bright_mask, 8)
    if component_count <= 1:
        return False
    largest_area = max(int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, component_count))
    largest_ratio = largest_area / total_area
    return 0.006 <= largest_ratio <= 0.35


def _ink_mask(gray, *, high: int):
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        16,
    )
    threshold = max(0, min(235, high - 35))
    dark = np.where(gray < threshold, 255, 0).astype(np.uint8)
    return cv2.bitwise_or(adaptive, dark)


def _clean_ink_mask(mask):
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)


def _content_box(mask, *, image_shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    height, width = image_shape
    points = cv2.findNonZero(mask)
    if points is None:
        return None
    x, y, box_width, box_height = cv2.boundingRect(points)
    if box_width * box_height < width * height * 0.002:
        return None
    if np.count_nonzero(mask[y : y + box_height, x : x + box_width]) < width * height * 0.0008:
        return None
    return x, y, x + box_width, y + box_height


def _expand_box(
    box: tuple[int, int, int, int],
    *,
    image_shape: tuple[int, int],
    padding: int,
) -> tuple[int, int, int, int]:
    height, width = image_shape
    left, top, right, bottom = box
    return (
        max(0, left - padding),
        max(0, top - padding),
        min(width, right + padding),
        min(height, bottom + padding),
    )


def _detect_question_regions(mask, *, image_size: tuple[int, int]) -> list[ImageRegion]:
    width, height = image_size
    ranges = _column_ranges(mask)
    regions: list[ImageRegion] = []
    for left, right in ranges:
        column_mask = mask[:, left:right]
        for top, bottom in _row_groups(column_mask):
            region = _region_for_span(column_mask, left=left, top=top, bottom=bottom, image_size=image_size)
            if region is not None:
                regions.append(region)
    return sorted(_dedupe_regions(regions), key=lambda region: (region.y, region.x))


def _column_ranges(mask) -> list[tuple[int, int]]:
    height, width = mask.shape[:2]
    column_counts = np.count_nonzero(mask, axis=0)
    active_columns = np.flatnonzero(column_counts > max(4, height // 180))
    if active_columns.size == 0:
        return [(0, width)]

    min_x = int(active_columns[0])
    max_x = int(active_columns[-1]) + 1
    if max_x - min_x < width * 0.45:
        return [(min_x, max_x)]

    search_left = max(min_x, int(width * 0.32))
    search_right = min(max_x, int(width * 0.68))
    if search_right <= search_left:
        return [(min_x, max_x)]

    smoothed = cv2.blur(column_counts.astype(np.float32).reshape(1, -1), (1, 31)).reshape(-1)
    valley = int(search_left + np.argmin(smoothed[search_left:search_right]))
    local_peak = float(max(smoothed[min_x:search_left].max(initial=0), smoothed[search_right:max_x].max(initial=0)))
    valley_value = float(smoothed[valley])
    if local_peak <= 0 or valley_value > local_peak * 0.28:
        return [(min_x, max_x)]

    gap_left = valley
    while gap_left > min_x and smoothed[gap_left] < local_peak * 0.42:
        gap_left -= 1
    gap_right = valley
    while gap_right < max_x - 1 and smoothed[gap_right] < local_peak * 0.42:
        gap_right += 1
    if gap_right - gap_left < width * 0.05:
        return [(min_x, max_x)]

    ranges = [(min_x, gap_left), (gap_right, max_x)]
    return [(left, right) for left, right in ranges if right - left >= width * 0.12]


def _row_groups(mask) -> list[tuple[int, int]]:
    height, width = mask.shape[:2]
    row_counts = np.count_nonzero(mask, axis=1)
    active_rows = np.flatnonzero(row_counts >= max(8, width // 120))
    if active_rows.size == 0:
        return []
    line_groups: list[tuple[int, int]] = []
    start = int(active_rows[0])
    previous = int(active_rows[0])
    for raw_row in active_rows[1:]:
        row = int(raw_row)
        if row - previous <= 5:
            previous = row
            continue
        line_groups.append((start, previous))
        start = row
        previous = row
    line_groups.append((start, previous))

    groups: list[tuple[int, int]] = []
    start: int | None = None
    min_height = max(14, height // 65)
    previous_bottom: int | None = None
    merge_gap = max(48, min(74, height // 14))
    for top, bottom in line_groups:
        if start is None:
            start = top
            previous_bottom = bottom
            continue
        assert previous_bottom is not None
        if top - previous_bottom <= merge_gap:
            previous_bottom = bottom
            continue
        if previous_bottom - start >= min_height:
            groups.append((start, previous_bottom))
        start = top
        previous_bottom = bottom
    if start is not None and previous_bottom is not None and previous_bottom - start >= min_height:
        groups.append((start, previous_bottom))
    return groups


def _fill_small_gaps(active, max_gap: int):
    filled = active.copy()
    previous: int | None = None
    for index, value in enumerate(active):
        if not value:
            continue
        if previous is not None and 0 < index - previous - 1 <= max_gap:
            filled[previous + 1 : index] = True
        previous = index
    return filled


def _region_for_span(
    mask,
    *,
    left: int,
    top: int,
    bottom: int,
    image_size: tuple[int, int],
) -> ImageRegion | None:
    width, height = image_size
    block = mask[max(0, top) : min(mask.shape[0], bottom + 1), :]
    ys, xs = np.nonzero(block)
    if xs.size == 0:
        return None
    min_x = int(xs.min()) + left
    max_x = int(xs.max()) + left
    min_y = int(ys.min()) + max(0, top)
    max_y = int(ys.max()) + max(0, top)
    padding_x = max(12, width // 80)
    padding_y = max(16, height // 60)
    box_left = max(0, min_x - padding_x)
    box_top = max(0, min_y - padding_y)
    box_right = min(width, max_x + padding_x)
    box_bottom = min(height, max_y + padding_y)
    if box_right - box_left < width * 0.05 or box_bottom - box_top < height * 0.022:
        return None
    return _normalize_region(box_left, box_top, box_right, box_bottom, image_size=image_size)


def _dedupe_regions(regions: list[ImageRegion]) -> list[ImageRegion]:
    deduped: list[ImageRegion] = []
    for region in regions:
        if any(_region_iou(region, existing) > 0.72 for existing in deduped):
            continue
        deduped.append(region)
    return deduped


def _region_iou(first: ImageRegion, second: ImageRegion) -> float:
    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.x + first.width, second.x + second.width)
    bottom = min(first.y + first.height, second.y + second.height)
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    first_area = first.width * first.height
    second_area = second.width * second.height
    return intersection / max(1, first_area + second_area - intersection)


def _regions_to_processed(
    regions: list[ImageRegion],
    *,
    crop_box: tuple[int, int, int, int],
    original_size: tuple[int, int],
    processed_size: tuple[int, int],
) -> list[ImageRegion]:
    original_width, original_height = original_size
    processed_width, processed_height = processed_size
    left, top, _, _ = crop_box
    mapped: list[ImageRegion] = []
    for region in regions:
        abs_left = region.x / 1000.0 * original_width
        abs_top = region.y / 1000.0 * original_height
        abs_width = region.width / 1000.0 * original_width
        abs_height = region.height / 1000.0 * original_height
        mapped.append(
            _normalize_region(
                round(abs_left - left),
                round(abs_top - top),
                round(abs_left - left + abs_width),
                round(abs_top - top + abs_height),
                image_size=(processed_width, processed_height),
            )
        )
    return mapped


def _normalize_region(
    left: int,
    top: int,
    right: int,
    bottom: int,
    *,
    image_size: tuple[int, int],
) -> ImageRegion:
    width, height = image_size
    clamped_left = max(0, min(width - 1, left))
    clamped_top = max(0, min(height - 1, top))
    clamped_right = max(clamped_left + 1, min(width, right))
    clamped_bottom = max(clamped_top + 1, min(height, bottom))
    x = max(0, min(999, round(clamped_left / width * 1000)))
    y = max(0, min(999, round(clamped_top / height * 1000)))
    region_width = max(1, min(1000 - x, round((clamped_right - clamped_left) / width * 1000)))
    region_height = max(1, min(1000 - y, round((clamped_bottom - clamped_top) / height * 1000)))
    return ImageRegion(x=x, y=y, width=region_width, height=region_height)


def _enhance_for_ocr(image):
    if image.size == 0:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=23, sigmaY=23)
    normalized = cv2.divide(gray, background, scale=255)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def _encode_jpeg(image) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 76])
    return encoded.tobytes() if ok else b""


def _has_meaningful_ink(mask) -> bool:
    if mask.size == 0:
        return False
    return np.count_nonzero(mask) >= max(20, int(mask.size * 0.0008))


def _append_unique(values: list[str], additions: list[str]) -> list[str]:
    result = list(values)
    for value in additions:
        if value not in result:
            result.append(value)
    return result


def _quality_message(warnings: list[str]) -> str:
    values = set(warnings)
    if "photo_decode_failed" in values:
        return "照片没有读取成功，请重新拍照后再提交。"
    if "no_homework_content" in values:
        return "没有清楚识别到题目内容，请先核对识别内容；如果题目或答案缺失，请把作业放满画面、对焦后重拍。"
    if {"small_image", "low_contrast", "blurry_image", "glare_or_overexposed_area"} & values:
        return "照片可能不够清楚，请先核对识别内容；如果题目或答案不完整，建议靠近题目、避开反光并对焦重拍。"
    return ""


def _mime_type_for_filename(filename: str) -> str:
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix in {"jpg", "jpeg"}:
        return "image/jpeg"
    if suffix == "webp":
        return "image/webp"
    return "image/png"

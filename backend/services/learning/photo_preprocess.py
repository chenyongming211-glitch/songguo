from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np
from pydantic import BaseModel, Field


A4_ASPECT_RATIO = math.sqrt(2.0)
PORTRAIT_TO_LANDSCAPE_ROTATION_PENALTY = 16.0


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
    preview_content: bytes = b""
    mime_type: str = "image/jpeg"
    question_regions: list[ImageRegion] = Field(default_factory=list)
    processed_question_regions: list[ImageRegion] = Field(default_factory=list)
    quality_warnings: list[str] = Field(default_factory=list)
    quality_message: str = ""
    source: str = "opencv_projection_v0.2"

    @property
    def region_count(self) -> int:
        return len(self.question_regions)


@dataclass(frozen=True)
class _PhotoCandidate:
    image: np.ndarray
    source: str
    score: float
    rotation_penalty: float = 0.0


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
    working, source = _best_document_candidate(image)
    working_height, working_width = working.shape[:2]
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    low, high = _percentiles(gray, 0.005, 0.995)
    quality_warnings = _quality_warnings(gray, width=working_width, height=working_height, low=low, high=high)
    if source.startswith(("opencv_document_perspective_v0.3", "opencv_document_rotation_v0.3")):
        quality_warnings = _append_unique(quality_warnings, ["photo_not_level"])
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
            processed_width=working_width,
            processed_height=working_height,
            processed_content=_encode_jpeg(_enhance_for_ocr(working)),
            preview_content=_encode_jpeg(working),
            quality_warnings=warnings,
            quality_message=_quality_message(warnings),
            source="opencv_no_content_v0.2",
        )

    if source == "opencv_document_perspective_v0.3":
        crop_box = (0, 0, working_width, working_height)
    else:
        crop_box = _expand_box(content_box, image_shape=gray.shape, padding=max(24, min(gray.shape) // 40))
    left, top, right, bottom = crop_box
    cropped = working[top:bottom, left:right]
    crop_ink = ink[top:bottom, left:right]
    processed_width = max(1, right - left)
    processed_height = max(1, bottom - top)
    processed_regions = _detect_question_regions(
        crop_ink,
        image_size=(processed_width, processed_height),
    )

    if not processed_regions and "retake_required" not in quality_warnings:
        quality_warnings = _append_unique(quality_warnings, ["no_homework_content"])
    if not _has_meaningful_ink(crop_ink):
        quality_warnings = _append_unique(quality_warnings, ["no_homework_content"])

    return HomeworkPhotoAnalysis(
        original_width=original_width,
        original_height=original_height,
        processed_x=left,
        processed_y=top,
        processed_width=processed_width,
        processed_height=processed_height,
        processed_content=_encode_jpeg(_enhance_for_ocr(cropped)),
        preview_content=_encode_jpeg(cropped),
        question_regions=processed_regions,
        processed_question_regions=processed_regions,
        quality_warnings=quality_warnings,
        quality_message=_quality_message(quality_warnings),
        source=source,
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


def _best_document_candidate(image):
    candidates = []
    original_height, original_width = image.shape[:2]
    for label, candidate_image, penalty in _orientation_candidates(image):
        working, source = _document_view(candidate_image)
        source_label = source if label == "original" else f"{source}_{label}"
        candidate_height, candidate_width = working.shape[:2]
        orientation_penalty = penalty
        if label != "original" and original_height >= original_width and candidate_width > candidate_height:
            orientation_penalty += PORTRAIT_TO_LANDSCAPE_ROTATION_PENALTY
        candidates.append(
            _PhotoCandidate(
                image=working,
                source=source_label,
                score=_document_candidate_score(working, source=source) - orientation_penalty,
                rotation_penalty=orientation_penalty,
            )
        )
    best = max(candidates, key=lambda candidate: candidate.score)
    return best.image, best.source


def _orientation_candidates(image):
    return [
        ("original", image, 0.0),
        ("orientation_90_ccw", cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), 0.8),
        ("orientation_180", cv2.rotate(image, cv2.ROTATE_180), 28.0),
        ("orientation_90_cw", cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), 0.8),
    ]


def _document_view(image):
    warped = _warp_document_if_possible(image)
    if warped is None:
        rotated = _rotate_document_if_needed(image)
        if rotated is not None:
            return rotated, "opencv_document_rotation_v0.3"
        return image, "opencv_document_projection_v0.2"
    return warped, "opencv_document_perspective_v0.3"


def _document_candidate_score(image, *, source: str) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _low, high = _percentiles(gray, 0.005, 0.995)
    ink = _clean_ink_mask(_ink_mask(gray, high=high))
    content_box = _content_box(ink, image_shape=gray.shape)
    if content_box is None:
        return -1000.0

    height, width = gray.shape[:2]
    if source == "opencv_document_perspective_v0.3":
        crop_box = (0, 0, width, height)
    else:
        crop_box = _expand_box(content_box, image_shape=gray.shape, padding=max(24, min(gray.shape) // 40))
    left, top, right, bottom = crop_box
    crop_width = max(1, right - left)
    crop_height = max(1, bottom - top)
    crop_ink = ink[top:bottom, left:right]
    regions = _detect_question_regions(crop_ink, image_size=(crop_width, crop_height))
    ink_ratio = np.count_nonzero(crop_ink) / max(1, crop_ink.size)
    content_ratio = (crop_width * crop_height) / max(1, width * height)
    aspect = crop_height / max(1, crop_width)

    score = 0.0
    score += min(len(regions), 8) * 10.0
    score += min(ink_ratio * 900.0, 18.0)
    score += min(content_ratio * 10.0, 8.0)
    if 0.55 <= aspect <= 1.85:
        score += 8.0
    if crop_height >= crop_width:
        score += 3.0
    if source == "opencv_document_perspective_v0.3":
        score += 5.0
    score += _upright_line_orientation_score(gray[top:bottom, left:right]) * 12.0
    return score


def _upright_line_orientation_score(gray_crop) -> float:
    height, width = gray_crop.shape[:2]
    if min(width, height) < 160:
        return 0.0
    blurred = cv2.GaussianBlur(gray_crop, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(40, min(width, height) // 12),
        minLineLength=max(40, min(width, height) // 10),
        maxLineGap=20,
    )
    if lines is None:
        return 0.0

    horizontal = 0.0
    vertical = 0.0
    diagonal = 0.0
    min_length = max(30.0, min(width, height) * 0.05)
    for x1, y1, x2, y2 in lines[:, 0, :]:
        dx = int(x2) - int(x1)
        dy = int(y2) - int(y1)
        length = math.hypot(dx, dy)
        if length < min_length:
            continue
        angle = abs(math.degrees(math.atan2(dy, dx)))
        angle = min(angle, 180.0 - angle)
        if angle <= 18.0:
            horizontal += length
        elif angle >= 72.0:
            vertical += length
        else:
            diagonal += length * 0.25

    total = horizontal + vertical + diagonal
    if total <= 0.0:
        return 0.0
    return max(-1.0, min(1.0, (horizontal - vertical) / total))


def _warp_document_if_possible(image):
    quad = _find_document_quad(image)
    if quad is None:
        return None
    warped = _four_point_warp(image, quad)
    if warped is None or warped.size == 0:
        return None
    height, width = warped.shape[:2]
    if min(width, height) < 240:
        return None
    residual_skew = _estimate_document_skew_angle(warped)
    if residual_skew is not None and abs(residual_skew) > 6.0:
        return None
    return warped


def _rotate_document_if_needed(image):
    angle = _estimate_document_skew_angle(image)
    if angle is None:
        return None
    if abs(angle) < 1.1 or abs(angle) > 25.0:
        return None
    return _rotate_bound(image, -angle)


def _estimate_document_skew_angle(image) -> float | None:
    height, width = image.shape[:2]
    if min(width, height) < 240:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 130)
    threshold = max(80, min(width, height) // 6)
    min_line_length = max(120, min(width, height) // 5)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=25,
    )
    if lines is None:
        return None

    angles: list[float] = []
    weights: list[float] = []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        dx = int(x2) - int(x1)
        dy = int(y2) - int(y1)
        length = math.hypot(dx, dy)
        if length < min(width, height) * 0.12:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        while angle <= -90.0:
            angle += 180.0
        while angle > 90.0:
            angle -= 180.0
        if abs(angle) <= 30.0:
            normalized = angle
        elif abs(abs(angle) - 90.0) <= 30.0:
            normalized = angle - (90.0 if angle > 0 else -90.0)
        else:
            continue
        angles.append(float(normalized))
        weights.append(float(length))

    if len(angles) < 4:
        return None
    return _weighted_median(angles, weights)


def _weighted_median(values: list[float], weights: list[float]) -> float:
    pairs = sorted(zip(values, weights), key=lambda pair: pair[0])
    total = sum(weight for _value, weight in pairs)
    midpoint = total / 2.0
    running = 0.0
    for value, weight in pairs:
        running += weight
        if running >= midpoint:
            return value
    return pairs[-1][0]


def _rotate_bound(image, angle: float):
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    matrix[0, 2] += (new_width / 2.0) - center[0]
    matrix[1, 2] += (new_height / 2.0) - center[1]
    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _find_document_quad(image):
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return None
    bright_quad = _find_document_quad_from_bright_mask(image)
    if bright_quad is not None:
        return bright_quad
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 130)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2)
    contours, _hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = float(width * height)
    best_quad = None
    best_area = 0.0
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        area = float(cv2.contourArea(contour))
        if area < image_area * 0.18 or area > image_area * 0.98:
            continue
        perimeter = cv2.arcLength(contour, True)
        for epsilon_ratio in (0.015, 0.02, 0.03, 0.045):
            approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            quad = approx.reshape(4, 2).astype("float32")
            if not _valid_document_quad(quad, image_area=image_area):
                continue
            if area > best_area:
                best_quad = quad
                best_area = area
            break
    return best_quad


def _find_document_quad_from_bright_mask(image):
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu_threshold, _mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    base_threshold = max(120, min(235, int(otsu_threshold)))
    thresholds = _unique_thresholds(
        [
            max(base_threshold, min(235, int(otsu_threshold) + 25)),
            max(base_threshold, 180),
            max(base_threshold, 190),
            base_threshold,
        ]
    )
    for threshold in thresholds:
        mask = np.where(blurred >= threshold, 255, 0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8), iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8), iterations=1)
        quad = _find_document_quad_from_mask(mask, image_area=float(width * height))
        if quad is not None:
            return quad
    return None


def _unique_thresholds(values: list[int]) -> list[int]:
    ordered: list[int] = []
    for value in values:
        threshold = max(0, min(255, int(value)))
        if threshold not in ordered:
            ordered.append(threshold)
    return ordered


def _find_document_quad_from_mask(mask, *, image_area: float):
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        area = float(cv2.contourArea(contour))
        if area < image_area * 0.18 or area > image_area * 0.985:
            continue
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        for epsilon_ratio in (0.015, 0.02, 0.03, 0.045, 0.06, 0.08):
            approx = cv2.approxPolyDP(hull, epsilon_ratio * perimeter, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            quad = approx.reshape(4, 2).astype("float32")
            if _valid_document_quad(quad, image_area=image_area):
                return quad

        rect = cv2.minAreaRect(contour)
        quad = cv2.boxPoints(rect).astype("float32")
        if _valid_document_quad(quad, image_area=image_area):
            return quad
    return None


def _valid_document_quad(quad, *, image_area: float) -> bool:
    area = abs(float(cv2.contourArea(quad.astype("float32"))))
    if area < image_area * 0.18:
        return False
    ordered = _order_quad_points(quad)
    top_width = np.linalg.norm(ordered[1] - ordered[0])
    bottom_width = np.linalg.norm(ordered[2] - ordered[3])
    left_height = np.linalg.norm(ordered[3] - ordered[0])
    right_height = np.linalg.norm(ordered[2] - ordered[1])
    min_side = min(top_width, bottom_width, left_height, right_height)
    if min_side < 120:
        return False
    aspect = max(top_width, bottom_width) / max(1.0, max(left_height, right_height))
    return 0.35 <= aspect <= 1.75


def _order_quad_points(points):
    rect = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def _four_point_warp(image, points):
    rect = _order_quad_points(points)
    top_left, top_right, bottom_right, bottom_left = rect
    width_a = np.linalg.norm(bottom_right - bottom_left)
    width_b = np.linalg.norm(top_right - top_left)
    height_a = np.linalg.norm(top_right - bottom_right)
    height_b = np.linalg.norm(top_left - bottom_left)
    measured_width = max(1, int(round(max(width_a, width_b))))
    measured_height = max(1, int(round(max(height_a, height_b))))
    max_width, max_height = _a4_target_size(measured_width, measured_height)
    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def _a4_target_size(width: int, height: int) -> tuple[int, int]:
    if height >= width:
        return max(1, int(round(height / A4_ASPECT_RATIO))), height
    return width, max(1, int(round(width / A4_ASPECT_RATIO)))


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
    if width < 640:
        return [(min_x, max_x)]
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
    min_column_width = max(220, int(width * 0.12))
    filtered = [(left, right) for left, right in ranges if right - left >= min_column_width]
    if not filtered:
        return [(min_x, max_x)]
    return filtered


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
    merge_gap = max(60, min(74, height // 14))
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
    if "photo_not_level" in values:
        return "照片已自动扶正；如果题目框仍有偏移，请将手机放平、让作业纸四边尽量完整后重拍。"
    return ""


def _mime_type_for_filename(filename: str) -> str:
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix in {"jpg", "jpeg"}:
        return "image/jpeg"
    if suffix == "webp":
        return "image/webp"
    return "image/png"

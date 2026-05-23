from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from songguo.backend.services.learning.photo_preprocess import analyze_homework_photo


def _blank(width: int, height: int, color: tuple[int, int, int] = (250, 250, 246)):
    return np.full((height, width, 3), color, dtype=np.uint8)


def _jpeg_bytes(image) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    assert ok
    return encoded.tobytes()


def test_analyze_homework_photo_crops_blank_margins_and_detects_question_regions() -> None:
    image = _blank(1400, 1000)
    # Three question-like groups separated by vertical whitespace.
    for y in (160, 430, 700):
        cv2.rectangle(image, (170, y), (1230, y + 24), (28, 28, 28), thickness=-1)
        cv2.rectangle(image, (190, y + 62), (760, y + 84), (38, 38, 38), thickness=-1)

    result = analyze_homework_photo(_jpeg_bytes(image), filename="homework.jpg")

    assert result.original_width == 1400
    assert result.original_height == 1000
    assert result.source.startswith("opencv_")
    assert result.processed_width < result.original_width
    assert result.processed_height < result.original_height
    assert result.region_count == 3
    assert [region.y for region in result.question_regions] == sorted(
        region.y for region in result.question_regions
    )
    assert all(0 <= region.x <= 1000 for region in result.question_regions)
    assert result.quality_warnings == []


def test_analyze_homework_photo_reports_low_quality_small_low_contrast_image() -> None:
    image = _blank(420, 320, (240, 240, 240))
    cv2.rectangle(image, (80, 140), (330, 148), (226, 226, 226), thickness=-1)

    result = analyze_homework_photo(_jpeg_bytes(image), filename="small.jpg")

    assert "small_image" in result.quality_warnings
    assert "low_contrast" in result.quality_warnings
    assert result.quality_message
    assert "核对识别内容" in result.quality_message
    assert "重拍" in result.quality_message
    assert result.region_count == 0


def test_analyze_homework_photo_reports_glare_or_overexposed_area() -> None:
    image = _blank(1400, 1000)
    cv2.putText(image, "48 / 6 = ?", (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 4)
    cv2.putText(image, "Answer: 8", (160, 340), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (40, 40, 40), 3)
    cv2.ellipse(image, (780, 280), (260, 120), -12, 0, 360, (255, 255, 255), thickness=-1)

    result = analyze_homework_photo(_jpeg_bytes(image), filename="glare.jpg")

    assert "glare_or_overexposed_area" in result.quality_warnings
    assert result.quality_message
    assert "核对识别内容" in result.quality_message
    assert "重拍" in result.quality_message


def test_analyze_homework_photo_reports_generated_glare_variant(tmp_path) -> None:
    from songguo.backend.evaluation.ocr_stress_samples import (
        OCRStressCase,
        OCRStressItem,
        OCRStressVariant,
        generate_ocr_stress_samples,
    )

    manifest = generate_ocr_stress_samples(
        output_dir=tmp_path,
        cases=[
            OCRStressCase(
                sample_id="math_glare",
                subject="math",
                lines=["48 ÷ 6 = ?", "孩子答案：8"],
                expected_items=[OCRStressItem(question_text="48 ÷ 6 = ?", child_answer="8")],
            )
        ],
        variants=[
            OCRStressVariant(
                name="glare_reflection",
                rotate_degrees=-2,
                shadow=True,
                glare=True,
                notebook_lines=True,
                jpeg_quality=74,
            )
        ],
    )
    image_path = tmp_path / "images" / "math_glare_glare_reflection.jpg"
    assert manifest.exists()
    assert image_path.exists()

    result = analyze_homework_photo(image_path.read_bytes(), filename=image_path.name)

    assert "glare_or_overexposed_area" in result.quality_warnings


def test_analyze_homework_photo_reports_retake_for_unreadable_upload() -> None:
    result = analyze_homework_photo(b"not an image", filename="broken.jpg")

    assert result.source == "opencv_decode_failed"
    assert result.processed_content == b""
    assert result.quality_warnings == ["photo_decode_failed", "retake_required"]
    assert "重新拍照" in result.quality_message
    assert result.region_count == 0


def test_analyze_homework_photo_splits_two_column_numbered_questions() -> None:
    image = _blank(1600, 1000)
    for x, y, label in [
        (100, 110, "1. 48 / 6 = ?"),
        (100, 310, "2. 235 - 80 = ?"),
        (870, 110, "3. 560 - 80 = ?"),
    ]:
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.45, (20, 20, 20), 4)
        cv2.putText(image, "Answer: 8", (x + 42, y + 76), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (35, 35, 35), 3)

    result = analyze_homework_photo(_jpeg_bytes(image), filename="two-column.jpg")

    assert result.source.startswith("opencv_")
    assert result.region_count >= 3
    assert len(result.processed_question_regions) == result.region_count
    assert [region.y for region in result.question_regions] == sorted(
        region.y for region in result.question_regions
    )


def test_analyze_homework_photo_perspective_corrects_skewed_page() -> None:
    page = _blank(900, 1200)
    cv2.rectangle(page, (70, 80), (830, 1120), (230, 230, 226), thickness=3)
    for index, y in enumerate((230, 410, 590), start=1):
        cv2.putText(page, f"{index}. 21 x 50 = ?", (120, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (25, 25, 25), 3)
        cv2.putText(page, "Answer: 105", (160, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (35, 35, 35), 2)

    canvas = np.full((1400, 1200, 3), (128, 128, 122), dtype=np.uint8)
    source = np.float32([[0, 0], [899, 0], [899, 1199], [0, 1199]])
    target = np.float32([[170, 120], [1010, 70], [1090, 1300], [90, 1240]])
    matrix = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(page, matrix, (1200, 1400))
    mask = cv2.warpPerspective(np.full((1200, 900), 255, dtype=np.uint8), matrix, (1200, 1400))
    canvas[mask > 0] = warped[mask > 0]

    result = analyze_homework_photo(_jpeg_bytes(canvas), filename="skewed-homework.jpg")

    assert result.source == "opencv_document_perspective_v0.3"
    assert "photo_not_level" in result.quality_warnings
    assert "放平" in result.quality_message
    assert result.processed_content
    corrected = cv2.imdecode(np.frombuffer(result.processed_content, np.uint8), cv2.IMREAD_COLOR)
    assert corrected is not None
    corrected_height, corrected_width = corrected.shape[:2]
    assert corrected_height > corrected_width
    assert result.region_count >= 3
    assert len(result.processed_question_regions) == result.region_count


def test_analyze_homework_photo_scanner_mode_preserves_a4_page_ratio() -> None:
    page = _blank(900, 1273)
    cv2.rectangle(page, (55, 70), (845, 1203), (230, 230, 226), thickness=3)
    for index, y in enumerate((240, 430, 620), start=1):
        cv2.putText(page, f"{index}. 21 x 50 = ?", (120, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (25, 25, 25), 3)
        cv2.putText(page, "Answer: 105", (160, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (35, 35, 35), 2)

    canvas = np.full((1500, 1200, 3), (128, 128, 122), dtype=np.uint8)
    source = np.float32([[0, 0], [899, 0], [899, 1272], [0, 1272]])
    target = np.float32([[180, 120], [1000, 60], [1090, 1400], [80, 1330]])
    matrix = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(page, matrix, (1200, 1500))
    mask = cv2.warpPerspective(np.full((1273, 900), 255, dtype=np.uint8), matrix, (1200, 1500))
    canvas[mask > 0] = warped[mask > 0]

    result = analyze_homework_photo(_jpeg_bytes(canvas), filename="a4-scanner.jpg")

    assert result.source == "opencv_document_perspective_v0.3"
    preview = cv2.imdecode(np.frombuffer(result.preview_content, np.uint8), cv2.IMREAD_COLOR)
    assert preview is not None
    height, width = preview.shape[:2]
    assert abs((height / width) - np.sqrt(2)) < 0.03
    assert result.processed_x == 0
    assert result.processed_y == 0


def test_analyze_homework_photo_keeps_natural_deskewed_preview_separate_from_ocr_image() -> None:
    page = _blank(900, 1200, (218, 218, 212))
    cv2.rectangle(page, (60, 70), (840, 1130), (205, 205, 198), thickness=3)
    for index, y in enumerate((230, 410, 590), start=1):
        cv2.putText(page, f"{index}. 21 x 50 = ?", (120, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (25, 25, 25), 3)
        cv2.putText(page, "Answer: 105", (160, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (35, 35, 35), 2)

    canvas = np.full((1400, 1200, 3), (128, 128, 122), dtype=np.uint8)
    source = np.float32([[0, 0], [899, 0], [899, 1199], [0, 1199]])
    target = np.float32([[170, 120], [1010, 70], [1090, 1300], [90, 1240]])
    matrix = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(page, matrix, (1200, 1400))
    mask = cv2.warpPerspective(np.full((1200, 900), 255, dtype=np.uint8), matrix, (1200, 1400))
    canvas[mask > 0] = warped[mask > 0]

    result = analyze_homework_photo(_jpeg_bytes(canvas), filename="deskew-preview.jpg")

    preview_content = getattr(result, "preview_content", b"")
    assert preview_content
    assert preview_content != result.processed_content

    preview = cv2.imdecode(np.frombuffer(preview_content, np.uint8), cv2.IMREAD_COLOR)
    ocr_image = cv2.imdecode(np.frombuffer(result.processed_content, np.uint8), cv2.IMREAD_COLOR)
    assert preview is not None
    assert ocr_image is not None
    assert preview.shape[:2] == ocr_image.shape[:2]
    assert float(np.mean(preview)) < float(np.mean(ocr_image))


def test_analyze_homework_photo_rotates_real_partial_corner_page() -> None:
    image_path = (
        Path(__file__).resolve().parents[3]
        / "zhaopian"
        / "038c4944757951e6c22694eb7f9c015d.jpg"
    )

    result = analyze_homework_photo(image_path.read_bytes(), filename=image_path.name)

    assert result.source in {
        "opencv_document_rotation_v0.3",
        "opencv_document_perspective_v0.3",
    }
    assert "photo_not_level" in result.quality_warnings
    assert result.processed_content


def test_analyze_homework_photo_splits_generated_multi_item_page(tmp_path) -> None:
    from songguo.backend.evaluation.ocr_stress_samples import generate_ocr_stress_samples

    manifest = generate_ocr_stress_samples(output_dir=tmp_path)
    image_path = tmp_path / "images" / "math_multi_item_page_clean.jpg"
    assert manifest.exists()
    assert image_path.exists()

    result = analyze_homework_photo(image_path.read_bytes(), filename=image_path.name)

    assert result.region_count >= 3
    assert [region.y for region in result.question_regions] == sorted(
        region.y for region in result.question_regions
    )

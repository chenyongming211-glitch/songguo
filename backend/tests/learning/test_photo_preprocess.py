from __future__ import annotations

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

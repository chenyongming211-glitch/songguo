from __future__ import annotations

import json

from PIL import Image

from songguo.backend.evaluation.ocr_stress_samples import (
    OCRStressCase,
    OCRStressItem,
    OCRStressVariant,
    _wrap_text_to_width,
    default_stress_cases,
    default_stress_variants,
    generate_ocr_stress_samples,
)


def test_generate_ocr_stress_samples_writes_images_and_manifest(tmp_path) -> None:
    case = OCRStressCase(
        sample_id="english_tense",
        subject="english",
        grade=4,
        lines=[
            "Choose the correct tense:",
            "He ____ to school yesterday.",
            "Child answer: go",
        ],
        expected_items=[
            OCRStressItem(
                question_text="Choose the correct tense: He ____ to school yesterday.",
                child_answer="go",
            )
        ],
    )
    variant = OCRStressVariant(name="tilted", rotate_degrees=-5, jpeg_quality=72)

    manifest_path = generate_ocr_stress_samples(
        output_dir=tmp_path,
        cases=[case],
        variants=[variant],
    )

    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["sample_id"] == "english_tense_tilted"
    assert rows[0]["image_path"] == "images/english_tense_tilted.jpg"
    assert rows[0]["subject"] == "english"
    assert rows[0]["grade"] == 4
    assert rows[0]["tags"] == ["synthetic", "tilted", "english"]
    assert rows[0]["expected_items"] == [
        {
            "question_text": "Choose the correct tense: He ____ to school yesterday.",
            "child_answer": "go",
        }
    ]
    assert (tmp_path / rows[0]["image_path"]).exists()


def test_generate_ocr_stress_samples_can_limit_output(tmp_path) -> None:
    manifest_path = generate_ocr_stress_samples(output_dir=tmp_path, limit=2)

    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    assert all((tmp_path / row["image_path"]).exists() for row in rows)


def test_default_stress_suite_includes_multi_item_and_phone_like_variants() -> None:
    cases = default_stress_cases()
    variants = default_stress_variants()

    multi_item = next(case for case in cases if case.sample_id == "math_multi_item_page")
    assert len(multi_item.expected_items) == 3
    assert any(variant.two_columns for variant in variants)
    assert any(variant.crop_ratio > 0 for variant in variants)
    assert any(variant.handwriting_like for variant in variants)
    assert any(variant.blur_radius >= 2.0 for variant in variants)
    assert any(variant.glare for variant in variants)


def test_generate_ocr_stress_samples_writes_variant_tags_for_phone_like_layouts(tmp_path) -> None:
    manifest_path = generate_ocr_stress_samples(output_dir=tmp_path)

    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert any(row["sample_id"] == "math_multi_item_page_two_column" for row in rows)
    two_column = next(row for row in rows if row["sample_id"] == "math_multi_item_page_two_column")
    assert len(two_column["expected_items"]) == 3
    assert "two_column" in two_column["tags"]
    assert any("cropped_edge" in row["tags"] for row in rows)
    assert any("handwriting_like" in row["tags"] for row in rows)
    assert any("glare_reflection" in row["tags"] for row in rows)


def test_generate_ocr_stress_samples_keeps_many_item_stacked_page_visible(tmp_path) -> None:
    case = _ten_item_math_case()

    manifest_path = generate_ocr_stress_samples(
        output_dir=tmp_path,
        cases=[case],
        variants=[OCRStressVariant(name="clean", jpeg_quality=92)],
    )

    row = json.loads(manifest_path.read_text(encoding="utf-8").strip())
    image = Image.open(tmp_path / row["image_path"])
    assert image.height >= 2100


def test_generate_ocr_stress_samples_balances_many_item_two_column_page(tmp_path) -> None:
    case = _ten_item_math_case()

    manifest_path = generate_ocr_stress_samples(
        output_dir=tmp_path,
        cases=[case],
        variants=[OCRStressVariant(name="two_column", two_columns=True, jpeg_quality=92)],
    )

    row = json.loads(manifest_path.read_text(encoding="utf-8").strip())
    image = Image.open(tmp_path / row["image_path"]).convert("L")
    width, height = image.size
    left_lower_ink = _dark_pixel_count(image.crop((0, height // 2, width // 2, height)))
    right_lower_ink = _dark_pixel_count(image.crop((width // 2, height // 2, width, height)))

    assert left_lower_ink > 300
    assert right_lower_ink > 300


def test_wrap_text_to_width_keeps_long_chinese_line_inside_column() -> None:
    class FakeDraw:
        def textlength(self, value: str, *, font=None) -> int:
            return len(value) * 20

    wrapped = _wrap_text_to_width(
        FakeDraw(),
        "3. 甲仓库有560袋米，乙仓库有420袋，甲运给乙80袋后，甲比乙少还是多多少袋？",
        font=None,
        max_width=300,
    )

    assert len(wrapped) > 1
    assert all(len(line) * 20 <= 300 for line in wrapped)


def _ten_item_math_case() -> OCRStressCase:
    expected = [
        ("48 ÷ 6 = ?", "8"),
        ("72 ÷ 9 = ?", "8"),
        ("56 ÷ 7 = ?", "8"),
        ("9 × 6 = ?", "54"),
        ("36 ÷ 5 = ?", "7余1"),
        ("120 + 85 = ?", "205"),
        ("300 - 128 = ?", "172"),
        ("25 × 4 = ?", "100"),
        ("81 ÷ 9 = ?", "9"),
        ("64 ÷ 8 = ?", "8"),
    ]
    lines: list[str] = []
    for index, (question, answer) in enumerate(expected, start=1):
        lines.append(f"{index}. {question}")
        lines.append(f"孩子答案：{answer}")
        if index < len(expected):
            lines.append("")
    return OCRStressCase(
        sample_id="math_10_item_page",
        subject="math",
        grade=4,
        lines=lines,
        expected_items=[
            OCRStressItem(question_text=question, child_answer=answer)
            for question, answer in expected
        ],
    )


def _dark_pixel_count(image: Image.Image) -> int:
    return sum(1 for value in image.getdata() if value < 90)

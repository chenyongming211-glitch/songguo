from __future__ import annotations

import json
import random
from pathlib import Path

from pydantic import BaseModel, Field


class OCRStressItem(BaseModel):
    question_text: str
    child_answer: str = ""


class OCRStressCase(BaseModel):
    sample_id: str
    subject: str
    grade: int = 3
    lines: list[str]
    expected_items: list[OCRStressItem] = Field(default_factory=list)


class OCRStressVariant(BaseModel):
    name: str
    rotate_degrees: float = 0.0
    blur_radius: float = 0.0
    contrast: float = 1.0
    brightness: float = 1.0
    shadow: bool = False
    glare: bool = False
    noise: bool = False
    notebook_lines: bool = False
    two_columns: bool = False
    crop_ratio: float = 0.0
    handwriting_like: bool = False
    jpeg_quality: int = 86


def default_stress_cases() -> list[OCRStressCase]:
    return [
        OCRStressCase(
            sample_id="english_wrong_tense",
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
        ),
        OCRStressCase(
            sample_id="chinese_sentence",
            subject="chinese",
            grade=4,
            lines=[
                "用“因为……所以……”造句。",
                "孩子答案：因为下雨，所以我带伞。",
            ],
            expected_items=[
                OCRStressItem(
                    question_text="用“因为……所以……”造句。",
                    child_answer="因为下雨，所以我带伞。",
                )
            ],
        ),
        OCRStressCase(
            sample_id="math_division",
            subject="math",
            grade=4,
            lines=[
                "48 ÷ 6 = ?",
                "孩子答案：8",
            ],
            expected_items=[
                OCRStressItem(
                    question_text="48 ÷ 6 = ?",
                    child_answer="8",
                )
            ],
        ),
        OCRStressCase(
            sample_id="math_multi_item_page",
            subject="math",
            grade=4,
            lines=[
                "1. 48 ÷ 6 = ?",
                "孩子答案：8",
                "",
                "2. 一根彩带2米35厘米，剪去80厘米，还剩多少厘米？",
                "孩子答案：155厘米",
                "",
                "3. 甲仓库有560袋米，乙仓库有420袋，甲运给乙80袋后，甲比乙少还是多多少袋？",
                "孩子答案：甲多20袋",
            ],
            expected_items=[
                OCRStressItem(question_text="48 ÷ 6 = ?", child_answer="8"),
                OCRStressItem(
                    question_text="一根彩带2米35厘米，剪去80厘米，还剩多少厘米？",
                    child_answer="155厘米",
                ),
                OCRStressItem(
                    question_text="甲仓库有560袋米，乙仓库有420袋，甲运给乙80袋后，甲比乙少还是多多少袋？",
                    child_answer="甲多20袋",
                ),
            ],
        ),
    ]


def default_stress_variants() -> list[OCRStressVariant]:
    return [
        OCRStressVariant(name="clean", jpeg_quality=92),
        OCRStressVariant(
            name="tilted_shadow",
            rotate_degrees=-6,
            shadow=True,
            notebook_lines=True,
            jpeg_quality=76,
        ),
        OCRStressVariant(
            name="blur_low_contrast",
            rotate_degrees=3,
            blur_radius=1.1,
            contrast=0.72,
            brightness=0.92,
            noise=True,
            notebook_lines=True,
            jpeg_quality=64,
        ),
        OCRStressVariant(
            name="out_of_focus",
            rotate_degrees=2,
            blur_radius=2.4,
            contrast=0.86,
            brightness=0.98,
            noise=True,
            notebook_lines=True,
            jpeg_quality=68,
        ),
        OCRStressVariant(
            name="glare_reflection",
            rotate_degrees=-2,
            shadow=True,
            glare=True,
            notebook_lines=True,
            jpeg_quality=74,
        ),
        OCRStressVariant(
            name="two_column",
            rotate_degrees=-2,
            shadow=True,
            noise=True,
            notebook_lines=True,
            two_columns=True,
            jpeg_quality=72,
        ),
        OCRStressVariant(
            name="cropped_edge",
            rotate_degrees=4,
            shadow=True,
            notebook_lines=True,
            crop_ratio=0.035,
            jpeg_quality=70,
        ),
        OCRStressVariant(
            name="handwriting_like",
            rotate_degrees=-3,
            contrast=0.82,
            brightness=0.96,
            noise=True,
            notebook_lines=True,
            handwriting_like=True,
            jpeg_quality=68,
        ),
    ]


def generate_ocr_stress_samples(
    *,
    output_dir: Path | str,
    cases: list[OCRStressCase] | None = None,
    variants: list[OCRStressVariant] | None = None,
    limit: int | None = None,
) -> Path:
    output = Path(output_dir)
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    selected_cases = cases or default_stress_cases()
    selected_variants = variants or default_stress_variants()

    rows: list[dict] = []
    for case in selected_cases:
        for variant in selected_variants:
            if limit is not None and len(rows) >= limit:
                break
            sample_id = f"{case.sample_id}_{variant.name}"
            image_name = f"{sample_id}.jpg"
            _render_sample_image(
                target=image_dir / image_name,
                case=case,
                variant=variant,
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "image_path": f"images/{image_name}",
                    "subject": case.subject,
                    "grade": case.grade,
                    "tags": _variant_tags(case=case, variant=variant),
                    "expected_items": [
                        item.model_dump(mode="json") for item in case.expected_items
                    ],
                }
            )
        if limit is not None and len(rows) >= limit:
            break

    manifest = output / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest


def _render_sample_image(
    *,
    target: Path,
    case: OCRStressCase,
    variant: OCRStressVariant,
) -> None:
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

    width = 1600
    height = _canvas_height_for_lines(case.lines, two_columns=variant.two_columns)
    image = Image.new("RGB", (width, height), (250, 249, 244))
    draw = ImageDraw.Draw(image)
    if variant.notebook_lines:
        for y in range(90, height - 80, 72):
            draw.line((70, y + 52, width - 70, y + 52), fill=(220, 226, 226), width=2)
    if variant.shadow:
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(overlay)
        shadow_draw.polygon(
            [(width * 0.58, 0), (width, 0), (width, height), (width * 0.76, height)],
            fill=(0, 0, 0, 38),
        )
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(image)

    font = _load_font(46 if len(case.lines) > 4 else 50)
    if variant.two_columns and len(case.lines) > 4:
        _draw_two_column_text(draw, case.lines, font=font, variant=variant)
    else:
        _draw_stacked_text(draw, case.lines, font=font, variant=variant)
    if variant.glare:
        image = _add_glare(image)

    if variant.noise:
        image = _add_noise(image, seed=case.sample_id + variant.name)
    if variant.rotate_degrees:
        image = image.rotate(
            variant.rotate_degrees,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=(244, 243, 238),
        )
    if variant.blur_radius:
        image = image.filter(ImageFilter.GaussianBlur(radius=variant.blur_radius))
    if variant.contrast != 1.0:
        image = ImageEnhance.Contrast(image).enhance(variant.contrast)
    if variant.brightness != 1.0:
        image = ImageEnhance.Brightness(image).enhance(variant.brightness)
    if variant.crop_ratio > 0:
        image = _crop_edges(image, ratio=variant.crop_ratio)

    image.save(target, format="JPEG", quality=max(30, min(95, variant.jpeg_quality)))


def _variant_tags(*, case: OCRStressCase, variant: OCRStressVariant) -> list[str]:
    tags = ["synthetic", variant.name, case.subject]
    if len(case.expected_items) > 1:
        tags.append("multi_item")
    if variant.two_columns:
        tags.append("two_column")
    if variant.crop_ratio > 0:
        tags.append("cropped_edge")
    if variant.handwriting_like:
        tags.append("handwriting_like")
    if variant.blur_radius >= 2.0:
        tags.append("out_of_focus")
    if variant.glare:
        tags.append("glare_reflection")
    return list(dict.fromkeys(tags))


def _draw_stacked_text(draw, lines: list[str], *, font, variant: OCRStressVariant) -> None:
    y = 105
    for index, line in enumerate(lines):
        if not line:
            y += 42
            continue
        _draw_line(draw, (95, y), line, font=font, variant=variant, seed=index)
        y += 82


def _draw_two_column_text(draw, lines: list[str], *, font, variant: OCRStressVariant) -> None:
    groups = _split_question_groups(lines)
    left_groups, right_groups = _split_two_column_groups(groups)
    _draw_groups(draw, left_groups, x=90, y=95, font=font, variant=variant, seed_offset=0)
    _draw_groups(draw, right_groups, x=850, y=95, font=font, variant=variant, seed_offset=100)


def _canvas_height_for_lines(lines: list[str], *, two_columns: bool) -> int:
    if len(lines) <= 4:
        return 860
    if not two_columns:
        non_empty_count = sum(1 for line in lines if line)
        blank_count = len(lines) - non_empty_count
        return max(1100, 105 + non_empty_count * 82 + blank_count * 42 + 120)

    groups = _split_question_groups(lines)
    left_groups, right_groups = _split_two_column_groups(groups)
    return max(1100, _column_height(left_groups), _column_height(right_groups))


def _split_two_column_groups(groups: list[list[str]]) -> tuple[list[list[str]], list[list[str]]]:
    midpoint = (len(groups) + 1) // 2
    return groups[:midpoint], groups[midpoint:]


def _column_height(groups: list[list[str]]) -> int:
    line_count = sum(len(group) for group in groups)
    return 95 + line_count * 62 + len(groups) * 38 + 120


def _split_question_groups(lines: list[str]) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line:
            current.append(line)
            continue
        if current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _draw_groups(
    draw,
    groups: list[list[str]],
    *,
    x: int,
    y: int,
    font,
    variant: OCRStressVariant,
    seed_offset: int,
) -> None:
    current_y = y
    for group_index, group in enumerate(groups):
        for line_index, line in enumerate(group):
            seed = seed_offset + group_index * 10 + line_index
            for wrapped_line in _wrap_text_to_width(draw, line, font=font, max_width=650):
                _draw_line(draw, (x, current_y), wrapped_line, font=font, variant=variant, seed=seed)
                current_y += 62
        current_y += 38


def _wrap_text_to_width(draw, text: str, *, font, max_width: int) -> list[str]:
    if not text:
        return [text]
    lines: list[str] = []
    current = ""
    for char in text:
        candidate = f"{current}{char}"
        try:
            width = draw.textlength(candidate, font=font)
        except Exception:
            width = len(candidate) * 24
        if current and width > max_width:
            lines.append(current)
            current = char
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _draw_line(draw, xy: tuple[int, int], text: str, *, font, variant: OCRStressVariant, seed: int) -> None:
    if not variant.handwriting_like:
        draw.text(xy, text, fill=(28, 28, 28), font=font)
        return
    rng = random.Random(f"{text}:{seed}")
    x, y = xy
    for char in text:
        dx = rng.randint(-1, 2)
        dy = rng.randint(-2, 2)
        draw.text((x + dx, y + dy), char, fill=(34, 34, 34), font=font)
        try:
            char_width = draw.textlength(char, font=font)
        except Exception:
            char_width = 28
        x += int(char_width) + rng.randint(-1, 2)


def _crop_edges(image, *, ratio: float):
    width, height = image.size
    crop_x = int(width * ratio)
    crop_y = int(height * ratio * 0.6)
    return image.crop((crop_x, crop_y, width - crop_x, height - crop_y))


def _load_font(size: int):
    from PIL import ImageFont

    for path in [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _add_noise(image, *, seed: str):
    rng = random.Random(seed)
    pixels = image.load()
    width, height = image.size
    for _ in range(int(width * height * 0.015)):
        x = rng.randrange(width)
        y = rng.randrange(height)
        r, g, b = pixels[x, y]
        delta = rng.randrange(-28, 29)
        pixels[x, y] = (
            max(0, min(255, r + delta)),
            max(0, min(255, g + delta)),
            max(0, min(255, b + delta)),
        )
    return image


def _add_glare(image):
    from PIL import Image, ImageDraw, ImageFilter

    width, height = image.size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.ellipse(
        (
            int(width * 0.44),
            int(height * 0.14),
            int(width * 0.86),
            int(height * 0.42),
        ),
        fill=(255, 255, 255, 190),
    )
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=max(8, width // 120)))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

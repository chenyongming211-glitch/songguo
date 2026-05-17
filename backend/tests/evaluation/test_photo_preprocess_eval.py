from __future__ import annotations

import json

import cv2
import numpy as np

from songguo.backend.evaluation.ocr_eval import load_ocr_samples
from songguo.backend.evaluation.photo_preprocess_eval import evaluate_photo_preprocess_samples


def _write_image(path, image) -> None:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    assert ok
    path.write_bytes(encoded.tobytes())


def test_evaluate_photo_preprocess_samples_reports_region_quality_metrics(tmp_path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image = np.full((900, 1300, 3), (250, 250, 246), dtype=np.uint8)
    for y, number in ((150, "1."), (420, "2.")):
        cv2.putText(image, f"{number} 48 / 6 = ?", (160, y), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (20, 20, 20), 4)
        cv2.putText(image, "Answer: 8", (200, y + 78), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 3)
    _write_image(image_dir / "math.jpg", image)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "math_two_items",
                "image_path": "images/math.jpg",
                "subject": "math",
                "expected_items": [
                    {"question_text": "48 / 6 = ?", "child_answer": "8"},
                    {"question_text": "48 / 6 = ?", "child_answer": "8"},
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = evaluate_photo_preprocess_samples(samples=load_ocr_samples(manifest))

    assert report.total == 1
    assert report.failure_count == 0
    assert report.usable_region_rate == 1.0
    assert report.valid_box_rate == 1.0
    assert report.strict_region_rate == 1.0
    assert report.average_preprocess_ms >= 0

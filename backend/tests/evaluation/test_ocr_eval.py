from __future__ import annotations

import json
from pathlib import Path

from songguo.backend.evaluation.ocr_eval import (
    OCRExpectedItem,
    evaluate_ocr_samples,
    load_ocr_samples,
)
from songguo.backend.services.learning.photo_review import (
    DeterministicOCRProvider,
    OCRDraft,
    OCRItemDraft,
)


def test_load_ocr_samples_reads_jsonl_manifest_with_relative_paths(tmp_path) -> None:
    image_path = tmp_path / "fixtures" / "english.txt"
    image_path.parent.mkdir()
    image_path.write_text(
        "QUESTION: Choose the correct tense: He ____ to school yesterday.\nANSWER: go",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "english_tense_001",
                "image_path": "fixtures/english.txt",
                "subject": "english",
                "expected_items": [
                    {
                        "question_text": "Choose the correct tense: He ____ to school yesterday.",
                        "child_answer": "go",
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    samples = load_ocr_samples(manifest)

    assert len(samples) == 1
    assert samples[0].sample_id == "english_tense_001"
    assert samples[0].image_path == image_path
    assert samples[0].expected_items == [
        OCRExpectedItem(
            question_text="Choose the correct tense: He ____ to school yesterday.",
            child_answer="go",
        )
    ]


def test_evaluate_ocr_samples_scores_item_question_and_answer_matches(tmp_path) -> None:
    image_path = tmp_path / "math.txt"
    image_path.write_text("QUESTION: 48 ÷ 6 = ?\nANSWER: 8", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "math_division_001",
                "image_path": "math.txt",
                "subject": "math",
                "expected_items": [{"question_text": "48 ÷ 6 = ?", "child_answer": "8"}],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = evaluate_ocr_samples(
        samples=load_ocr_samples(manifest),
        ocr_provider=DeterministicOCRProvider(),
    )

    assert report.total == 1
    assert report.passed == 1
    assert report.pass_rate == 1.0
    assert report.item_count_match_rate == 1.0
    assert report.question_match_rate == 1.0
    assert report.answer_match_rate == 1.0
    assert report.results[0].status == "passed"
    assert report.results[0].actual_items[0].child_answer == "8"


def test_evaluate_ocr_samples_reports_bbox_quality_metrics(tmp_path) -> None:
    image_path = tmp_path / "math.txt"
    image_path.write_text("placeholder", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "math_with_bbox",
                "image_path": "math.txt",
                "subject": "math",
                "expected_items": [{"question_text": "48 ÷ 6 = ?", "child_answer": "8"}],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    class BBoxOCRProvider:
        def recognize(self, content: bytes, *, filename: str):
            return OCRDraft(
                raw_text="48 ÷ 6 = ?\n孩子答案：8",
                confidence=0.95,
                needs_confirmation=False,
                detected_regions=[],
                items=[
                    OCRItemDraft(
                        item_index=1,
                        question_text="48 ÷ 6 = ?",
                        child_answer="8",
                        confidence=0.95,
                        bbox={"x": 80, "y": 120, "width": 820, "height": 260},
                    )
                ],
            )

    report = evaluate_ocr_samples(
        samples=load_ocr_samples(manifest),
        ocr_provider=BBoxOCRProvider(),
    )

    assert report.passed == 1
    assert report.usable_bbox_rate == 1.0
    assert report.valid_bbox_rate == 1.0
    assert report.results[0].usable_bbox_count == 1
    assert report.results[0].valid_bbox_count == 1


def test_evaluate_ocr_samples_reports_answer_coverage_and_action_counts(tmp_path) -> None:
    image_path = tmp_path / "multi.txt"
    image_path.write_text("placeholder", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "multi_with_missing_answer",
                "image_path": "multi.txt",
                "subject": "math",
                "expected_items": [
                    {"question_text": "48 ÷ 6 = ?", "child_answer": "8"},
                    {"question_text": "36 ÷ 5 = ?", "child_answer": "7余1"},
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    class ActionOCRProvider:
        def recognize(self, content: bytes, *, filename: str):
            return OCRDraft(
                raw_text="1. 48 ÷ 6 = ?\n孩子答案：8\n2. 36 ÷ 5 = ?",
                confidence=0.88,
                source="aliyun_edu",
                items=[
                    OCRItemDraft(
                        item_index=1,
                        question_text="48 ÷ 6 = ?",
                        child_answer="8",
                        source_action="RecognizeEduPaperCut",
                    ),
                    OCRItemDraft(
                        item_index=2,
                        question_text="36 ÷ 5 = ?",
                        child_answer="",
                        source_action="RecognizeEduPaperCut",
                    ),
                ],
            )

    report = evaluate_ocr_samples(
        samples=load_ocr_samples(manifest),
        ocr_provider=ActionOCRProvider(),
    )

    assert report.answer_coverage_rate == 0.5
    assert report.ocr_action_counts == {"RecognizeEduPaperCut": 2}
    assert report.results[0].actual_answer_count == 1
    assert report.results[0].answer_coverage_rate == 0.5


def test_evaluate_ocr_samples_reports_answer_mismatch(tmp_path) -> None:
    image_path = tmp_path / "wrong_answer.txt"
    image_path.write_text("QUESTION: 48 ÷ 6 = ?\nANSWER: 7", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "math_division_wrong",
                "image_path": "wrong_answer.txt",
                "expected_items": [{"question_text": "48 ÷ 6 = ?", "child_answer": "8"}],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = evaluate_ocr_samples(
        samples=load_ocr_samples(manifest),
        ocr_provider=DeterministicOCRProvider(),
    )

    assert report.total == 1
    assert report.failed == 1
    assert report.answer_match_rate == 0.0
    assert report.results[0].status == "failed"
    assert "answer_mismatch:item_1" in report.results[0].errors


def test_evaluate_ocr_samples_ignores_quote_style_for_chinese_question_match(tmp_path) -> None:
    image_path = tmp_path / "chinese.txt"
    image_path.write_text("placeholder", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "chinese_sentence_quote_style",
                "image_path": "chinese.txt",
                "subject": "chinese",
                "expected_items": [
                    {
                        "question_text": "用“因为……所以……”造句。",
                        "child_answer": "因为下雨，所以我带伞。",
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    class QuoteStyleOCRProvider:
        def recognize(self, content: bytes, *, filename: str):
            return OCRDraft(
                raw_text='用 "因为……所以……" 造句。\n孩子答案：因为下雨，所以我带伞。',
                confidence=0.96,
                needs_confirmation=False,
                items=[
                    OCRItemDraft(
                        item_index=1,
                        question_text='用 "因为……所以……" 造句。',
                        child_answer="因为下雨，所以我带伞。",
                        confidence=0.96,
                    )
                ],
            )

    report = evaluate_ocr_samples(
        samples=load_ocr_samples(manifest),
        ocr_provider=QuoteStyleOCRProvider(),
    )

    assert report.passed == 1
    assert report.question_match_rate == 1.0

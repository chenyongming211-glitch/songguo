from __future__ import annotations

import json
import time

from songguo.backend.evaluation.ocr_eval import load_ocr_samples
from songguo.backend.evaluation.photo_submission_e2e_eval import (
    evaluate_photo_submission_samples,
)
from songguo.backend.services.learning.photo_review import (
    DeterministicOCRProvider,
    OCRDraft,
    OCRItemDraft,
)
from songguo.backend.services.learning.service import LearningService
from songguo.backend.services.learning.store import InMemoryLearningStore


def test_evaluate_photo_submission_samples_runs_ocr_to_submission_flow(tmp_path) -> None:
    fixture = tmp_path / "english.txt"
    fixture.write_text(
        "QUESTION: Choose the correct tense: He ____ to school yesterday.\nANSWER: go",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "english_tense",
                "image_path": "english.txt",
                "subject": "english",
                "grade": 4,
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

    report = evaluate_photo_submission_samples(
        samples=load_ocr_samples(manifest),
        ocr_provider=DeterministicOCRProvider(),
    )

    assert report.total == 1
    assert report.passed == 1
    assert report.pass_rate == 1.0
    assert report.subject_match_rate == 1.0
    assert report.route_match_rate == 1.0
    assert report.no_manual_confirm_rate == 1.0
    result = report.results[0]
    assert result.status == "passed"
    assert result.detected_subject == "english"
    assert result.route_to == "english_basic_tutor"
    assert result.submission_status == "tutoring"
    assert result.wrong_count == 1
    assert result.item_results[0].judge_result == "wrong"
    assert result.item_results[0].rubric_outcome == "wrong"
    assert result.errors == []
    assert result.final_judged_item_count == 1
    assert report.final_judgement_rate == 1.0
    assert report.answer_coverage_rate == 1.0
    assert report.pending_rate == 0.0


def test_evaluate_photo_submission_samples_records_ocr_observability(tmp_path) -> None:
    fixture = tmp_path / "math.txt"
    fixture.write_text("QUESTION: 48 ÷ 6 = ?\nANSWER: 8", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "math_division",
                "image_path": "math.txt",
                "subject": "math",
                "grade": 4,
                "expected_items": [{"question_text": "48 ÷ 6 = ?", "child_answer": "8"}],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    store = InMemoryLearningStore()
    service = LearningService(store=store)

    report = evaluate_photo_submission_samples(
        samples=load_ocr_samples(manifest),
        ocr_provider=DeterministicOCRProvider(),
        service=service,
    )

    assert report.passed == 1
    ocr_log = next(
        log for log in store.list_ai_call_logs("child_photo_e2e")
        if log.operation == "photo_ocr.recognize"
    )
    assert ocr_log.provider == "deterministic_ocr"
    assert ocr_log.model == "local_fixture"
    assert ocr_log.status == "success"
    assert ocr_log.agent == "DeterministicOCRProvider"
    assert ocr_log.latency_ms >= 0
    assert ocr_log.metadata["sample_id"] == "math_division"
    assert ocr_log.metadata["item_count"] == 1
    assert ocr_log.metadata["answer_count"] == 1


def test_evaluate_photo_submission_samples_reports_bbox_and_final_judgement_metrics(tmp_path) -> None:
    fixture = tmp_path / "math.txt"
    fixture.write_text("QUESTION: 48 ÷ 6 = ?\nANSWER: 8", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "math_division",
                "image_path": "math.txt",
                "subject": "math",
                "grade": 4,
                "expected_items": [{"question_text": "48 ÷ 6 = ?", "child_answer": "8"}],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = evaluate_photo_submission_samples(
        samples=load_ocr_samples(manifest),
        ocr_provider=DeterministicOCRProvider(),
    )

    assert report.passed == 1
    assert report.usable_bbox_rate == 1.0
    assert report.valid_bbox_rate == 1.0
    assert report.final_judgement_rate == 1.0
    assert report.results[0].usable_bbox_count == 1
    assert report.results[0].valid_bbox_count == 1
    assert report.results[0].final_judged_item_count == 1


def test_evaluate_photo_submission_samples_reports_pending_fallback_metrics(tmp_path) -> None:
    fixture = tmp_path / "missing_answer.txt"
    fixture.write_text("placeholder", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "missing_answer",
                "image_path": "missing_answer.txt",
                "subject": "math",
                "grade": 4,
                "expected_items": [{"question_text": "48 ÷ 6 = ?", "child_answer": "8"}],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    class MissingAnswerProvider:
        def recognize(self, content: bytes, *, filename: str):
            return OCRDraft(
                raw_text="48 ÷ 6 = ?",
                confidence=0.72,
                source="aliyun_edu",
                items=[
                    OCRItemDraft(
                        item_index=1,
                        question_text="48 ÷ 6 = ?",
                        child_answer="",
                        confidence=0.72,
                        source_action="RecognizeEduPaperCut",
                        quality_warnings=["answer_missing"],
                    )
                ],
            )

    report = evaluate_photo_submission_samples(
        samples=load_ocr_samples(manifest),
        ocr_provider=MissingAnswerProvider(),
    )

    assert report.failed == 1
    assert report.answer_coverage_rate == 0.0
    assert report.fallback_running_rate == 1.0
    assert report.pending_rate == 1.0
    assert report.ocr_action_counts == {"RecognizeEduPaperCut": 1}
    assert report.results[0].fallback_running_count == 1
    assert report.results[0].pending_item_count == 1
    assert report.results[0].item_results[0].display_status == "fallback_running"


def test_evaluate_photo_submission_samples_reports_subject_mismatch(tmp_path) -> None:
    fixture = tmp_path / "math.txt"
    fixture.write_text("QUESTION: 48 ÷ 6 = ?\nANSWER: 8", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "math_as_english_expected",
                "image_path": "math.txt",
                "subject": "english",
                "grade": 4,
                "expected_items": [{"question_text": "48 ÷ 6 = ?", "child_answer": "8"}],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = evaluate_photo_submission_samples(
        samples=load_ocr_samples(manifest),
        ocr_provider=DeterministicOCRProvider(),
    )

    assert report.total == 1
    assert report.failed == 1
    assert report.subject_match_rate == 0.0
    assert report.results[0].status == "failed"
    assert "subject_mismatch:expected_english:actual_math" in report.results[0].errors


def test_evaluate_photo_submission_samples_retries_transient_ocr_error(tmp_path) -> None:
    fixture = tmp_path / "math.txt"
    fixture.write_text("QUESTION: 48 ÷ 6 = ?\nANSWER: 8", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "math_division",
                "image_path": "math.txt",
                "subject": "math",
                "grade": 4,
                "expected_items": [{"question_text": "48 ÷ 6 = ?", "child_answer": "8"}],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    class FlakyOCRProvider:
        def __init__(self) -> None:
            self.calls = 0
            self.fallback = DeterministicOCRProvider()

        def recognize(self, content: bytes, *, filename: str):
            self.calls += 1
            if self.calls == 1:
                time.sleep(0.003)
                raise TimeoutError("temporary network failure")
            return self.fallback.recognize(content, filename=filename)

    provider = FlakyOCRProvider()
    store = InMemoryLearningStore()
    service = LearningService(store=store)

    report = evaluate_photo_submission_samples(
        samples=load_ocr_samples(manifest),
        ocr_provider=provider,
        service=service,
        sample_retries=1,
    )

    assert provider.calls == 2
    assert report.passed == 1
    assert report.results[0].status == "passed"
    assert report.results[0].retry_count == 1
    ocr_logs = [
        log for log in store.list_ai_call_logs("child_photo_e2e")
        if log.operation == "photo_ocr.recognize"
    ]
    assert [log.status for log in ocr_logs] == ["error", "success"]
    assert ocr_logs[0].failure_reason == "TimeoutError"
    assert ocr_logs[0].latency_ms > 0
    assert ocr_logs[0].metadata["retry_count"] == 0
    assert ocr_logs[1].metadata["retry_count"] == 1


def test_evaluate_photo_submission_samples_counts_ocr_error_against_match_rates(tmp_path) -> None:
    fixture = tmp_path / "math.txt"
    fixture.write_text("QUESTION: 48 ÷ 6 = ?\nANSWER: 8", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "math_division",
                "image_path": "math.txt",
                "subject": "math",
                "grade": 4,
                "expected_items": [{"question_text": "48 ÷ 6 = ?", "child_answer": "8"}],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    class FailingOCRProvider:
        def recognize(self, content: bytes, *, filename: str):
            raise TimeoutError("temporary network failure")

    report = evaluate_photo_submission_samples(
        samples=load_ocr_samples(manifest),
        ocr_provider=FailingOCRProvider(),
        sample_retries=0,
    )

    assert report.failed == 1
    assert report.question_match_rate == 0.0
    assert report.answer_match_rate == 0.0
    assert report.results[0].expected_item_count == 1

from __future__ import annotations

import threading
import time

from songguo.backend.services.learning.math_structuring import ProblemAnalysis
from songguo.backend.services.learning.submission_evaluator import (
    evaluate_basic_subject_submission_items,
    evaluate_submission_items,
)
from songguo.backend.services.learning.submission_models import (
    EvidenceType,
    JudgeResult,
    LearningItem,
    SourceType,
)
from songguo.backend.services.learning.store import InMemoryLearningStore


class FixtureMathGateway:
    def analyze(self, *, question_text: str, grade: int, subject: str) -> ProblemAnalysis:
        if "36" in question_text:
            return ProblemAnalysis(
                subject=subject,
                grade=grade,
                problem_type="division_with_remainder",
                knowledge_points=["有余数除法"],
                target="每只几颗，还剩几颗",
                final_answer="每只7颗，还剩1颗",
                confidence=0.96,
                source="fixture",
                solution_steps=[
                    {"id": "step_1", "goal": "先试商", "expression": "36 ÷ 5", "result": "7余1"}
                ],
                common_misconceptions=[
                    {"tag": "remainder_not_less_than_divisor", "description": "余数没有小于除数"}
                ],
            )
        return ProblemAnalysis(
            subject=subject,
            grade=grade,
            problem_type="division",
            knowledge_points=["表内除法"],
            target="计算商",
            final_answer="8",
            confidence=0.98,
            source="fixture",
            solution_steps=[
                {"id": "step_1", "goal": "计算除法", "expression": "48 ÷ 6", "result": "8"}
            ],
        )


def test_evaluate_submission_items_deposits_correct_and_wrong_items() -> None:
    store = InMemoryLearningStore()
    submission = store.create_submission(
        child_id="child_001",
        subject="math",
        grade=4,
        source_type=SourceType.TEXT,
        raw_text="两道题",
    )
    store.add_submission_item(
        LearningItem(
            submission_id=submission.submission_id,
            child_id="child_001",
            item_index=1,
            question_text="36 颗松果平均分给 5 只小松鼠，每只几颗，还剩几颗？",
            child_answer="每只 6 颗，还剩 6 颗",
        )
    )
    store.add_submission_item(
        LearningItem(
            submission_id=submission.submission_id,
            child_id="child_001",
            item_index=2,
            question_text="48 ÷ 6 = ?",
            child_answer="8",
        )
    )

    snapshot = evaluate_submission_items(
        store=store,
        submission_id=submission.submission_id,
        math_gateway=FixtureMathGateway(),
    )

    first, second = snapshot.items
    assert first.judge_result == JudgeResult.WRONG
    assert first.question_type_id == "division_with_remainder"
    assert second.judge_result == JudgeResult.CORRECT
    assert second.question_type_id == "division"
    assert snapshot.submission.correct_count == 1
    assert snapshot.submission.wrong_count == 1
    assert [item.item_id for item in snapshot.tutor_queue] == [first.item_id]
    assert {evidence.evidence_type for evidence in snapshot.mastery_evidence} == {
        EvidenceType.WRONG_UNRESOLVED,
        EvidenceType.SUBMISSION_CORRECT,
    }
    assert store.list_wrong_questions("child_001")[0].normalized_question == first.question_text


def test_evaluate_submission_items_records_successful_math_gateway_observability() -> None:
    class ObservableMathGateway(FixtureMathGateway):
        provider = "deepseek"
        model = "deepseek-v4-flash"

    store = InMemoryLearningStore()
    submission = store.create_submission(
        child_id="child_001",
        subject="math",
        grade=4,
        source_type=SourceType.PHOTO,
        raw_text="48 ÷ 6 = ?\n孩子答案：8",
    )
    item = store.add_submission_item(
        LearningItem(
            submission_id=submission.submission_id,
            child_id="child_001",
            item_index=1,
            question_text="48 ÷ 6 = ?",
            child_answer="8",
        )
    )

    evaluate_submission_items(
        store=store,
        submission_id=submission.submission_id,
        math_gateway=ObservableMathGateway(),
    )

    logs = store.list_ai_call_logs("child_001")
    math_log = next(log for log in logs if log.operation == "math_gateway.analyze")
    assert math_log.status == "success"
    assert math_log.provider == "deepseek"
    assert math_log.model == "deepseek-v4-flash"
    assert math_log.agent == "MathProblemStructuringGateway"
    assert math_log.submission_id == submission.submission_id
    assert math_log.item_id == item.item_id
    assert math_log.latency_ms >= 0
    assert math_log.confidence == 0.98
    assert math_log.metadata["question_type_id"] == "division"
    assert math_log.metadata["source"] == "fixture"


def test_evaluate_submission_items_analyzes_math_items_concurrently(monkeypatch) -> None:
    class SlowObservableMathGateway(FixtureMathGateway):
        def __init__(self) -> None:
            self.active_count = 0
            self.max_active_count = 0
            self.lock = threading.Lock()

        def analyze(self, *, question_text: str, grade: int, subject: str) -> ProblemAnalysis:
            with self.lock:
                self.active_count += 1
                self.max_active_count = max(self.max_active_count, self.active_count)
            try:
                time.sleep(0.05)
                return super().analyze(
                    question_text=question_text,
                    grade=grade,
                    subject=subject,
                )
            finally:
                with self.lock:
                    self.active_count -= 1

    monkeypatch.setenv("SONGGUO_SUBMISSION_EVAL_MAX_WORKERS", "4")
    gateway = SlowObservableMathGateway()
    store = InMemoryLearningStore()
    submission = store.create_submission(
        child_id="child_001",
        subject="math",
        grade=4,
        source_type=SourceType.PHOTO,
        raw_text="四道题",
    )
    for index in range(1, 5):
        store.add_submission_item(
            LearningItem(
                submission_id=submission.submission_id,
                child_id="child_001",
                item_index=index,
                question_text="48 ÷ 6 = ?",
                child_answer="8",
            )
        )

    snapshot = evaluate_submission_items(
        store=store,
        submission_id=submission.submission_id,
        math_gateway=gateway,
    )

    assert gateway.max_active_count > 1
    assert snapshot.submission.correct_count == 4


def test_evaluate_submission_items_marks_item_manual_confirm_when_math_gateway_fails() -> None:
    class FailingMathGateway:
        def analyze(self, *, question_text: str, grade: int, subject: str):
            raise TimeoutError("model timed out")

    store = InMemoryLearningStore()
    submission = store.create_submission(
        child_id="child_001",
        subject="math",
        grade=4,
        source_type=SourceType.PHOTO,
        raw_text="48 ÷ 6 = ?\n孩子答案：8",
    )
    item = store.add_submission_item(
        LearningItem(
            submission_id=submission.submission_id,
            child_id="child_001",
            item_index=1,
            question_text="48 ÷ 6 = ?",
            child_answer="8",
        )
    )

    snapshot = evaluate_submission_items(
        store=store,
        submission_id=submission.submission_id,
        math_gateway=FailingMathGateway(),
    )

    judged_item = snapshot.items[0]
    assert judged_item.judge_result == JudgeResult.NEEDS_MANUAL_CONFIRM
    assert judged_item.data_json["reason"] == "math_gateway_error"
    assert judged_item.data_json["failure_reason"] == "TimeoutError"
    logs = store.list_ai_call_logs("child_001")
    assert logs[-1].operation == "math_gateway.analyze"
    assert logs[-1].status == "error"
    assert logs[-1].item_id == item.item_id
    assert logs[-1].failure_reason == "TimeoutError"


def test_evaluate_basic_subject_submission_items_records_correct_english_rubric() -> None:
    store = InMemoryLearningStore()
    submission = store.create_submission(
        child_id="child_001",
        subject="english",
        detected_subject="english",
        detected_task_type="grammar_fix",
        subject_confidence=0.91,
        route_to="english_basic_tutor",
        grade=4,
        source_type=SourceType.TEXT,
        raw_text="Choose the correct tense: He ____ to school yesterday.\nanswer: went",
    )
    item = store.add_submission_item(
        LearningItem(
            submission_id=submission.submission_id,
            child_id="child_001",
            item_index=1,
            question_text="Choose the correct tense: He ____ to school yesterday.",
            child_answer="went",
            detected_subject="english",
            detected_task_type="grammar_fix",
        )
    )

    snapshot = evaluate_basic_subject_submission_items(
        store=store,
        submission_id=submission.submission_id,
    )

    judged_item = snapshot.items[0]
    assert judged_item.judge_result == JudgeResult.CORRECT
    assert judged_item.question_type_id == "grammar_fix"
    assert judged_item.knowledge_point == "english_sentence_pattern"
    assert judged_item.data_json["basic_subject_rubric"]["outcome"] == "correct"
    assert snapshot.submission.correct_count == 1
    assert snapshot.submission.wrong_count == 0
    assert snapshot.tutor_queue == []
    assert store.list_wrong_questions("child_001") == []
    assert [(e.item_id, e.evidence_type, e.is_correct) for e in snapshot.mastery_evidence] == [
        (item.item_id, EvidenceType.SUBMISSION_CORRECT, True)
    ]


def test_evaluate_basic_subject_submission_items_queues_wrong_english_with_specific_rubric() -> None:
    store = InMemoryLearningStore()
    submission = store.create_submission(
        child_id="child_001",
        subject="english",
        detected_subject="english",
        detected_task_type="grammar_fix",
        subject_confidence=0.91,
        route_to="english_basic_tutor",
        grade=4,
        source_type=SourceType.TEXT,
        raw_text="Choose the correct tense: He ____ to school yesterday.\nanswer: go",
    )
    item = store.add_submission_item(
        LearningItem(
            submission_id=submission.submission_id,
            child_id="child_001",
            item_index=1,
            question_text="Choose the correct tense: He ____ to school yesterday.",
            child_answer="go",
            detected_subject="english",
            detected_task_type="grammar_fix",
        )
    )

    snapshot = evaluate_basic_subject_submission_items(
        store=store,
        submission_id=submission.submission_id,
    )

    judged_item = snapshot.items[0]
    assert judged_item.judge_result == JudgeResult.WRONG
    assert judged_item.question_type_id == "grammar_fix"
    assert judged_item.knowledge_point == "english_sentence_pattern"
    assert judged_item.misconception_tag == "english_past_tense_missing"
    assert judged_item.data_json["basic_subject_rubric"]["outcome"] == "wrong"
    assert judged_item.data_json["basic_subject_rubric"]["rubric_scores"]["form_accuracy"] == 0
    assert snapshot.submission.correct_count == 0
    assert snapshot.submission.wrong_count == 1
    assert [queue.item_id for queue in snapshot.tutor_queue] == [item.item_id]
    assert snapshot.mastery_evidence[0].evidence_type == EvidenceType.WRONG_UNRESOLVED
    assert store.list_wrong_questions("child_001")[0].last_misconception == "english_past_tense_missing"


def test_evaluate_basic_subject_submission_items_records_plausible_chinese_answer() -> None:
    store = InMemoryLearningStore()
    submission = store.create_submission(
        child_id="child_001",
        subject="chinese",
        detected_subject="chinese",
        detected_task_type="reading_comprehension",
        subject_confidence=0.88,
        route_to="chinese_basic_tutor",
        grade=4,
        source_type=SourceType.TEXT,
        raw_text="阅读短文，回答作者为什么这样做？\n孩子答案：因为他很着急，想快点帮别人。",
    )
    store.add_submission_item(
        LearningItem(
            submission_id=submission.submission_id,
            child_id="child_001",
            item_index=1,
            question_text="阅读短文，回答作者为什么这样做？",
            child_answer="因为他很着急，想快点帮别人。",
            detected_subject="chinese",
            detected_task_type="reading_comprehension",
        )
    )

    snapshot = evaluate_basic_subject_submission_items(
        store=store,
        submission_id=submission.submission_id,
    )

    judged_item = snapshot.items[0]
    assert judged_item.judge_result == JudgeResult.CORRECT
    assert judged_item.question_type_id == "reading_comprehension"
    assert judged_item.knowledge_point == "chinese_reading_summary"
    assert judged_item.data_json["basic_subject_rubric"]["outcome"] == "correct"
    assert judged_item.data_json["basic_subject_rubric"]["rubric_scores"]["task_alignment"] >= 1
    assert snapshot.submission.correct_count == 1
    assert snapshot.tutor_queue == []

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
        if "丙小区" in question_text:
            return ProblemAnalysis(
                subject=subject,
                grade=grade,
                problem_type="time calculation",
                knowledge_points=["时间表应用"],
                target="选择正确小区",
                final_answer="丙小区",
                confidence=0.94,
                source="fixture",
                solution_steps=[
                    {
                        "id": "step_1",
                        "goal": "根据下午4时停电时间段判断",
                        "expression": "15:00-16:10",
                        "result": "丙小区",
                    }
                ],
            )
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
    assert second.question_type_id == "math_oral_calculation"
    assert snapshot.submission.correct_count == 1
    assert snapshot.submission.wrong_count == 1
    assert [item.item_id for item in snapshot.tutor_queue] == [first.item_id]
    assert {evidence.evidence_type for evidence in snapshot.mastery_evidence} == {
        EvidenceType.WRONG_UNRESOLVED,
        EvidenceType.SUBMISSION_CORRECT,
    }
    assert store.list_wrong_questions("child_001")[0].normalized_question == first.question_text


def test_evaluate_submission_items_matches_choice_letter_to_option_text() -> None:
    store = InMemoryLearningStore()
    submission = store.create_submission(
        child_id="child_001",
        subject="math",
        grade=3,
        source_type=SourceType.PHOTO,
        raw_text="选择题",
    )
    item = store.add_submission_item(
        LearningItem(
            submission_id=submission.submission_id,
            child_id="child_001",
            item_index=1,
            question_text=(
                "下面是某日部分小区停电公告，李阿姨下午4时回家，发现家中没电，"
                "她家在( )。A.甲小区 B.乙小区 C.丙小区"
            ),
            child_answer="C",
        )
    )

    snapshot = evaluate_submission_items(
        store=store,
        submission_id=submission.submission_id,
        math_gateway=FixtureMathGateway(),
    )

    judged = snapshot.items[0]
    assert judged.item_id == item.item_id
    assert judged.judge_result == JudgeResult.CORRECT
    assert judged.correct_answer == "丙小区"


def test_evaluate_submission_items_defers_choice_item_with_non_choice_ocr_answer() -> None:
    store = InMemoryLearningStore()
    submission = store.create_submission(
        child_id="child_001",
        subject="math",
        grade=3,
        source_type=SourceType.PHOTO,
        raw_text="选择题",
    )
    store.add_submission_item(
        LearningItem(
            submission_id=submission.submission_id,
            child_id="child_001",
            item_index=1,
            question_text="世界杯历时( )天。 A.38 B.39 C.40",
            child_answer="13",
            data_json={"ocr_action": "RecognizeEduPaperCut"},
        )
    )

    snapshot = evaluate_submission_items(
        store=store,
        submission_id=submission.submission_id,
        math_gateway=FixtureMathGateway(),
    )

    judged = snapshot.items[0]
    assert judged.judge_result == JudgeResult.NEEDS_MANUAL_CONFIRM
    assert judged.question_type_id == "math_choice"


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
            question_text="一共有48颗松果，平均分给6只小松鼠，每只几颗？",
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
                question_text="一共有48颗松果，平均分给6只小松鼠，每只几颗？",
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
            question_text="一共有48颗松果，平均分给6只小松鼠，每只几颗？",
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


def test_evaluate_submission_items_fast_judges_objective_math_before_gateway() -> None:
    class ShouldNotCallMathGateway:
        def analyze(self, *, question_text: str, grade: int, subject: str):
            raise AssertionError("objective math item should not call math gateway")

    store = InMemoryLearningStore()
    submission = store.create_submission(
        child_id="child_001",
        subject="math",
        grade=3,
        source_type=SourceType.PHOTO,
        raw_text="objective worksheet",
    )
    choice_item = store.add_submission_item(
        LearningItem(
            submission_id=submission.submission_id,
            child_id="child_001",
            item_index=1,
            question_text="积大约是5600的算式是( )。 A.59×79 B.79×61 C.79×71",
            child_answer="C",
        )
    )
    fill_item = store.add_submission_item(
        LearningItem(
            submission_id=submission.submission_id,
            child_id="child_001",
            item_index=2,
            question_text="87×23的积是( )位数，12×55的积的末尾有( )个0。",
            child_answer="四；1",
        )
    )

    snapshot = evaluate_submission_items(
        store=store,
        submission_id=submission.submission_id,
        math_gateway=ShouldNotCallMathGateway(),
    )

    assert snapshot.submission.correct_count == 2
    assert snapshot.submission.wrong_count == 0
    assert snapshot.tutor_queue == []
    assert [item.judge_result for item in snapshot.items] == [
        JudgeResult.CORRECT,
        JudgeResult.CORRECT,
    ]
    assert snapshot.items[0].data_json["objective_judge"]["correct_answer"] == "C"
    assert snapshot.items[1].data_json["objective_judge"]["correct_answer"] == "四；1"
    assert {evidence.item_id for evidence in snapshot.mastery_evidence} == {
        choice_item.item_id,
        fill_item.item_id,
    }


def test_evaluate_submission_items_routes_common_question_types_before_gateway() -> None:
    class ShouldNotCallMathGateway:
        def analyze(self, *, question_text: str, grade: int, subject: str):
            raise AssertionError("routed deterministic math item should not call math gateway")

    store = InMemoryLearningStore()
    submission = store.create_submission(
        child_id="child_001",
        subject="math",
        grade=3,
        source_type=SourceType.PHOTO,
        raw_text="mixed objective worksheet",
    )
    for index, question_text, child_answer in [
        (1, "48 ÷ 6 = ?", "8"),
        (2, "50×40( )15×80", ">"),
        (3, "两位数乘两位数，积可能是三位数，也可能是四位数。", "√"),
        (4, "积大约是5600的算式是( )。 A.59×79 B.79×61 C.79×71", "C"),
        (5, "87×23的积是( )位数，12×55的积的末尾有( )个0。", "四；1"),
        (6, "一盒月饼12个，王老师买了22盒，一共买了( )个。", "264"),
    ]:
        store.add_submission_item(
            LearningItem(
                submission_id=submission.submission_id,
                child_id="child_001",
                item_index=index,
                question_text=question_text,
                child_answer=child_answer,
            )
        )

    snapshot = evaluate_submission_items(
        store=store,
        submission_id=submission.submission_id,
        math_gateway=ShouldNotCallMathGateway(),
    )

    assert snapshot.submission.correct_count == 6
    assert snapshot.submission.wrong_count == 0
    assert snapshot.submission.needs_manual_confirm_count == 0
    assert snapshot.tutor_queue == []
    assert {item.question_type_id for item in snapshot.items} == {
        "math_oral_calculation",
        "math_comparison_sign",
        "math_true_false_fact",
        "math_choice_estimation",
        "math_fill_blank_calculation",
        "math_fill_blank_word_problem",
    }


def test_evaluate_submission_items_judges_deterministic_photo_items_beyond_sync_limit() -> None:
    class ShouldNotCallMathGateway:
        def analyze(self, *, question_text: str, grade: int, subject: str):
            raise AssertionError("deterministic objective items should not call math gateway")

    store = InMemoryLearningStore()
    submission = store.create_submission(
        child_id="child_001",
        subject="math",
        grade=3,
        source_type=SourceType.PHOTO,
        raw_text="long objective worksheet",
    )
    for index in range(1, 13):
        store.add_submission_item(
            LearningItem(
                submission_id=submission.submission_id,
                child_id="child_001",
                item_index=index,
                question_text="积大约是5600的算式是( )。 A.59×79 B.79×61 C.79×71",
                child_answer="C",
                data_json={"ocr_action": "RecognizeEduPaperCut"},
            )
        )

    snapshot = evaluate_submission_items(
        store=store,
        submission_id=submission.submission_id,
        math_gateway=ShouldNotCallMathGateway(),
    )

    assert snapshot.submission.correct_count == 12
    assert snapshot.submission.needs_manual_confirm_count == 0
    assert [item.judge_result for item in snapshot.items] == [JudgeResult.CORRECT] * 12
    assert all("visual_fallback" not in item.data_json for item in snapshot.items)


def test_evaluate_submission_items_splits_grouped_true_false_before_gateway() -> None:
    class ShouldNotCallMathGateway:
        def analyze(self, *, question_text: str, grade: int, subject: str):
            raise AssertionError("split true/false items should not call math gateway")

    store = InMemoryLearningStore()
    submission = store.create_submission(
        child_id="child_001",
        subject="math",
        grade=3,
        source_type=SourceType.PHOTO,
        raw_text="grouped worksheet block",
    )
    store.add_submission_item(
        LearningItem(
            submission_id=submission.submission_id,
            child_id="child_001",
            item_index=1,
            question_text=(
                "二、判断。1.两位数乘两位数，积可能是三位数，也可能是四位数。(V) "
                "2.两个乘数末尾共有2个0，积的末尾也一定有2个0。(x)"
            ),
            child_answer="V；x",
            data_json={"ocr_action": "RecognizeEduPaperCut"},
        )
    )

    snapshot = evaluate_submission_items(
        store=store,
        submission_id=submission.submission_id,
        math_gateway=ShouldNotCallMathGateway(),
    )

    assert snapshot.submission.correct_count == 2
    assert snapshot.submission.needs_manual_confirm_count == 0
    assert [(item.question_text, item.child_answer, item.judge_result) for item in snapshot.items] == [
        ("两位数乘两位数，积可能是三位数，也可能是四位数。", "V", JudgeResult.CORRECT),
        ("两个乘数末尾共有2个0，积的末尾也一定有2个0。", "x", JudgeResult.CORRECT),
    ]
    assert {item.data_json["split_from"]["question_type_id"] for item in snapshot.items} == {
        "math_grouped_true_false"
    }


def test_evaluate_submission_items_splits_grouped_comparison_before_gateway() -> None:
    class ShouldNotCallMathGateway:
        def analyze(self, *, question_text: str, grade: int, subject: str):
            raise AssertionError("split comparison items should not call math gateway")

    store = InMemoryLearningStore()
    submission = store.create_submission(
        child_id="child_001",
        subject="math",
        grade=3,
        source_type=SourceType.PHOTO,
        raw_text="multi blank worksheet block",
    )
    store.add_submission_item(
        LearningItem(
            submission_id=submission.submission_id,
            child_id="child_001",
            item_index=1,
            question_text="在○里填上“>”“<”或“=”。50×40○15×80 63×27○27×85 40×125○100×28",
            child_answer="><>",
            data_json={"ocr_action": "RecognizeEduPaperCut"},
        )
    )

    snapshot = evaluate_submission_items(
        store=store,
        submission_id=submission.submission_id,
        math_gateway=ShouldNotCallMathGateway(),
    )

    assert snapshot.submission.correct_count == 3
    assert snapshot.submission.needs_manual_confirm_count == 0
    assert [(item.question_text, item.child_answer, item.judge_result) for item in snapshot.items] == [
        ("50×40( )15×80", ">", JudgeResult.CORRECT),
        ("63×27( )27×85", "<", JudgeResult.CORRECT),
        ("40×125( )100×28", ">", JudgeResult.CORRECT),
    ]
    assert snapshot.items[0].data_json["evidence_trace"][0]["stage"] == "question_type_route"
    assert any(
        entry["stage"] == "objective_judge"
        for entry in snapshot.items[0].data_json["evidence_trace"]
    )


def test_evaluate_submission_items_splits_uncertain_multi_blank_answers() -> None:
    class ShouldNotCallMathGateway:
        def analyze(self, *, question_text: str, grade: int, subject: str):
            raise AssertionError("split multi-blank item should not call math gateway")

    store = InMemoryLearningStore()
    submission = store.create_submission(
        child_id="child_001",
        subject="math",
        grade=3,
        source_type=SourceType.PHOTO,
        raw_text="multi blank worksheet block",
    )
    store.add_submission_item(
        LearningItem(
            submission_id=submission.submission_id,
            child_id="child_001",
            item_index=1,
            question_text="口算21×50时，可以先算21×5=( )，再在积的后面添上( )个0。",
            child_answer="1055",
            data_json={"ocr_action": "RecognizeEduPaperCut"},
        )
    )

    snapshot = evaluate_submission_items(
        store=store,
        submission_id=submission.submission_id,
        math_gateway=ShouldNotCallMathGateway(),
    )

    assert [(item.question_text, item.child_answer, item.judge_result) for item in snapshot.items] == [
        ("21×5=( )", "105", JudgeResult.CORRECT),
        ("口算21×50时，再在积的后面添上( )个0。", "5", JudgeResult.WRONG),
    ]
    assert snapshot.items[1].correct_answer == "1"


def test_evaluate_submission_items_splits_grouped_oral_calculation_block() -> None:
    class ShouldNotCallMathGateway:
        def analyze(self, *, question_text: str, grade: int, subject: str):
            raise AssertionError("split oral calculation items should not call math gateway")

    store = InMemoryLearningStore()
    submission = store.create_submission(
        child_id="child_001",
        subject="math",
        grade=3,
        source_type=SourceType.PHOTO,
        raw_text="merged oral calculation worksheet block",
    )
    store.add_submission_item(
        LearningItem(
            submission_id=submission.submission_id,
            child_id="child_001",
            item_index=1,
            question_text="直接写得数。",
            child_answer="30000",
            data_json={
                "ocr_action": "RecognizeEduPaperOcr",
                "answer_extraction": {
                    "work_steps": "5×30=150 32×30=960 20×32=60 600×50=30000",
                },
            },
        )
    )

    snapshot = evaluate_submission_items(
        store=store,
        submission_id=submission.submission_id,
        math_gateway=ShouldNotCallMathGateway(),
    )

    assert [(item.question_text, item.child_answer, item.judge_result, item.correct_answer) for item in snapshot.items] == [
        ("5×30=", "150", JudgeResult.CORRECT, "150"),
        ("32×30=", "960", JudgeResult.CORRECT, "960"),
        ("20×32=", "60", JudgeResult.WRONG, "640"),
        ("600×50=", "30000", JudgeResult.CORRECT, "30000"),
    ]
    assert snapshot.submission.correct_count == 3
    assert snapshot.submission.wrong_count == 1


def test_evaluate_submission_items_keeps_vertical_calculation_block_pending() -> None:
    class ShouldNotCallMathGateway:
        def analyze(self, *, question_text: str, grade: int, subject: str):
            raise AssertionError("vertical calculation blocks need visual/coordinate review first")

    store = InMemoryLearningStore()
    submission = store.create_submission(
        child_id="child_001",
        subject="math",
        grade=3,
        source_type=SourceType.PHOTO,
        raw_text="vertical calculation worksheet block",
    )
    store.add_submission_item(
        LearningItem(
            submission_id=submission.submission_id,
            child_id="child_001",
            item_index=1,
            question_text=(
                "竖式计算-多位数乘法 题数：20 360×20=7200 "
                "18×11=198 270×13=3510 370×20=360"
            ),
            child_answer="360",
            data_json={"ocr_action": "RecognizeEduPaperOcr"},
        )
    )

    snapshot = evaluate_submission_items(
        store=store,
        submission_id=submission.submission_id,
        math_gateway=ShouldNotCallMathGateway(),
    )

    assert len(snapshot.items) == 1
    assert snapshot.items[0].judge_result == JudgeResult.NEEDS_MANUAL_CONFIRM
    assert snapshot.items[0].question_type_id == "math_vertical_calculation_block"
    assert snapshot.submission.needs_manual_confirm_count == 1


def test_evaluate_submission_items_splits_vertical_process_block() -> None:
    class ShouldNotCallMathGateway:
        def analyze(self, *, question_text: str, grade: int, subject: str):
            raise AssertionError("vertical process equations should be judged deterministically")

    store = InMemoryLearningStore()
    submission = store.create_submission(
        child_id="child_001",
        subject="math",
        grade=3,
        source_type=SourceType.PHOTO,
        raw_text="vertical process worksheet block",
    )
    store.add_submission_item(
        LearningItem(
            submission_id=submission.submission_id,
            child_id="child_001",
            item_index=1,
            question_text=(
                "用竖式计算。32×21=( 672 ) 32 46 208 "
                "解题过程：46×18=828 176×50=8800 208×74=15392"
            ),
            child_answer="15392",
            data_json={"ocr_action": "RecognizeEduPaperOcr"},
        )
    )

    snapshot = evaluate_submission_items(
        store=store,
        submission_id=submission.submission_id,
        math_gateway=ShouldNotCallMathGateway(),
    )

    assert [(item.question_text, item.child_answer, item.judge_result) for item in snapshot.items] == [
        ("46×18=", "828", JudgeResult.CORRECT),
        ("176×50=", "8800", JudgeResult.CORRECT),
        ("208×74=", "15392", JudgeResult.CORRECT),
    ]
    assert snapshot.submission.correct_count == 3


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
    assert any(
        entry["stage"] == "basic_subject_rubric"
        for entry in judged_item.data_json["evidence_trace"]
    )
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

from __future__ import annotations

from songguo.backend.services.learning.math_structuring import ProblemAnalysis
from songguo.backend.services.learning.submission_evaluator import evaluate_submission_items
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

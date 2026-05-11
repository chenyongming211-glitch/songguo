from __future__ import annotations

from langgraph.graph.state import CompiledStateGraph

from songguo.backend.services.learning.llm_session_runner import LLMSessionRunner
from songguo.backend.services.learning.math_structuring import ProblemAnalysis
from songguo.backend.services.learning.store import InMemoryLearningStore
from songguo.backend.services.learning.submission_graph import LearningSubmissionGraph
from songguo.backend.services.learning.submission_models import (
    EvidenceType,
    LearningSubmissionStatus,
    TutorQueueStatus,
)


def test_learning_submission_graph_uses_compiled_langgraph_state_graph() -> None:
    graph = LearningSubmissionGraph(store=InMemoryLearningStore(), session_runner=_fake_runner())

    assert isinstance(graph.graph, CompiledStateGraph)


def test_learning_submission_graph_create_draft_only_parses_items_before_confirmation() -> None:
    store = InMemoryLearningStore()
    graph = LearningSubmissionGraph(
        store=store,
        session_runner=_fake_runner(),
        math_gateway=_fixture_gateway(),
    )

    result = graph.start(
        child_id="child_001",
        subject="math",
        grade=4,
        source_type="text",
        raw_text="""
        1. 36 颗松果平均分给 5 只小松鼠，每只几颗，还剩几颗？
        孩子答案：每只 6 颗，还剩 6 颗

        2. 48 ÷ 6 = ?
        孩子答案：8
        """,
    )

    snapshot = store.get_submission_snapshot(result.submission_id)

    assert isinstance(result.submission_id, str)
    assert result.status == LearningSubmissionStatus.INTAKE_PENDING
    assert result.item_count == 2
    assert result.correct_count == 0
    assert result.wrong_count == 0
    assert snapshot.mastery_evidence == []
    assert snapshot.tutor_queue == []
    assert store.get_active_tutor_item(result.submission_id) is None


def test_learning_submission_graph_confirm_deposits_all_items_and_starts_first_wrong_tutor() -> None:
    store = InMemoryLearningStore()
    graph = LearningSubmissionGraph(
        store=store,
        session_runner=_fake_runner(),
        math_gateway=_fixture_gateway(),
    )
    draft = graph.start(
        child_id="child_001",
        subject="math",
        grade=4,
        source_type="text",
        raw_text="""
        1. 36 颗松果平均分给 5 只小松鼠，每只几颗，还剩几颗？
        孩子答案：每只 6 颗，还剩 6 颗

        2. 48 ÷ 6 = ?
        孩子答案：8
        """,
    )

    result = graph.confirm(draft.submission_id)
    snapshot = store.get_submission_snapshot(result.submission_id)
    active = store.get_active_tutor_item(result.submission_id)

    assert result.status == LearningSubmissionStatus.TUTORING
    assert result.item_count == 2
    assert result.correct_count == 1
    assert result.wrong_count == 1
    assert {e.evidence_type for e in snapshot.mastery_evidence} == {
        EvidenceType.SUBMISSION_CORRECT,
        EvidenceType.WRONG_UNRESOLVED,
    }
    assert len(snapshot.tutor_queue) == 1
    assert active is not None
    assert active.status == TutorQueueStatus.ACTIVE
    assert active.tutor_session_id is not None
    assert store.require_session(active.tutor_session_id).question_text.startswith("36 颗松果")


def test_learning_submission_graph_completes_first_wrong_item_and_starts_next() -> None:
    store = InMemoryLearningStore()
    graph = LearningSubmissionGraph(
        store=store,
        session_runner=_fake_runner(),
        math_gateway=_fixture_gateway(),
    )
    draft = graph.start(
        child_id="child_001",
        subject="math",
        grade=4,
        source_type="text",
        raw_text="""
        1. 36 颗松果平均分给 5 只小松鼠，每只几颗，还剩几颗？
        孩子答案：每只 6 颗，还剩 6 颗

        2. 36 颗松果平均分给 5 只小松鼠，每只几颗，还剩几颗？
        孩子答案：每只 8 颗
        """,
    )
    result = graph.confirm(draft.submission_id)
    first_active = store.get_active_tutor_item(result.submission_id)

    after_first = graph.complete_active_tutor_item(
        submission_id=result.submission_id,
        tutor_session_id=first_active.tutor_session_id,
    )
    second_active = store.get_active_tutor_item(result.submission_id)

    assert after_first.status == LearningSubmissionStatus.TUTORING
    assert second_active is not None
    assert second_active.queue_item_id != first_active.queue_item_id
    assert second_active.status == TutorQueueStatus.ACTIVE
    assert second_active.tutor_session_id is not None

    after_second = graph.complete_active_tutor_item(
        submission_id=result.submission_id,
        tutor_session_id=second_active.tutor_session_id,
    )

    assert after_second.status == LearningSubmissionStatus.COMPLETED
    assert store.get_active_tutor_item(result.submission_id) is None


def _fixture_gateway():
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
                    key_points=[
                        {
                            "id": "kp_division_remainder",
                            "name": "求商和余数",
                            "teaching_goal": "理解平均分时商和余数的含义",
                            "release_stage": "HINT_STEP_1",
                            "unlock_condition": "question_started",
                            "child_prompt": "36 除以 5，先想一想商可能是几？",
                            "expected_child_response": ["7", "7余1"],
                            "forbidden_content": ["每只7颗，还剩1颗"],
                        }
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
                key_points=[
                    {
                        "id": "kp_division",
                        "name": "表内除法",
                        "teaching_goal": "计算除法结果",
                        "release_stage": "HINT_STEP_1",
                        "unlock_condition": "question_started",
                        "child_prompt": "48 里面有几个 6？",
                        "expected_child_response": ["8"],
                        "forbidden_content": ["8"],
                    }
                ],
            )

    return FixtureMathGateway()


def _fake_runner() -> LLMSessionRunner:
    class FakeRunner:
        def start(self, **kwargs):
            from songguo.backend.services.learning.llm_session_runner import LLMSessionOutput

            return LLMSessionOutput(
                child_message="先说说你看到题目里要平均分给几只？",
            )

    return FakeRunner()

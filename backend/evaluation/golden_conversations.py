from __future__ import annotations

from pydantic import BaseModel, Field

from songguo.backend.services.learning.context_pack import build_student_context_pack
from songguo.backend.services.learning.llm_session_runner import LLMSessionRunner
from songguo.backend.services.learning.store import InMemoryLearningStore


class GoldenConversationCase(BaseModel):
    case_id: str
    category: str
    grade: int = 4
    question_text: str = "甲仓库有560袋米，乙仓库有420袋，甲运给乙80袋后，甲比乙少还是多多少袋？"
    prior_assistant_message: str = "先算甲仓库运出80袋后还剩多少袋？"
    child_message: str
    expected_readiness: str
    expected_child_answer_status: str
    expected_teacher_move: str
    expected_phase: str
    forbidden_message_fragments: list[str] = Field(default_factory=list)


class GoldenConversationReport(BaseModel):
    total: int
    passed_count: int = 0
    readiness_failures: int = 0
    child_answer_status_failures: int = 0
    teacher_move_failures: int = 0
    phase_failures: int = 0
    message_contract_failures: int = 0
    provider_failures: int = 0
    failed_case_ids: list[str] = Field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed_count / self.total


def build_golden_conversation_cases() -> list[GoldenConversationCase]:
    return [
        GoldenConversationCase(
            case_id="gc_learning_resistance",
            category="learning_resistance",
            child_message="今天不想练了，明天再说吧",
            expected_readiness="not_ready",
            expected_child_answer_status="learning_resistance",
            expected_teacher_move="pause_learning",
            expected_phase="LEARNING_PAUSED",
            forbidden_message_fragments=["你算算看", "继续做", "先把题目理一理"],
        ),
        GoldenConversationCase(
            case_id="gc_wellbeing_not_ready",
            category="wellbeing_not_ready",
            child_message="我今天有点头疼，所以一点思路都没有",
            expected_readiness="not_ready",
            expected_child_answer_status="wellbeing_not_ready",
            expected_teacher_move="pause_learning",
            expected_phase="LEARNING_PAUSED",
            forbidden_message_fragments=["你算算看", "甲仓库还剩多少", "先把题目理一理"],
        ),
        GoldenConversationCase(
            case_id="gc_answer_seeking",
            category="answer_seeking",
            child_message="你直接告诉我答案吧",
            expected_readiness="ready",
            expected_child_answer_status="answer_seeking",
            expected_teacher_move="refuse_direct_answer",
            expected_phase="WAIT_CHILD_ATTEMPT",
            forbidden_message_fragments=["甲比乙少20袋", "答案是"],
        ),
        GoldenConversationCase(
            case_id="gc_off_task",
            category="off_task",
            child_message="甲仓库像不像游戏背包？我想聊游戏装备",
            expected_readiness="needs_support",
            expected_child_answer_status="off_task",
            expected_teacher_move="redirect_to_learning",
            expected_phase="WAIT_CHILD_ATTEMPT",
            forbidden_message_fragments=["真有意思", "我们来聊游戏"],
        ),
        GoldenConversationCase(
            case_id="gc_wrong_answer_followup",
            category="wrong_answer_followup",
            child_message="560减80等于480，所以甲比乙多60袋",
            expected_readiness="ready",
            expected_child_answer_status="wrong",
            expected_teacher_move="ask_diagnostic_question",
            expected_phase="WAIT_CHILD_ATTEMPT",
            forbidden_message_fragments=["很棒", "算得很好", "甲比乙少20袋"],
        ),
        GoldenConversationCase(
            case_id="gc_confused",
            category="confused",
            child_message="我看不懂题，也不知道先算什么",
            expected_readiness="needs_support",
            expected_child_answer_status="confused",
            expected_teacher_move="simplify",
            expected_phase="WAIT_CHILD_ATTEMPT",
            forbidden_message_fragments=["答案是", "完整解法"],
        ),
        GoldenConversationCase(
            case_id="gc_resume_learning",
            category="resume_learning",
            child_message="我现在可以继续了",
            expected_readiness="ready",
            expected_child_answer_status="resume_request",
            expected_teacher_move="continue_tutoring",
            expected_phase="WAIT_CHILD_ATTEMPT",
            forbidden_message_fragments=["重新开始一道新题", "本次总结"],
        ),
    ]


def evaluate_golden_conversations(
    *,
    cases: list[GoldenConversationCase],
    session_runner: LLMSessionRunner,
) -> GoldenConversationReport:
    report = GoldenConversationReport(total=len(cases))
    for case in cases:
        try:
            output = session_runner.run(
                child_id="golden_conversation_child",
                grade=case.grade,
                context_pack=build_student_context_pack(
                    InMemoryLearningStore(),
                    child_id="golden_conversation_child",
                    grade=case.grade,
                    current_question=case.question_text,
                ),
                messages=[
                    {"role": "system", "content": f"golden_case_id: {case.case_id}"},
                    {"role": "user", "content": case.question_text},
                    {"role": "assistant", "content": case.prior_assistant_message},
                    {"role": "user", "content": case.child_message},
                ],
            )
        except Exception:
            report.provider_failures += 1
            report.failed_case_ids.append(f"{case.case_id}:provider")
            continue

        case_failed = False
        intent = output.teaching_intent
        if intent.learner_readiness != case.expected_readiness:
            report.readiness_failures += 1
            report.failed_case_ids.append(f"{case.case_id}:readiness")
            case_failed = True
        if intent.child_answer_status != case.expected_child_answer_status:
            report.child_answer_status_failures += 1
            report.failed_case_ids.append(f"{case.case_id}:child_answer_status")
            case_failed = True
        if intent.teacher_move != case.expected_teacher_move:
            report.teacher_move_failures += 1
            report.failed_case_ids.append(f"{case.case_id}:teacher_move")
            case_failed = True
        if output.structured_state.phase != case.expected_phase:
            report.phase_failures += 1
            report.failed_case_ids.append(f"{case.case_id}:phase")
            case_failed = True
        if _message_violates_contract(output.child_message, case):
            report.message_contract_failures += 1
            report.failed_case_ids.append(f"{case.case_id}:message_contract")
            case_failed = True
        if not case_failed:
            report.passed_count += 1
    return report


def _message_violates_contract(message: str, case: GoldenConversationCase) -> bool:
    return any(fragment in message for fragment in case.forbidden_message_fragments)

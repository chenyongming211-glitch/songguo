from __future__ import annotations

import json

from songguo.backend.evaluation.golden_conversations import (
    GoldenConversationCase,
    build_golden_conversation_cases,
    evaluate_golden_conversations,
)
from songguo.backend.services.learning.llm_session_runner import LLMSessionRunner


def test_golden_conversation_set_covers_child_real_world_intents() -> None:
    cases = build_golden_conversation_cases()
    categories = {case.category for case in cases}

    assert len(cases) >= 7
    assert {
        "learning_resistance",
        "wellbeing_not_ready",
        "answer_seeking",
        "off_task",
        "wrong_answer_followup",
        "confused",
        "resume_learning",
    }.issubset(categories)
    assert all(case.expected_phase for case in cases)
    assert all(case.expected_teacher_move for case in cases)


def test_golden_conversation_evaluator_checks_structured_intent_and_message_contract() -> None:
    case_outputs = {
        "gc_learning_resistance": {
            "learner_readiness": "not_ready",
            "child_answer_status": "learning_resistance",
            "teacher_move": "pause_learning",
            "phase": "LEARNING_PAUSED",
            "child_message": "可以，今天先到这里。明天回来时，我们从这道题继续。",
        },
        "gc_wellbeing_not_ready": {
            "learner_readiness": "not_ready",
            "child_answer_status": "wellbeing_not_ready",
            "teacher_move": "pause_learning",
            "phase": "LEARNING_PAUSED",
            "child_message": "那先休息。等舒服一点回来，我们再继续这道题。",
        },
        "gc_answer_seeking": {
            "learner_readiness": "ready",
            "child_answer_status": "answer_seeking",
            "teacher_move": "refuse_direct_answer",
            "phase": "WAIT_CHILD_ATTEMPT",
            "child_message": "我不能直接告诉答案。你先说说题目里甲仓库发生了什么变化？",
        },
        "gc_off_task": {
            "learner_readiness": "needs_support",
            "child_answer_status": "off_task",
            "teacher_move": "redirect_to_learning",
            "phase": "WAIT_CHILD_ATTEMPT",
            "child_message": "我们先回到这道题，只看一个小问题：甲运出80袋后会变多还是变少？",
        },
        "gc_wrong_answer_followup": {
            "learner_readiness": "ready",
            "child_answer_status": "wrong",
            "teacher_move": "ask_diagnostic_question",
            "phase": "WAIT_CHILD_ATTEMPT",
            "child_message": "先核对一下：乙仓库收到80袋以后，乙仓库现在是多少袋？",
        },
        "gc_confused": {
            "learner_readiness": "needs_support",
            "child_answer_status": "confused",
            "teacher_move": "simplify",
            "phase": "WAIT_CHILD_ATTEMPT",
            "child_message": "没关系，我们只看第一步：甲仓库运走80袋，是用加法还是减法？",
        },
        "gc_resume_learning": {
            "learner_readiness": "ready",
            "child_answer_status": "resume_request",
            "teacher_move": "continue_tutoring",
            "phase": "WAIT_CHILD_ATTEMPT",
            "child_message": "好，我们接着这道题。甲运出80袋以后，甲仓库还剩多少袋？",
        },
    }

    def fake_llm(prompt: str) -> str:
        case_id = next(key for key in case_outputs if key in prompt)
        output = case_outputs[case_id]
        return json.dumps(
            {
                "child_message": output["child_message"],
                "structured_state": {
                    "phase": output["phase"],
                    "answer_unlocked": False,
                    "current_key_point": "转移后比较",
                    "should_end_session": False,
                },
                "learning_deposit_delta": {
                    "knowledge_point": "比较问题",
                    "question_type": "转移后比较",
                    "evidence": "黄金会话评测。",
                },
                "teaching_intent": {
                    "learner_readiness": output["learner_readiness"],
                    "child_answer_status": output["child_answer_status"],
                    "teacher_move": output["teacher_move"],
                    "should_reveal_final_answer": False,
                    "next_question": "",
                },
                "practice_items": [],
            },
            ensure_ascii=False,
        )

    report = evaluate_golden_conversations(
        cases=build_golden_conversation_cases(),
        session_runner=LLMSessionRunner(llm_func=fake_llm),
    )

    assert report.total == len(build_golden_conversation_cases())
    assert report.passed_count == report.total
    assert report.failed_case_ids == []


def test_golden_conversation_evaluator_counts_pause_message_conflicts() -> None:
    def bad_pause_llm(_prompt: str) -> str:
        return json.dumps(
            {
                "child_message": "可以先休息一下。不过我们先把题目理一理，甲仓库还剩多少袋？",
                "structured_state": {
                    "phase": "LEARNING_PAUSED",
                    "answer_unlocked": False,
                    "current_key_point": "转移后比较",
                    "should_end_session": False,
                },
                "teaching_intent": {
                    "learner_readiness": "not_ready",
                    "child_answer_status": "wellbeing_not_ready",
                    "teacher_move": "pause_learning",
                    "should_reveal_final_answer": False,
                    "next_question": "",
                },
            },
            ensure_ascii=False,
        )

    report = evaluate_golden_conversations(
        cases=[
            GoldenConversationCase(
                case_id="gc_bad_pause",
                category="wellbeing_not_ready",
                child_message="我今天有点头疼，所以一点思路都没有",
                expected_readiness="not_ready",
                expected_child_answer_status="wellbeing_not_ready",
                expected_teacher_move="pause_learning",
                expected_phase="LEARNING_PAUSED",
                forbidden_message_fragments=["甲仓库还剩多少袋", "我们先把题目理一理"],
            )
        ],
        session_runner=LLMSessionRunner(llm_func=bad_pause_llm),
    )

    assert report.total == 1
    assert report.message_contract_failures == 1
    assert report.failed_case_ids == ["gc_bad_pause:message_contract"]

from __future__ import annotations

import json

from songguo.backend.evaluation.graph_runner import (
    GraphEvaluationCase,
    evaluate_math_tutor_graph,
)
from songguo.backend.services.learning.llm_session_runner import LLMSessionRunner


def test_graph_evaluation_runs_full_tutor_loop_and_reports_learning_assets() -> None:
    prompts: list[str] = []

    def fake_llm(prompt: str) -> str:
        prompts.append(prompt)
        if '"content": "128"' in prompt:
            message = "你已经算出总人数了。接下来想想：2辆车最多坐多少人？够不够？"
        else:
            message = "先不急着说几辆车。你能先算出一共有多少人吗？"
        return json.dumps(
            {
                "child_message": message,
                "structured_state": {
                    "phase": "WAIT_CHILD_ATTEMPT",
                    "answer_unlocked": False,
                    "current_key_point": "判断车辆容量",
                    "should_end_session": False,
                },
                "learning_deposit_delta": {
                    "knowledge_point": "限载进一应用题",
                    "question_type": "capacity_round_up",
                    "evidence": "模型正在引导孩子完成限载进一题。",
                },
                "practice_items": [],
            },
            ensure_ascii=False,
        )

    report = evaluate_math_tutor_graph(
        cases=[
            GraphEvaluationCase(
                question_id="graph_capacity_001",
                grade=3,
                question_text=(
                    "学校组织三年级学生春游，一共有4个班，每班32人。"
                    "如果每辆大巴车限坐45人，那么至少需要多少辆大巴车？"
                ),
                final_answer="3",
                wrong_answers=["128"],
                correct_answer="3",
            )
        ],
        session_runner=LLMSessionRunner(llm_func=fake_llm),
    )

    assert report.total == 1
    assert report.completed_session_count == 1
    assert report.learning_deposit_count == 1
    assert report.parent_feedback_count == 1
    assert report.practice_generated_count == 1
    assert report.practice_quality_failure_count == 0
    assert report.answer_leakage_count == 0
    assert report.model_fallback_count == 0
    assert report.success_rate == 1.0
    assert len(prompts) == 2


def test_graph_evaluation_counts_placeholder_practice_as_failure() -> None:
    report = evaluate_math_tutor_graph(
        cases=[
            GraphEvaluationCase(
                question_id="graph_arithmetic_001",
                grade=3,
                question_text="4*3+2*2=?",
                final_answer="16",
                correct_answer="16",
            )
        ]
    )

    assert report.total == 1
    assert report.completed_session_count == 1
    assert report.practice_generated_count == 1
    assert report.practice_quality_failure_count == 0
    assert report.failed_case_ids == []

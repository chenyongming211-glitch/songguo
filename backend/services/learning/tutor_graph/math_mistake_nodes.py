from __future__ import annotations

import re

from songguo.backend.services.learning.ai_engine import (
    AIEngineContext,
    DeterministicFallbackProvider,
)
from songguo.backend.services.learning.answer_matching import answers_match
from songguo.backend.services.learning.math_structuring import (
    ProblemAnalysis,
    reliable_final_answer,
)
from songguo.backend.services.learning.tutor_graph.state import (
    MathMistakeTutorGraphState,
    RuleJudgeResult,
)


def input_normalize_node(state: MathMistakeTutorGraphState) -> MathMistakeTutorGraphState:
    question_text = " ".join(state.question_text.split())
    return state.model_copy(update={"question_text": question_text, "input_type": "text"})


def ocr_or_text_confirm_node(state: MathMistakeTutorGraphState) -> MathMistakeTutorGraphState:
    if state.input_type != "image":
        return state
    ocr_result = state.ocr_result or {}
    confidence = float(ocr_result.get("confidence") or 0)
    if confidence < 0.8:
        return state.model_copy(
            update={"ocr_result": {**ocr_result, "needs_human_confirm": True}}
        )
    text = str(ocr_result.get("text") or state.question_text).strip()
    return state.model_copy(update={"question_text": text})


def math_problem_parse_node(state: MathMistakeTutorGraphState) -> MathMistakeTutorGraphState:
    analysis = DeterministicFallbackProvider().structure_math_problem(
        question_text=state.question_text,
        grade=state.grade,
        context=AIEngineContext(child_id=state.child_id, session_id=state.session_id),
    )
    return state.model_copy(update={"problem_analysis": analysis.model_dump(mode="json")})


def rule_judge_node(
    state: MathMistakeTutorGraphState,
    *,
    child_answer: str,
) -> MathMistakeTutorGraphState:
    if not state.problem_analysis:
        return state.model_copy(
            update={"rule_judge_result": RuleJudgeResult().model_dump(mode="json")}
        )

    analysis = ProblemAnalysis.model_validate(state.problem_analysis)
    expected_answer = reliable_final_answer(analysis)
    normalized_child = _normalize_answer(child_answer)
    if _answer_matches_expected(child_answer, expected_answer):
        result = RuleJudgeResult(
            determined=True,
            correct=True,
            expected_answer=expected_answer,
            evidence="孩子提交的答案与后端确定性答案一致。",
        )
        return state.model_copy(update={"rule_judge_result": result.model_dump(mode="json")})

    partial = _partial_match(analysis, normalized_child)
    if partial:
        misconception = _partial_misconception_for_key_point(analysis, partial, normalized_child)
        result = RuleJudgeResult(
            determined=False,
            correct=False,
            partially_correct=True,
            expected_answer=expected_answer,
            misconception_tag=misconception,
            evidence="孩子答对了中间步骤，但还没有完成最终问题。",
            matched_key_point_id=partial,
            next_key_point_id=_next_key_point_id(analysis, partial),
        )
        return state.model_copy(update={"rule_judge_result": result.model_dump(mode="json")})

    misconception = _misconception_for_wrong_answer(analysis, normalized_child)
    result = RuleJudgeResult(
        determined=misconception is not None,
        correct=False,
        partially_correct=False,
        expected_answer=expected_answer,
        misconception_tag=misconception,
        evidence=(
            "孩子提交了常见错误答案，后端可以确定错因。"
            if misconception
            else "后端无法确定孩子答案，需要模型继续追问。"
        ),
    )
    return state.model_copy(update={"rule_judge_result": result.model_dump(mode="json")})


def _partial_match(analysis: ProblemAnalysis, normalized_child: str) -> str | None:
    if not normalized_child:
        return None
    for step in analysis.solution_steps:
        result = _normalize_answer(step.result)
        if result and normalized_child == result:
            if analysis.key_points:
                return analysis.key_points[0].id
            return step.id
    for key_point in analysis.key_points:
        for expected in key_point.expected_child_response:
            if normalized_child == _normalize_answer(expected):
                return key_point.id
    return None


def _next_key_point_id(analysis: ProblemAnalysis, matched: str) -> str | None:
    next_key_point = analysis.next_key_point_after(matched)
    return next_key_point.id if next_key_point else None


def _partial_misconception_for_key_point(
    analysis: ProblemAnalysis,
    key_point_id: str,
    normalized_child: str,
) -> str | None:
    try:
        key_point = analysis.get_key_point(key_point_id)
    except (KeyError, ValueError):
        return None
    for tag, responses in key_point.misconception_responses.items():
        if any(normalized_child == _normalize_answer(response) for response in responses):
            return tag
    return key_point.partial_misconception_tag


def _misconception_for_wrong_answer(
    analysis: ProblemAnalysis,
    normalized_child: str,
) -> str | None:
    if analysis.problem_type == "capacity_round_up":
        quotient_result = ""
        for step in analysis.solution_steps:
            if step.id == "step_divide_capacity":
                quotient_result = step.result
                break
        quotient = re.match(r"(\d+)", quotient_result or "")
        if quotient and normalized_child == quotient.group(1):
            return "math_capacity_ignored_remainder_round_up"
    if analysis.problem_type in {"remainder_division", "division_with_remainder"}:
        expected_numbers = re.findall(r"\d+", reliable_final_answer(analysis) or "")
        child_numbers = re.findall(r"\d+", normalized_child)
        if len(expected_numbers) >= 2 and len(child_numbers) == 1:
            expected_quotient = int(expected_numbers[0])
            child_quotient = int(child_numbers[0])
            if child_quotient > expected_quotient:
                return "math_division_quotient_too_large"
            if child_quotient < expected_quotient:
                return "math_division_quotient_too_small"
            return "math_division_missing_remainder"
    return None


def _normalize_answer(value: str | None) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = text.replace("辆", "").replace("人", "").replace("车", "")
    text = text.replace(" ", "")
    text = text.replace("×", "x")
    return text


def _answer_matches_expected(child_answer: str, expected_answer: str | None) -> bool:
    return answers_match(child_answer, expected_answer)

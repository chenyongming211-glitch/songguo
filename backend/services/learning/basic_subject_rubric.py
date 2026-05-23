from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from songguo.backend.services.learning.llm_session_runner import _extract_json_object
from songguo.backend.services.learning.tutor_graph.basic_subject_graph import (
    knowledge_point_for_basic_subject,
)


RUBRIC_VERSION = "basic_subject_rubric_v0.1"
RUBRIC_OUTCOMES = {"correct", "partial", "wrong", "needs_manual_confirm"}
CANONICAL_MISCONCEPTION_TAGS = {
    "english_past_tense_missing",
    "english_form_uncertain",
    "english_answer_key_missing",
    "chinese_answer_missing",
    "chinese_evidence_missing",
    "chinese_answer_key_missing",
    "general_needs_evidence",
}


class BasicSubjectRubricContext(BaseModel):
    subject: str = "general"
    task_type: str = "unknown"
    grade: int = 3
    question_text: str
    child_answer: str
    route_to: str = ""


class BasicSubjectRubricResult(BaseModel):
    outcome: str = "needs_manual_confirm"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    question_type_id: str = "unknown"
    knowledge_point: str = "general_learning_strategy"
    misconception_tag: str | None = None
    feedback_summary: str = ""
    criteria_evidence: list[str] = Field(default_factory=list)
    rubric_scores: dict[str, int] = Field(default_factory=dict)
    provider: str = "basic_subject_rubric"
    model: str = "deterministic"
    source: str = "deterministic"
    rubric_version: str = RUBRIC_VERSION

    @property
    def is_correct(self) -> bool:
        return self.outcome == "correct"


class BasicSubjectRubricEvaluator:
    """Rubric judge for non-math homework submissions.

    The LLM path is used when injected/configured. The deterministic fallback is
    intentionally narrow: it only makes high-confidence calls for common private
    test cases, otherwise it records partial/wrong feedback and lets the tutor
    continue without revealing a final answer.
    """

    def __init__(
        self,
        *,
        evaluator_func: Callable[[BasicSubjectRubricContext], dict[str, Any] | BasicSubjectRubricResult]
        | None = None,
        llm_client: Any | None = None,
        rubric_version: str = RUBRIC_VERSION,
        fast_path_enabled: bool = False,
        fast_path_min_confidence: float = 0.82,
    ) -> None:
        self.evaluator_func = evaluator_func
        self.llm_client = llm_client
        self.rubric_version = rubric_version
        self.fast_path_enabled = fast_path_enabled
        self.fast_path_min_confidence = fast_path_min_confidence

    def evaluate(self, context: BasicSubjectRubricContext) -> BasicSubjectRubricResult:
        if not context.question_text.strip() or not context.child_answer.strip():
            return _needs_manual_result(context, reason="missing_question_or_answer")
        try:
            if self.evaluator_func is not None:
                raw = self.evaluator_func(context)
                result = raw if isinstance(raw, BasicSubjectRubricResult) else BasicSubjectRubricResult.model_validate(raw)
                return _normalize_result(result, context=context, rubric_version=self.rubric_version)
            if self.fast_path_enabled:
                fast_path = _fallback_evaluate(context)
                if (
                    fast_path.outcome in {"correct", "wrong"}
                    and fast_path.confidence >= self.fast_path_min_confidence
                ):
                    return fast_path.model_copy(
                        update={
                            "provider": "basic_subject_rubric_fast_path",
                            "model": "deterministic",
                            "source": "fast_path",
                        }
                    )
            if self.llm_client is not None:
                return self._evaluate_with_llm(context)
        except Exception as exc:
            fallback = _fallback_evaluate(context)
            return fallback.model_copy(
                update={
                    "source": "deterministic_fallback_after_error",
                    "criteria_evidence": [
                        *fallback.criteria_evidence,
                        f"rubric_provider_error:{exc.__class__.__name__}",
                    ],
                }
            )
        return _fallback_evaluate(context)

    def _evaluate_with_llm(self, context: BasicSubjectRubricContext) -> BasicSubjectRubricResult:
        prompt = build_basic_subject_rubric_prompt(context)
        raw = self.llm_client.complete_sync(
            prompt,
            system_prompt=(
                "你是松果AI的语文/英语作业 rubric 判分智能体。"
                "只输出 JSON，不要输出 Markdown、解释或代码块。"
            ),
            temperature=0,
        )
        payload = _extract_json_object(raw)
        payload.setdefault("provider", getattr(self.llm_client, "provider", "llm"))
        payload.setdefault("model", getattr(self.llm_client, "model", "configured"))
        payload.setdefault("source", "llm")
        result = BasicSubjectRubricResult.model_validate(payload)
        return _normalize_result(result, context=context, rubric_version=self.rubric_version)


def build_basic_subject_rubric_prompt(context: BasicSubjectRubricContext) -> str:
    payload = {
        "task": "judge_language_homework_with_rubric",
        "rules": [
            "只判断孩子答案是否满足题目要求，并输出 rubric 证据。",
            "不要讲解题目，不要生成陪练话术，不要输出标准答案或改正后的完整答案。",
            "如果题干或答案不足以判断，outcome=needs_manual_confirm。",
            "correct 表示可以直接记录掌握证据；partial/wrong 表示需要进入陪练。",
            "feedback_summary 只能描述判断依据和下一步关注点，不能泄露标准答案。",
            "rubric_scores 每项使用 0、1、2 三档分值。",
        ],
        "allowed_values": {
            "outcome": ["correct", "partial", "wrong", "needs_manual_confirm"],
            "question_type_id": [
                "reading_comprehension",
                "sentence_rewrite",
                "translation",
                "grammar_fix",
                "composition_fragment",
                "general_learning_strategy",
                "unknown",
            ],
        },
        "homework_context": context.model_dump(mode="json"),
        "required_json_schema": {
            "outcome": "correct|partial|wrong|needs_manual_confirm",
            "confidence": "0.0-1.0",
            "question_type_id": "题型枚举",
            "knowledge_point": "english_sentence_pattern|chinese_reading_summary|general_learning_strategy",
            "misconception_tag": "具体薄弱点标签，正确时可为 null",
            "feedback_summary": "面向系统和家长的简短判分依据，不要输出标准答案",
            "criteria_evidence": ["简短证据"],
            "rubric_scores": {"task_alignment": "0|1|2"},
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def _fallback_evaluate(context: BasicSubjectRubricContext) -> BasicSubjectRubricResult:
    subject = _normalize_subject(context.subject)
    if subject == "english":
        return _fallback_english(context)
    if subject == "chinese":
        return _fallback_chinese(context)
    return BasicSubjectRubricResult(
        outcome="partial",
        confidence=0.55,
        question_type_id=context.task_type or "general_learning_strategy",
        knowledge_point=knowledge_point_for_basic_subject(subject),
        misconception_tag="general_needs_evidence",
        feedback_summary="答案需要进一步说明判断依据，先进入基础陪练确认思路。",
        criteria_evidence=["通用学习问题缺少可判定 rubric。"],
        rubric_scores={"task_alignment": 1, "answer_completeness": 1},
        source="deterministic",
    )


def _fallback_english(context: BasicSubjectRubricContext) -> BasicSubjectRubricResult:
    answer = _normalize_answer(context.child_answer)
    expected = _infer_english_expected_answer(context)
    if expected is None and _looks_like_objective_language_answer(context.child_answer):
        return _needs_manual_result(context, reason="answer_key_missing")
    scores = {"task_alignment": 1, "form_accuracy": 0, "answer_completeness": 1}
    if expected and answer == expected:
        scores["form_accuracy"] = 2
        return BasicSubjectRubricResult(
            outcome="correct",
            confidence=0.92,
            question_type_id=context.task_type or "grammar_fix",
            knowledge_point="english_sentence_pattern",
            feedback_summary="孩子能根据题干时间线索选择匹配的动词形式。",
            criteria_evidence=["作答和题干时间线索一致。"],
            rubric_scores=scores,
            source="deterministic",
        )
    if expected and answer:
        return BasicSubjectRubricResult(
            outcome="wrong",
            confidence=0.88,
            question_type_id=context.task_type or "grammar_fix",
            knowledge_point="english_sentence_pattern",
            misconception_tag="english_past_tense_missing",
            feedback_summary="答案没有随题干时间线索调整动词形式，需要先找时间线索。",
            criteria_evidence=["作答和题干时间线索不一致。"],
            rubric_scores=scores,
            source="deterministic",
        )
    return BasicSubjectRubricResult(
        outcome="partial",
        confidence=0.62,
        question_type_id=context.task_type or "grammar_fix",
        knowledge_point="english_sentence_pattern",
        misconception_tag="english_form_uncertain",
        feedback_summary="答案形式还需要结合题干线索确认，先进入陪练核对依据。",
        criteria_evidence=["题干和答案不足以稳定判定具体语法点。"],
        rubric_scores={"task_alignment": 1, "form_accuracy": 1, "answer_completeness": 1},
        source="deterministic",
    )


def _fallback_chinese(context: BasicSubjectRubricContext) -> BasicSubjectRubricResult:
    answer = context.child_answer.strip()
    if _looks_like_objective_language_answer(answer):
        return _needs_manual_result(context, reason="answer_key_missing")
    if _is_unclear_answer(answer):
        return BasicSubjectRubricResult(
            outcome="wrong",
            confidence=0.86,
            question_type_id=context.task_type or "reading_comprehension",
            knowledge_point="chinese_reading_summary",
            misconception_tag="chinese_answer_missing",
            feedback_summary="孩子没有给出可判定的阅读理解答案，需要先回到题目要求。",
            criteria_evidence=["答案为空、不会或无法对应题目要求。"],
            rubric_scores={"task_alignment": 0, "evidence_detail": 0, "answer_completeness": 0},
            source="deterministic",
        )
    task_alignment = 2 if _answer_matches_chinese_question_type(context.question_text, answer) else 1
    evidence_detail = 2 if len(answer) >= 14 else 1 if len(answer) >= 6 else 0
    answer_completeness = 2 if len(answer) >= 10 else 1 if len(answer) >= 4 else 0
    scores = {
        "task_alignment": task_alignment,
        "evidence_detail": evidence_detail,
        "answer_completeness": answer_completeness,
    }
    if task_alignment >= 1 and evidence_detail >= 1 and answer_completeness >= 1:
        return BasicSubjectRubricResult(
            outcome="correct",
            confidence=0.82,
            question_type_id=context.task_type or "reading_comprehension",
            knowledge_point="chinese_reading_summary",
            feedback_summary="孩子能回应题目要求，并给出基本原因或依据。",
            criteria_evidence=["答案围绕题目要求展开，表达具备基本完整性。"],
            rubric_scores=scores,
            source="deterministic",
        )
    return BasicSubjectRubricResult(
        outcome="partial",
        confidence=0.72,
        question_type_id=context.task_type or "reading_comprehension",
        knowledge_point="chinese_reading_summary",
        misconception_tag="chinese_evidence_missing",
        feedback_summary="答案有回应方向，但依据或表达不够完整，需要继续追问。",
        criteria_evidence=["答案较短或缺少清楚的原因/依据。"],
        rubric_scores=scores,
        source="deterministic",
    )


def _normalize_result(
    result: BasicSubjectRubricResult,
    *,
    context: BasicSubjectRubricContext,
    rubric_version: str,
) -> BasicSubjectRubricResult:
    subject = _normalize_subject(context.subject)
    outcome = result.outcome if result.outcome in RUBRIC_OUTCOMES else "needs_manual_confirm"
    question_type_id = (result.question_type_id or context.task_type or "unknown").strip().lower()
    knowledge_point = result.knowledge_point or knowledge_point_for_basic_subject(subject)
    misconception_tag = _normalize_misconception_tag(
        result.misconception_tag,
        outcome=outcome,
        subject=subject,
        question_type_id=question_type_id,
        feedback_summary=result.feedback_summary,
        criteria_evidence=result.criteria_evidence,
    )
    return result.model_copy(
        update={
            "outcome": outcome,
            "confidence": max(0.0, min(1.0, result.confidence)),
            "question_type_id": question_type_id,
            "knowledge_point": knowledge_point,
            "misconception_tag": misconception_tag,
            "feedback_summary": _clean_feedback(result.feedback_summary),
            "criteria_evidence": [_clean_feedback(item) for item in result.criteria_evidence if item],
            "rubric_scores": _normalize_scores(result.rubric_scores),
            "rubric_version": rubric_version,
        }
    )


def _needs_manual_result(context: BasicSubjectRubricContext, *, reason: str) -> BasicSubjectRubricResult:
    subject = _normalize_subject(context.subject)
    return BasicSubjectRubricResult(
        outcome="needs_manual_confirm",
        confidence=0.0,
        question_type_id=context.task_type or "unknown",
        knowledge_point=knowledge_point_for_basic_subject(subject),
        misconception_tag=f"{subject}_{reason}",
        feedback_summary="题目或孩子答案不完整，需要先人工确认。",
        criteria_evidence=[reason],
        rubric_scores={"task_alignment": 0, "answer_completeness": 0},
    )


def _infer_english_expected_answer(context: BasicSubjectRubricContext) -> str | None:
    question = context.question_text.lower()
    answer = _normalize_answer(context.child_answer)
    if re.search(r"\b(yesterday|last\s+\w+|ago)\b", question):
        if answer in {"go", "goes", "went"} or " go " in f" {question} ":
            return "went"
        if answer in {"is", "am", "are", "was", "were"}:
            return "was"
    return None


def _answer_matches_chinese_question_type(question: str, answer: str) -> bool:
    if any(token in question for token in ("为什么", "原因", "为何")):
        return any(token in answer for token in ("因为", "由于", "为了", "所以", "想"))
    if any(token in question for token in ("概括", "主要内容", "中心")):
        return len(answer) >= 8
    return len(answer) >= 4


def _is_unclear_answer(answer: str) -> bool:
    normalized = answer.strip()
    return normalized in {"", "不知道", "不会", "不懂", "没写", "无"} or len(normalized) <= 1


def _looks_like_objective_language_answer(answer: str) -> bool:
    return bool(re.fullmatch(r"[A-Ea-eTtFf]|[√✓×xX]|对|错", (answer or "").strip()))


def _normalize_subject(subject: str | None) -> str:
    normalized = (subject or "").strip().lower()
    if normalized in {"english", "chinese"}:
        return normalized
    return "general"


def _normalize_answer(value: str) -> str:
    return re.sub(r"[\s。！？!?,，.]+", "", value.strip().lower())


def _normalize_scores(scores: dict[str, int]) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for key, value in (scores or {}).items():
        try:
            normalized[str(key)] = max(0, min(2, int(value)))
        except (TypeError, ValueError):
            normalized[str(key)] = 0
    return normalized


def _normalize_misconception_tag(
    tag: str | None,
    *,
    outcome: str,
    subject: str,
    question_type_id: str,
    feedback_summary: str,
    criteria_evidence: list[str],
) -> str | None:
    if outcome == "correct":
        return None
    candidate = str(tag or "").strip().lower()
    if candidate in CANONICAL_MISCONCEPTION_TAGS:
        return candidate
    joined = " ".join([feedback_summary, *criteria_evidence]).lower()
    if subject == "english" and question_type_id == "grammar_fix":
        if any(token in joined for token in ("yesterday", "past", "tense", "过去", "时态")):
            return "english_past_tense_missing"
        return "english_form_uncertain"
    if subject == "chinese" and question_type_id == "reading_comprehension":
        if any(token in joined for token in ("空", "不知道", "不会", "缺失", "missing")):
            return "chinese_answer_missing"
        return "chinese_evidence_missing"
    return f"{subject}_needs_rubric_review"


def _clean_feedback(value: str) -> str:
    text = re.sub(r"```(?:json)?|```", "", str(value or "")).strip()
    text = re.sub(r"\s+", " ", text)
    return text[:180]

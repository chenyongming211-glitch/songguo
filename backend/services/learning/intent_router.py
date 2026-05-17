from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from songguo.backend.services.learning.llm_session_runner import _extract_json_object


ROUTER_VERSION = "intent_router_agent_v0.1"
CLARIFICATION_ROUTE = "clarification_tutor"
SUBJECT_ROUTES = {
    "math": "math_mistake_tutor",
    "chinese": "chinese_basic_tutor",
    "english": "english_basic_tutor",
    "unknown": CLARIFICATION_ROUTE,
}


class IntentRouterContext(BaseModel):
    child_id: str = ""
    subject_hint: str = "auto"
    grade: int = 3
    source_type: str = "text"
    raw_text: str = ""
    question_text: str = ""
    child_answer: str | None = None
    work_steps_or_context: str = ""
    input_confidence: float = 0.0


class IntentRoutingDecision(BaseModel):
    subject: str = "unknown"
    task_type: str = "unknown"
    user_intent: str = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    route_to: str = CLARIFICATION_ROUTE
    router_version: str = ROUTER_VERSION
    guard_reason: str = ""
    provider: str = "intent_router"
    model: str = "deterministic"
    source: str = "deterministic"


class IntentRouterAgent:
    """Agent-facing intent router.

    The optional agent_func is the production hook for an LLM/agent call. The
    local fallback keeps tests and offline development deterministic; RouterGuard
    remains responsible for refusing uncertain or contradictory output.
    """

    def __init__(
        self,
        *,
        agent_func: Callable[[IntentRouterContext], dict[str, Any] | IntentRoutingDecision] | None = None,
        llm_client: Any | None = None,
        router_version: str = ROUTER_VERSION,
        fast_path_enabled: bool = False,
        fast_path_min_confidence: float = 0.78,
    ) -> None:
        self.agent_func = agent_func
        self.llm_client = llm_client
        self.router_version = router_version
        self.fast_path_enabled = fast_path_enabled
        self.fast_path_min_confidence = fast_path_min_confidence

    def route(self, context: IntentRouterContext) -> IntentRoutingDecision:
        raw = self._route_raw(context)
        decision = raw if isinstance(raw, IntentRoutingDecision) else IntentRoutingDecision.model_validate(raw)
        if not decision.router_version:
            decision = decision.model_copy(update={"router_version": self.router_version})
        return _normalize_decision(decision, router_version=self.router_version)

    def _route_raw(self, context: IntentRouterContext) -> dict[str, Any] | IntentRoutingDecision:
        if self.agent_func is not None:
            return self.agent_func(context)
        if self.fast_path_enabled:
            fast_path = IntentRoutingDecision.model_validate(_fallback_agent_decision(context))
            if fast_path.subject != "unknown" and fast_path.confidence >= self.fast_path_min_confidence:
                return fast_path.model_copy(
                    update={
                        "provider": "intent_router_fast_path",
                        "model": "deterministic",
                        "source": "fast_path",
                    }
                )
        if self.llm_client is not None:
            return self._route_with_llm(context)
        return _fallback_agent_decision(context)

    def _route_with_llm(self, context: IntentRouterContext) -> dict[str, Any]:
        prompt = build_intent_router_prompt(context)
        raw = self.llm_client.complete_sync(
            prompt,
            system_prompt=(
                "你是松果AI的作业意图识别智能体。只输出 JSON，不要输出 Markdown、解释或代码块。"
            ),
            temperature=0,
        )
        payload = _extract_json_object(raw)
        payload["evidence"] = _safe_model_evidence(payload)
        payload.setdefault("router_version", self.router_version)
        payload.setdefault("provider", getattr(self.llm_client, "provider", "llm"))
        payload.setdefault("model", getattr(self.llm_client, "model", "configured"))
        payload.setdefault("source", "llm")
        return payload


def build_intent_router_prompt(context: IntentRouterContext) -> str:
    payload = {
        "task": "route_homework_intent",
        "rules": [
            "你只负责识别学科、题型、孩子意图和建议路由。",
            "不要生成讲解，不要判最终答案，不要进入教学。",
            "evidence 只能写学科/题型/意图判断依据，不能包含正确答案、改正后的答案、解题步骤或评分结论。",
            "如果题面、孩子答案或 OCR/ASR 结果不足，needs_clarification=true。",
            "置信度低于 0.65 时不要硬猜，subject=unknown 且 route_to=clarification_tutor。",
            "顶层必须是一个 JSON object，字段缺失时使用 unknown、空数组、false 或 0。",
        ],
        "allowed_values": {
            "subject": ["math", "chinese", "english", "unknown"],
            "task_type": [
                "calculation",
                "word_problem",
                "reading_comprehension",
                "sentence_rewrite",
                "translation",
                "grammar_fix",
                "composition_fragment",
                "unknown",
            ],
            "user_intent": [
                "check_answer",
                "explain_problem",
                "revise_expression",
                "translate",
                "clarify",
                "unknown",
            ],
            "route_to": [
                "math_mistake_tutor",
                "chinese_basic_tutor",
                "english_basic_tutor",
                "clarification_tutor",
            ],
        },
        "homework_context": context.model_dump(mode="json"),
        "required_json_schema": {
            "subject": "math|chinese|english|unknown",
            "task_type": "题型枚举",
            "user_intent": "孩子或家长提交这次作业的真实意图",
            "confidence": "0.0-1.0",
            "evidence": ["简短识别依据"],
            "needs_clarification": "boolean",
            "route_to": "建议进入哪个 TutorGraph",
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def _safe_model_evidence(payload: dict[str, Any]) -> list[str]:
    subject_labels = {
        "math": "数学",
        "chinese": "语文",
        "english": "英语",
        "unknown": "未知学科",
    }
    task_labels = {
        "calculation": "计算题",
        "word_problem": "应用题",
        "reading_comprehension": "阅读理解",
        "sentence_rewrite": "句子改写",
        "translation": "翻译",
        "grammar_fix": "语法改错",
        "composition_fragment": "作文片段",
        "unknown": "未知题型",
    }
    intent_labels = {
        "check_answer": "检查答案",
        "explain_problem": "讲解题目",
        "revise_expression": "修改表达",
        "translate": "翻译",
        "clarify": "补充信息",
        "unknown": "未知意图",
    }
    subject = str(payload.get("subject") or "unknown").lower()
    task_type = str(payload.get("task_type") or "unknown").lower()
    user_intent = str(payload.get("user_intent") or "unknown").lower()
    needs_clarification = bool(payload.get("needs_clarification"))
    if subject == "unknown" or needs_clarification:
        return ["模型认为题面、孩子作答或识别结果不足，需要先追问确认。"]
    return [
        (
            f"模型识别为{subject_labels.get(subject, '未知学科')}/"
            f"{task_labels.get(task_type, '未知题型')}，"
            f"意图是{intent_labels.get(user_intent, '未知意图')}；"
            "依据来自题干语言、任务要求和作答形式，不包含答案细节。"
        )
    ]


class RouterGuard:
    def __init__(self, *, min_confidence: float = 0.65) -> None:
        self.min_confidence = min_confidence

    def apply(self, decision: IntentRoutingDecision, *, raw_text: str = "") -> IntentRoutingDecision:
        normalized = _normalize_decision(decision)
        reasons: list[str] = []
        if normalized.confidence < self.min_confidence:
            reasons.append("low_confidence")
        if normalized.subject == "unknown" or normalized.route_to == CLARIFICATION_ROUTE:
            reasons.append("unknown_route")
        if not raw_text.strip():
            reasons.append("empty_input")
        if _has_obvious_english_math_conflict(normalized, raw_text):
            reasons.append("obvious_english_conflict")

        if not reasons and not normalized.needs_clarification:
            return normalized

        return normalized.model_copy(
            update={
                "subject": "unknown",
                "task_type": "unknown" if reasons else normalized.task_type,
                "route_to": CLARIFICATION_ROUTE,
                "needs_clarification": True,
                "guard_reason": "|".join(dict.fromkeys(reasons)) or normalized.guard_reason,
            }
        )


def _normalize_decision(
    decision: IntentRoutingDecision,
    *,
    router_version: str | None = None,
) -> IntentRoutingDecision:
    subject = (decision.subject or "unknown").strip().lower()
    if subject not in SUBJECT_ROUTES:
        subject = "unknown"
    route_to = (decision.route_to or SUBJECT_ROUTES[subject]).strip()
    expected_route = SUBJECT_ROUTES[subject]
    if subject != "unknown" and route_to not in set(SUBJECT_ROUTES.values()):
        route_to = expected_route
    if subject == "unknown":
        route_to = CLARIFICATION_ROUTE
    return decision.model_copy(
        update={
            "subject": subject,
            "task_type": (decision.task_type or "unknown").strip().lower(),
            "user_intent": (decision.user_intent or "unknown").strip().lower(),
            "route_to": route_to,
            "router_version": router_version or decision.router_version or ROUTER_VERSION,
        }
    )


def _fallback_agent_decision(context: IntentRouterContext) -> dict[str, Any]:
    text = " ".join(
        item
        for item in [context.raw_text, context.question_text, context.child_answer or ""]
        if item
    )
    if _looks_english(text):
        task_type = "translation" if "翻译" in text or "translate" in text.lower() else "grammar_fix"
        return _decision("english", task_type, "check_answer", 0.78, "题面包含英文句子或语法任务")
    if _looks_chinese_homework(text):
        task_type = "reading_comprehension"
        if "造句" in text:
            task_type = "sentence_rewrite"
        elif "修改病句" in text:
            task_type = "sentence_rewrite"
        return _decision("chinese", task_type, "check_answer", 0.82, "题面包含语文阅读或表达任务")
    if _looks_math(text):
        task_type = "calculation" if re.search(r"[=＝+\-×*÷/]", text) else "word_problem"
        return _decision("math", task_type, "check_answer", 0.78, "题面包含算式、数字或数量关系")
    return _decision("unknown", "unknown", "unknown", 0.3, "题面信息不足")


def _decision(
    subject: str,
    task_type: str,
    user_intent: str,
    confidence: float,
    evidence: str,
) -> dict[str, Any]:
    return {
        "subject": subject,
        "task_type": task_type,
        "user_intent": user_intent,
        "confidence": confidence,
        "evidence": [evidence],
        "needs_clarification": subject == "unknown",
        "route_to": SUBJECT_ROUTES[subject],
        "router_version": ROUTER_VERSION,
        "provider": "intent_router",
        "model": "deterministic",
        "source": "deterministic",
    }


def _looks_english(text: str) -> bool:
    return bool(re.search(r"\b(?:choose|correct|tense|translate|grammar|sentence|word)\b", text, re.I)) or bool(
        re.search(r"[A-Za-z]{3,}", text)
    )


def _looks_chinese_homework(text: str) -> bool:
    return any(
        token in text
        for token in ("阅读短文", "造句", "修改病句", "中心思想", "解释词语", "作文", "按要求写句子")
    ) or any(
        token in text
        for token in ("为什么", "为何", "原因", "概括", "主要内容", "中心思想")
    )


def _looks_math(text: str) -> bool:
    return bool(re.search(r"\d+.*(?:几|多少|求|计算|平均|一共|剩|米|厘米|元|辆)", text)) or bool(
        re.search(r"\d+\s*[=＝+\-×*÷/]\s*\d*", text)
    )


def _has_obvious_english_math_conflict(decision: IntentRoutingDecision, raw_text: str) -> bool:
    if decision.subject != "math":
        return False
    return _looks_english(raw_text) and not _looks_math(raw_text)

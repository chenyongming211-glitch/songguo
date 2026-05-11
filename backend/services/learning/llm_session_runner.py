from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from songguo.backend.services.learning.context_pack import StudentContextPack
from songguo.backend.services.learning.models import LearningMessage
from songguo.backend.services.learning.practice_recommender import PracticeItem


class StructuredState(BaseModel):
    phase: str = "WAIT_CHILD_ATTEMPT"
    answer_unlocked: bool = False
    current_key_point: str | None = None
    mastered_key_points: list[str] = Field(default_factory=list)
    main_misconception: str | None = None
    hint_dependency_delta: int = 0
    should_end_session: bool = False


class LearningDepositDelta(BaseModel):
    knowledge_point: str = "grade_math_unknown"
    question_type: str = "grade_math_unknown"
    is_correct: bool | None = None
    main_misconception: str | None = None
    evidence: str = ""
    need_review: bool = True
    parent_summary: str = ""


class TeachingIntent(BaseModel):
    learner_readiness: str = "ready"
    child_answer_status: str = "unknown"
    teacher_move: str = "ask_diagnostic_question"
    should_reveal_final_answer: bool = False
    next_question: str = ""


class ResponseSelfCheck(BaseModel):
    praised_wrong_answer: bool = False
    revealed_final_answer: bool = False
    one_question_only: bool = True
    directly_solved_multiple_steps: bool = False


class LLMSessionOutput(BaseModel):
    child_message: str
    structured_state: StructuredState = Field(default_factory=StructuredState)
    learning_deposit_delta: LearningDepositDelta = Field(default_factory=LearningDepositDelta)
    practice_items: list[PracticeItem] = Field(default_factory=list)
    teaching_intent: TeachingIntent = Field(default_factory=TeachingIntent)
    self_check: ResponseSelfCheck = Field(default_factory=ResponseSelfCheck)
    provider: str = "llm_session_runner"
    model: str = "unknown"
    token_estimate: int = 0
    status: str = "success"


class LLMSessionRunner:
    def __init__(
        self,
        *,
        llm_func: Callable[[str], str] | None = None,
        provider: str = "direct_llm",
        model: str = "configured",
        timeout_seconds: float = 20.0,
    ) -> None:
        self.llm_func = llm_func
        self.provider = provider
        self.model = model
        self.timeout_seconds = timeout_seconds

    def start(
        self,
        *,
        child_id: str,
        grade: int,
        question_text: str,
        context_pack: StudentContextPack,
    ) -> LLMSessionOutput:
        return self.run(
            child_id=child_id,
            grade=grade,
            context_pack=context_pack,
            messages=[
                {"role": "user", "content": question_text},
            ],
        )

    def run(
        self,
        *,
        child_id: str,
        grade: int,
        context_pack: StudentContextPack,
        messages: list[LearningMessage | dict[str, Any]],
    ) -> LLMSessionOutput:
        prompt = build_llm_session_prompt(
            child_id=child_id,
            grade=grade,
            context_pack=context_pack,
            messages=messages,
        )
        try:
            raw = self._complete(prompt)
            payload = _extract_json_object(raw)
            output = LLMSessionOutput.model_validate(payload)
            output.provider = str(payload.get("provider") or self.provider)
            output.model = str(payload.get("model") or self.model)
            output.token_estimate = output.token_estimate or _estimate_tokens(prompt + raw)
            return output
        except Exception as exc:
            return LLMSessionOutput(
                child_message="我们先慢一点。请先说说题目告诉了哪些条件？",
                structured_state=StructuredState(
                    phase="WAIT_CHILD_ATTEMPT",
                    answer_unlocked=False,
                    current_key_point="读题找条件",
                    should_end_session=False,
                ),
                learning_deposit_delta=LearningDepositDelta(
                    evidence=f"LLM 输出不可用，已使用安全兜底：{exc.__class__.__name__}",
                ),
                provider=self.provider,
                model=self.model,
                token_estimate=_estimate_tokens(prompt),
                status="fallback",
            )

    def _complete(self, prompt: str) -> str:
        if self.timeout_seconds > 0:
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(self._complete_without_timeout, prompt)
            try:
                return future.result(timeout=self.timeout_seconds)
            except FutureTimeoutError as exc:
                future.cancel()
                raise TimeoutError(
                    f"LLM session runner timed out after {self.timeout_seconds:.1f}s"
                ) from exc
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
        return self._complete_without_timeout(prompt)

    def _complete_without_timeout(self, prompt: str) -> str:
        if self.llm_func:
            return self.llm_func(prompt)
        from songguo.backend.services.learning.langchain_model_client import (
            get_langchain_llm_client,
        )

        client = get_langchain_llm_client()
        self.provider = client.provider
        self.model = client.model
        return client.complete_sync(
            prompt,
            system_prompt=(
                "你是松果AI的一题一会话教学运行器。只输出 JSON，不要输出 Markdown。"
            ),
            temperature=0,
        )

    def repair_child_message(
        self,
        *,
        question_text: str,
        draft_text: str,
        reason: str,
        expected_answer: str | None,
    ) -> str | None:
        payload = {
            "repair_request": {
                "task": "rewrite_child_message_without_answer_leakage",
                "question_text": question_text,
                "blocked_child_message": draft_text,
                "safety_reason": reason,
                "expected_answer": expected_answer,
                "rules": [
                    "保留松鼠博士的自然老师语气。",
                    "不能直接或间接说出最终答案。",
                    "必须贴合当前题，不要使用通用兜底话术。",
                    "一次只问一个问题。",
                    "只输出 JSON object。",
                ],
            },
            "output_schema": {
                "child_message": "改写后的安全追问",
            },
        }
        try:
            raw = self._complete(json.dumps(payload, ensure_ascii=False))
            value = _extract_json_object(raw)
        except Exception:
            return None
        message = str(value.get("child_message") or "").strip()
        return message or None

    def run_practice_control(
        self,
        *,
        child_id: str,
        grade: int,
        context_pack: StudentContextPack,
        messages: list[LearningMessage | dict[str, Any]],
        practice_items: list[dict[str, Any]],
        current_phase: str,
    ) -> LLMSessionOutput:
        prompt = build_practice_control_prompt(
            child_id=child_id,
            grade=grade,
            context_pack=context_pack,
            messages=messages,
            practice_items=practice_items,
            current_phase=current_phase,
        )
        try:
            raw = self._complete(prompt)
            payload = _extract_json_object(raw)
            output = LLMSessionOutput.model_validate(payload)
            output.provider = str(payload.get("provider") or self.provider)
            output.model = str(payload.get("model") or self.model)
            output.token_estimate = output.token_estimate or _estimate_tokens(prompt + raw)
            return output
        except Exception as exc:
            return LLMSessionOutput(
                child_message="松鼠博士刚刚没有接上，我们先停在这里。你再发一次，我会接着判断。",
                structured_state=StructuredState(
                    phase=current_phase,
                    answer_unlocked=True,
                    should_end_session=False,
                ),
                learning_deposit_delta=LearningDepositDelta(
                    evidence=f"同类题阶段模型输出不可用：{exc.__class__.__name__}",
                ),
                teaching_intent=TeachingIntent(
                    child_answer_status="unknown",
                    teacher_move="model_unavailable",
                    next_question="",
                ),
                provider=self.provider,
                model=self.model,
                token_estimate=_estimate_tokens(prompt),
                status="fallback",
            )


def build_llm_session_prompt(
    *,
    child_id: str,
    grade: int,
    context_pack: StudentContextPack,
    messages: list[LearningMessage | dict[str, Any]],
) -> str:
    normalized_messages = [_message_to_dict(message) for message in messages]
    latest_child_answer = _latest_user_message(normalized_messages)
    latest_assistant_message = _latest_assistant_message(normalized_messages)
    payload = {
        "system_protocol": {
            "persona": "松鼠博士",
            "grade": grade,
            "rules": [
                "不能肯定错误答案",
                "孩子答错或不确定时，只问一个诊断问题，不要直接完整讲解",
                "可以鼓励孩子愿意尝试，但不能肯定错误答案。",
                "孩子答错或不确定时，只问一个诊断问题，不要直接完整讲解。",
                "每轮 child_message 只推进一个认知动作，只问一个问题。",
                "每轮先判断孩子当前是否适合继续学习，不要只判断数学对错。",
                "如果孩子表达当前身体、情绪、注意力或意愿状态不适合继续学习，输出 learner_readiness=not_ready、teacher_move=pause_learning、phase=LEARNING_PAUSED，并且 child_message 只尊重暂停和说明之后可以继续，不要继续追问题。",
                "answer_unlocked=false 时不能输出最终答案。",
                "如果孩子的回答可能是错的，先让孩子解释来源或核对关系，不要说“很棒”“算得很好”。",
                "不要在 child_message 中连续讲两步以上的完整解法。",
                "基于当前题完整 messages 连续引导。",
                "顶层必须是一个 JSON object。",
                "只输出一个 JSON object，不要输出 Markdown、解释文字或代码块。",
                "不要输出 ```json 代码块，也不要在 JSON 前后添加任何解释文字。",
                "字段缺失时使用空字符串、空数组、false 或 null，不要省略必需字段。",
                "输出必须严格符合 required_json_schema。",
            ],
        },
        "student_context_pack": context_pack.model_dump(mode="json"),
        "current_turn": {
            "latest_child_answer": latest_child_answer,
            "latest_assistant_message": latest_assistant_message,
            "answer_unlocked": False,
            "task": "judge_latest_child_answer_then_ask_one_diagnostic_question",
            "learner_readiness_policy": [
                "不要用关键词机械判断，要根据孩子当前语义判断是否适合继续学习。",
                "ready: 孩子在答题、确认、追问，或愿意继续。",
                "needs_support: 孩子不会、看不懂、没思路，但仍适合继续，可降低难度。",
                "not_ready: 孩子表达身体、情绪、注意力或意愿状态不适合继续；先暂停学习。",
                "safety_risk: 涉及儿童安全、隐私、暴力自伤、色情裸露、违法诱导、陌生人风险。",
            ],
            "teacher_move_priority": [
                "先判断 learner_readiness。",
                "如果 learner_readiness=not_ready，teacher_move 必须是 pause_learning，phase 必须是 LEARNING_PAUSED，child_message 不要继续推进题目。",
                "如果孩子只是要最终答案，learner_readiness=ready，child_answer_status=answer_seeking，teacher_move=refuse_direct_answer，phase=WAIT_CHILD_ATTEMPT；拒绝直接给答案，只引导下一小步。",
                "如果孩子跑题、想玩、闲聊但没有明确暂停学习，learner_readiness=needs_support，child_answer_status=off_task，teacher_move=redirect_to_learning，phase=WAIT_CHILD_ATTEMPT；简短接住后拉回当前题。",
                "如果孩子表示现在可以继续、想接着做，learner_readiness=ready，child_answer_status=resume_request，teacher_move=continue_tutoring，phase=WAIT_CHILD_ATTEMPT；接上 latest_assistant_message 的下一小步。",
                "如果孩子说不会、看不懂、没思路但没有拒绝学习，learner_readiness=needs_support，child_answer_status=confused，teacher_move=simplify，phase=WAIT_CHILD_ATTEMPT。",
                "先判断 latest_child_answer 是 correct、partial、wrong 还是 unclear。",
                "如果 wrong 或 unclear，不要肯定答案，先问一个诊断问题。",
                "如果 partial，只肯定已确认的部分，再问下一小步。",
                "如果 correct，才可以进入总结或同类题。",
            ],
        },
        "anti_patterns": [
            {
                "latest_child_answer": "9乘以2是18颗",
                "bad_child_message": "你说9乘以2是18颗，算得很棒！总数就是30颗。你同意吗？",
                "good_child_message": "我们先核对一下：这个9是怎么来的？第一天吃完后剩下12颗，那“总数的一半多3颗”是吃掉的，还是剩下的？",
                "why": "孩子答错时不能肯定错误，也不能直接给最终答案。",
            }
        ],
        "current_problem_messages": normalized_messages,
        "output_schema": {
            "child_message": "给孩子看的话",
            "structured_state": {
                "phase": "WAIT_CHILD_ATTEMPT | LEARNING_PAUSED | SIMILAR_PRACTICE | SESSION_SUMMARY",
                "answer_unlocked": False,
                "current_key_point": "当前关键点",
                "mastered_key_points": [],
                "main_misconception": None,
                "hint_dependency_delta": 0,
                "should_end_session": False,
            },
            "learning_deposit_delta": {
                "knowledge_point": "标准知识点",
                "question_type": "题型",
                "is_correct": None,
                "main_misconception": None,
                "evidence": "判断依据",
                "need_review": True,
                "parent_summary": "家长可读摘要",
            },
            "teaching_intent": {
                "learner_readiness": "ready | needs_support | not_ready | safety_risk",
                "child_answer_status": "correct | partial | wrong | unclear | math_attempt | confused | answer_seeking | learning_resistance | wellbeing_not_ready | off_task | safety_risk | unknown",
                "teacher_move": "ask_diagnostic_question | ask_next_step | simplify | refuse_direct_answer | pause_learning | redirect_to_learning | continue_tutoring | summarize | generate_practice | safety_response",
                "should_reveal_final_answer": False,
                "next_question": "本轮只问孩子的一个问题",
            },
            "self_check": {
                "praised_wrong_answer": False,
                "revealed_final_answer": False,
                "one_question_only": True,
                "directly_solved_multiple_steps": False,
            },
            "practice_items": [],
        },
        "required_json_schema": {
            "title": "LLMSessionOutput",
            "schema": LLMSessionOutput.model_json_schema(),
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def build_practice_control_prompt(
    *,
    child_id: str,
    grade: int,
    context_pack: StudentContextPack,
    messages: list[LearningMessage | dict[str, Any]],
    practice_items: list[dict[str, Any]],
    current_phase: str,
) -> str:
    normalized_messages = [_message_to_dict(message) for message in messages]
    latest_child_answer = _latest_user_message(normalized_messages)
    recent_messages = _recent_messages_for_practice_control(normalized_messages)
    related_evidence = [
        {
            "knowledge_point": item.knowledge_point_label,
            "misconception": item.misconception_label,
            "evidence": item.evidence,
        }
        for item in context_pack.related_evidence[:1]
    ]
    payload = {
        "task": "similar_practice_control",
        "system_protocol": {
            "persona": "松鼠博士",
            "grade": grade,
            "rules": [
                "判断同类题阶段孩子这句话的会话控制意图。",
                "结合最近对话，不做关键词机械判断。",
                "如果 recent_messages 里的旧同类题和 current_practice_items 不一致，以 current_practice_items 为准。",
                "孩子暂停就自然收住；孩子回来就自然续接。",
                "回复简短，适合小学生。",
                "只输出 JSON object。",
            ],
        },
        "student_context": {
            "child_id": child_id,
            "grade": grade,
            "current_question": context_pack.current_question,
            "short_term_goal": context_pack.short_term_goal,
            "parent_focus": context_pack.parent_settings.focus[:2],
            "related_evidence": related_evidence,
        },
        "current_phase": current_phase,
        "current_practice_items": practice_items[:3],
        "current_turn": {
            "latest_child_answer": latest_child_answer,
            "allowed_teacher_moves": [
                "answer_practice_item",
                "resume_practice",
                "pause_practice",
                "start_new_question",
                "small_talk_or_other",
                "ask_clarification",
            ],
            "state_policy": {
                "answer_practice_item": "SIMILAR_PRACTICE",
                "resume_practice": "SIMILAR_PRACTICE",
                "pause_practice": "PRACTICE_PAUSED",
                "start_new_question": "PRACTICE_PAUSED",
                "small_talk_or_other": "PRACTICE_PAUSED",
                "ask_clarification": "PRACTICE_PAUSED",
            },
        },
        "recent_messages": recent_messages,
        "output_schema": {
            "child_message": "给孩子看的自然回复",
            "structured_state": {
                "phase": "SIMILAR_PRACTICE | PRACTICE_PAUSED | SESSION_SUMMARY",
                "answer_unlocked": True,
                "should_end_session": False,
            },
            "learning_deposit_delta": {
                "is_correct": None,
                "main_misconception": None,
                "evidence": "判断依据",
                "need_review": True,
            },
            "teaching_intent": {
                "child_answer_status": "practice_correct | practice_wrong | pause_request | resume_request | new_question | small_talk | unclear",
                "teacher_move": "answer_practice_item | resume_practice | pause_practice | start_new_question | small_talk_or_other | ask_clarification",
                "next_question": "需要孩子继续回答时，只问一个问题",
            },
            "self_check": {
                "praised_wrong_answer": False,
                "one_question_only": True,
            },
            "practice_items": [],
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def _recent_messages_for_practice_control(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "role": message.get("role", ""),
            "content": _trim_text(message.get("content", ""), limit=180),
        }
        for message in messages[-8:]
    ]


def _trim_text(value: str, *, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _message_to_dict(message: LearningMessage | dict[str, Any]) -> dict[str, str]:
    if isinstance(message, LearningMessage):
        return {"role": message.role, "content": message.content}
    return {
        "role": str(message.get("role") or ""),
        "content": str(message.get("content") or ""),
    }


def _latest_user_message(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def _latest_assistant_message(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return message.get("content", "")
    return ""


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("LLM session runner must return a JSON object")
    return value


def _estimate_tokens(text: str) -> int:
    stripped = text.strip()
    return max(1, len(stripped) // 4) if stripped else 0

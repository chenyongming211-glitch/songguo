from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

from songguo.backend.services.learning.deeptutor_adapter import TeachingDraft


class _LocalStreamEventType(str, Enum):
    CONTENT = "content"
    ERROR = "error"


@dataclass
class _LocalUnifiedContext:
    session_id: str
    user_message: str
    active_capability: str = "chat"
    enabled_tools: list[str] = field(default_factory=list)
    language: str = "zh"
    metadata: dict[str, Any] = field(default_factory=dict)


class OrchestratorDraftProvider:
    """Collect DeepTutor stream output into a structured teaching draft."""

    def __init__(self, *, orchestrator: Any | None = None) -> None:
        self.orchestrator = orchestrator

    def generate(self, payload: dict[str, Any]) -> TeachingDraft:
        return asyncio.run(self._generate_async(payload))

    async def _generate_async(self, payload: dict[str, Any]) -> TeachingDraft:
        orchestrator = self.orchestrator or _load_deeptutor_orchestrator()
        if orchestrator is None:
            return TeachingDraft(
                action=_action_from_prompt(payload),
                hint_level=payload.get("hint_level"),
                exposes_final_answer=False,
                text=_safe_fallback(payload),
                trace_id=f"deeptutor_unavailable_{uuid4().hex}",
                metadata={"provider": "deeptutor_orchestrator", "fallback": True},
            )
        content_parts: list[str] = []
        saw_error = False
        user_message = _build_user_message(payload)
        context = _LocalUnifiedContext(
            session_id=f"learning_{uuid4().hex}",
            user_message=user_message,
            metadata={
                "learning_prompt_id": payload.get("prompt_id", ""),
                "learning_prompt_version": payload.get("prompt_version", ""),
            },
        )
        async for event in orchestrator.handle(context):
            if _event_type_matches(event, "CONTENT") and event.content:
                content_parts.append(event.content)
            if _event_type_matches(event, "ERROR"):
                saw_error = True

        if saw_error or not content_parts:
            return TeachingDraft(
                action=_action_from_prompt(payload),
                hint_level=payload.get("hint_level"),
                exposes_final_answer=False,
                text=_safe_fallback(payload),
                trace_id=f"deeptutor_{context.session_id}",
                metadata={"provider": "orchestrator", "fallback": True},
            )

        return TeachingDraft(
            action=_action_from_prompt(payload),
            hint_level=payload.get("hint_level"),
            exposes_final_answer=_action_from_prompt(payload) == "explanation",
            misconception_tag=payload.get("misconception_tag"),
            text="".join(content_parts).strip(),
            trace_id=f"deeptutor_{context.session_id}",
            metadata={"provider": "orchestrator", "fallback": False},
        )


def _action_from_prompt(payload: dict[str, Any]) -> str:
    prompt_id = str(payload.get("prompt_id", ""))
    if "similar_practice" in prompt_id:
        return "similar_practice"
    if "explanation" in prompt_id:
        return "explanation"
    return "hint"


def _build_user_message(payload: dict[str, Any]) -> str:
    action = _action_from_prompt(payload)
    if action == "hint":
        return (
            "你是三年级数学老师。只给下一步提示，不要给最终答案。\n"
            f"题目：{payload.get('question_text', '')}\n"
            f"当前提示等级：{payload.get('hint_level', 1)}\n"
            f"知识点：{payload.get('knowledge_point', '')}\n"
            f"错因：{payload.get('misconception_tag', '') or '暂无'}"
        )
    if action == "explanation":
        return (
            "你是三年级数学老师。现在允许完整讲解，但要突出易错点。\n"
            f"题目：{payload.get('question_text', '')}"
        )
    return (
        "你是三年级数学老师。生成 1-3 道同类练习题，不要给答案。\n"
        f"知识点：{payload.get('knowledge_point', '')}\n"
        f"错因：{payload.get('misconception_tag', '') or '暂无'}\n"
        f"难度：{payload.get('difficulty', 1)}"
    )


def _safe_fallback(payload: dict[str, Any]) -> str:
    action = _action_from_prompt(payload)
    if action == "similar_practice":
        return "模型暂时不可用，先练一道同类小题：24 × 5 = ?"
    if action == "explanation":
        return "模型暂时不可用，先把题目拆成更小的步骤再看完整讲解。"
    return "模型暂时不可用，我们先不看答案。请先说说你准备从哪一步开始。"


def _load_deeptutor_orchestrator() -> Any | None:
    try:
        from deeptutor.runtime.orchestrator import ChatOrchestrator
    except Exception:
        return None
    return ChatOrchestrator()


def _event_type_matches(event: Any, expected: str) -> bool:
    event_type = getattr(event, "type", None)
    if getattr(event_type, "name", "") == expected:
        return True
    expected_value = getattr(_LocalStreamEventType, expected).value
    return str(event_type).lower().endswith(expected.lower()) or str(event_type) == expected_value

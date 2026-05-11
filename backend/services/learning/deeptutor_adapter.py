from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from songguo.backend.services.learning.labels import knowledge_point_label
from songguo.backend.services.learning.prompt_registry import PromptRegistry


class TeachingDraft(BaseModel):
    action: str
    hint_level: int | None = None
    exposes_final_answer: bool = False
    misconception_tag: str | None = None
    text: str
    trace_id: str = Field(default_factory=lambda: f"trace_{uuid4().hex}")
    metadata: dict[str, Any] = Field(default_factory=dict)


DraftGenerator = Callable[[dict[str, Any]], TeachingDraft]


class DeepTutorLearningAdapter:
    """Controlled seam around DeepTutor generation.

    The default implementation returns deterministic local drafts for P0 tests.
    A live DeepTutor-backed generator can be injected without changing callers.
    """

    def __init__(
        self,
        *,
        registry: PromptRegistry | None = None,
        draft_generator: DraftGenerator | None = None,
    ) -> None:
        self.registry = registry or PromptRegistry()
        self.draft_generator = draft_generator

    def generate_hint(
        self,
        *,
        question_text: str,
        grade: int,
        knowledge_point: str,
        hint_level: int,
        misconception_tag: str | None = None,
    ) -> TeachingDraft:
        template = self.registry.require("grade3_math_hint")
        payload = {
            "prompt_id": template.prompt_id,
            "prompt_version": template.version,
            "question_text": question_text,
            "grade": grade,
            "knowledge_point": knowledge_point,
            "hint_level": hint_level,
            "misconception_tag": misconception_tag,
        }
        if self.draft_generator:
            return self._generate_or_fallback(
                payload,
                fallback=_default_hint_draft(
                    knowledge_point=knowledge_point,
                    hint_level=hint_level,
                    question_text=question_text,
                    misconception_tag=misconception_tag,
                    metadata=payload,
                ),
            )
        return _default_hint_draft(
            knowledge_point=knowledge_point,
            hint_level=hint_level,
            question_text=question_text,
            misconception_tag=misconception_tag,
            metadata=payload,
        )

    def _generate_or_fallback(
        self,
        payload: dict[str, Any],
        *,
        fallback: TeachingDraft,
    ) -> TeachingDraft:
        try:
            if self.draft_generator is None:
                return fallback
            return self.draft_generator(payload)
        except Exception as exc:
            fallback.metadata = {
                **fallback.metadata,
                "fallback_reason": "draft_generator_error",
                "fallback_error": exc.__class__.__name__,
            }
            return fallback

    def generate_explanation(
        self,
        *,
        question_text: str,
        answer_unlocked: bool,
    ) -> TeachingDraft:
        if not answer_unlocked:
            raise PermissionError("Full explanation is not allowed while answer_unlocked=false")
        template = self.registry.require("grade3_math_explanation")
        payload = {
            "prompt_id": template.prompt_id,
            "prompt_version": template.version,
            "question_text": question_text,
        }
        return self._generate_or_fallback(
            payload,
            fallback=TeachingDraft(
                action="explanation",
                exposes_final_answer=True,
                text="现在可以看完整讲解：先拆解题目，再完成计算，并检查易错点。",
                metadata=payload,
            ),
        )

    def generate_similar_practice(
        self,
        *,
        knowledge_point: str,
        misconception_tag: str | None,
        difficulty: int,
    ) -> TeachingDraft:
        template = self.registry.require("grade3_math_similar_practice")
        payload = {
            "prompt_id": template.prompt_id,
            "prompt_version": template.version,
            "knowledge_point": knowledge_point,
            "misconception_tag": misconception_tag,
            "difficulty": difficulty,
        }
        return self._generate_or_fallback(
            payload,
            fallback=TeachingDraft(
                action="similar_practice",
                exposes_final_answer=False,
                misconception_tag=misconception_tag,
                text="生成 1-3 道同类练习，用来确认孩子是否真正掌握。",
                metadata=payload,
            ),
        )


def _default_hint_draft(
    *,
    knowledge_point: str,
    hint_level: int,
    question_text: str,
    misconception_tag: str | None,
    metadata: dict[str, Any],
) -> TeachingDraft:
    return TeachingDraft(
        action="hint",
        hint_level=hint_level,
        exposes_final_answer=False,
        misconception_tag=misconception_tag,
        text=_default_hint_text(knowledge_point, hint_level, question_text),
        metadata=metadata,
    )


def _default_hint_text(knowledge_point: str, hint_level: int, question_text: str) -> str:
    label = knowledge_point_label(knowledge_point)
    if knowledge_point == "english_sentence_pattern":
        return "这是一道英语表达题。先不用急着给完整句子，先说说句子里主语和动作分别是什么。"
    if knowledge_point == "chinese_reading_summary":
        return "这是一道语文阅读题。先不用急着写答案，先找出谁、做了什么、结果是什么。"
    if knowledge_point == "two_digit_times_one_digit" and hint_level <= 1:
        times_five_hint = _multiplication_by_five_hint(question_text)
        if times_five_hint:
            return times_five_hint
        return "先说说你看到这道题后，准备先从哪一步开始？"
    if hint_level <= 1:
        return f"先找一个和{label}有关的简单问题想一想。"
    if hint_level == 2:
        return "你已经开始思考了。现在把题目拆成两步，先完成第一步。"
    if hint_level == 3:
        return "注意刚才的错误模式，换一种拆法再试一次。"
    if hint_level == 4:
        return "现在给你一个更明确的提示，但最后一步还是由你来完成。"
    return "你已经多次尝试了，我们先整理完整思路。"


def _multiplication_by_five_hint(question_text: str) -> str | None:
    normalized = question_text.replace("×", "x").replace("*", "x")
    match = re.search(r"(?<!\d)(\d{1,3})\s*x\s*5(?!\d)", normalized, re.IGNORECASE)
    if not match:
        match = re.search(r"(?<!\d)5\s*x\s*(\d{1,3})(?!\d)", normalized, re.IGNORECASE)
    if not match:
        return None

    factor = int(match.group(1))
    return f"先不急着算最后答案。你先想一想：{factor} × 10 会是多少？乘以 5 和乘以 10 有什么关系？"

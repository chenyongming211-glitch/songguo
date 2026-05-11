from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel

from songguo.backend.services.learning.child_safety import (
    ChildSafetyAction,
    check_child_safety,
)


class LeakageAction(StrEnum):
    ALLOW = "allow"
    BLOCK = "block"


class LeakageVerdict(BaseModel):
    action: LeakageAction
    reason: str
    safe_text: str


class AnswerLeakageSignal(BaseModel):
    detected: bool
    reason: str = "safe"


FORBIDDEN_ANSWER_PHRASES = (
    "the answer is",
    "answer is",
    "so the answer",
    "final answer",
    "答案是",
    "所以答案",
    "最终答案",
    "直接答案",
)


def _fallback_text(hint_level: int) -> str:
    prompts = {
        1: "我们先不急着看答案。先找一个更简单的相关问题想一想。",
        2: "先说说你准备用哪一步开始，我会根据你的想法继续提示。",
        3: "把题目拆成两小步，先完成第一小步，再继续。",
        4: "现在可以看一个更明确的提示，但仍然先由你完成最后一步。",
        5: "你已经尝试了几次，我们先整理思路，再决定是否进入完整讲解。",
    }
    return prompts.get(max(1, min(hint_level, 5)), prompts[1])


def _contains_expected_answer(text: str, expected_answer: str | None) -> bool:
    if not expected_answer:
        return False
    answer = expected_answer.strip()
    if not answer:
        return False
    if answer.isdigit():
        return re.search(rf"(?<!\d){re.escape(answer)}(?!\d)", text) is not None
    return answer.lower() in text.lower()


def _contains_forbidden_phrase(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in FORBIDDEN_ANSWER_PHRASES)


def _looks_like_direct_answer(text: str, expected_answer: str | None) -> bool:
    if not _contains_expected_answer(text, expected_answer):
        return False
    answer = str(expected_answer or "").strip()
    normalized = re.sub(r"[\s，。,.？?！!：:；;、]", "", text.strip().lower())
    normalized_answer = re.sub(r"[\s，。,.？?！!：:；;、]", "", answer.lower())
    if normalized == normalized_answer:
        return True
    return re.search(
        rf"(答案|最终|结果|等于|=|是)\s*{re.escape(answer)}(?:\D*$|$)",
        text,
        re.IGNORECASE,
    ) is not None


def check_answer_leakage(
    *,
    draft_text: str,
    answer_unlocked: bool,
    expected_answer: str | None,
    hint_level: int,
    draft_hint_level: int | None = None,
) -> LeakageVerdict:
    verdict = check_child_safety(draft_text)
    if verdict.action == ChildSafetyAction.BLOCK:
        return LeakageVerdict(
            action=LeakageAction.BLOCK,
            reason=verdict.reason,
            safe_text=verdict.safe_text,
        )
    return LeakageVerdict(
        action=LeakageAction.ALLOW,
        reason="safe",
        safe_text=draft_text,
    )


def detect_answer_leakage(
    *,
    draft_text: str,
    answer_unlocked: bool,
    expected_answer: str | None,
) -> AnswerLeakageSignal:
    if answer_unlocked:
        return AnswerLeakageSignal(detected=False)

    if _contains_forbidden_phrase(draft_text):
        return AnswerLeakageSignal(
            detected=True,
            reason="forbidden_answer_phrase",
        )

    if _looks_like_direct_answer(draft_text, expected_answer):
        return AnswerLeakageSignal(
            detected=True,
            reason="direct_answer_leak",
        )

    return AnswerLeakageSignal(detected=False)


def _legacy_answer_leakage_verdict(
    *,
    draft_text: str,
    answer_unlocked: bool,
    expected_answer: str | None,
    hint_level: int,
    draft_hint_level: int | None = None,
) -> LeakageVerdict:
    if answer_unlocked:
        return LeakageVerdict(
            action=LeakageAction.ALLOW,
            reason="safe",
            safe_text=draft_text,
        )

    if draft_hint_level is not None and draft_hint_level > hint_level + 1:
        return LeakageVerdict(
            action=LeakageAction.BLOCK,
            reason="hint_level_skip",
            safe_text=_fallback_text(hint_level),
        )

    if _contains_forbidden_phrase(draft_text):
        return LeakageVerdict(
            action=LeakageAction.BLOCK,
            reason="forbidden_answer_phrase",
            safe_text=_fallback_text(hint_level),
        )

    if _looks_like_direct_answer(draft_text, expected_answer):
        return LeakageVerdict(
            action=LeakageAction.BLOCK,
            reason="direct_answer_leak",
            safe_text=_fallback_text(hint_level),
        )

    return LeakageVerdict(
        action=LeakageAction.ALLOW,
        reason="safe",
        safe_text=draft_text,
    )

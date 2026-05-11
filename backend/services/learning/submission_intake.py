from __future__ import annotations

import re

from songguo.backend.services.learning.submission_models import (
    LearningItemDraft,
    LearningSubmissionDraft,
    SourceType,
)
from songguo.backend.services.learning.input_safety import check_learning_input


_ANSWER_PREFIX_RE = re.compile(r"^(?:孩子)?答案\s*[:：]\s*(.+)$")
_QUESTION_NUMBER_RE = re.compile(r"^\s*(?:\d+[\.\uff0e、)]\s*)?(.*)$")


def parse_text_submission(
    *,
    child_id: str,
    subject: str,
    grade: int,
    raw_text: str,
) -> LearningSubmissionDraft:
    text = (raw_text or "").strip()
    safety = check_learning_input(text)
    if not safety.allowed:
        raise ValueError(f"{safety.reason}: {safety.message}")
    items = _parse_question_answer_blocks(text)
    return LearningSubmissionDraft(
        child_id=child_id,
        subject=subject,
        grade=grade,
        source_type=SourceType.TEXT,
        raw_text=text,
        items=items,
        needs_manual_confirm=bool(text and not items),
    )


def _parse_question_answer_blocks(text: str) -> list[LearningItemDraft]:
    if not text:
        return []

    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    parsed = [_parse_block(index, block) for index, block in enumerate(blocks, start=1)]
    items = [item for item in parsed if item is not None]
    if items:
        return _renumber(items)

    return _parse_line_pairs(text)


def _parse_block(index: int, block: str) -> LearningItemDraft | None:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    question_lines: list[str] = []
    child_answer: str | None = None
    for line in lines:
        match = _ANSWER_PREFIX_RE.match(line)
        if match:
            child_answer = match.group(1).strip()
            continue
        question_lines.append(line)

    question_text = _clean_question_text(" ".join(question_lines))
    if not _looks_like_question(question_text):
        return None
    return LearningItemDraft(
        item_index=index,
        question_text=question_text,
        child_answer=child_answer,
        confidence=0.9 if child_answer else 0.72,
    )


def _parse_line_pairs(text: str) -> list[LearningItemDraft]:
    items: list[LearningItemDraft] = []
    current_question: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        answer_match = _ANSWER_PREFIX_RE.match(line)
        if answer_match and current_question:
            items.append(
                LearningItemDraft(
                    item_index=len(items) + 1,
                    question_text=current_question,
                    child_answer=answer_match.group(1).strip(),
                    confidence=0.86,
                )
            )
            current_question = None
            continue
        cleaned = _clean_question_text(line)
        if _looks_like_question(cleaned):
            if current_question:
                items.append(
                    LearningItemDraft(
                        item_index=len(items) + 1,
                        question_text=current_question,
                        confidence=0.68,
                    )
                )
            current_question = cleaned
    if current_question:
        items.append(
            LearningItemDraft(
                item_index=len(items) + 1,
                question_text=current_question,
                confidence=0.68,
            )
        )
    return items


def _clean_question_text(value: str) -> str:
    match = _QUESTION_NUMBER_RE.match(value.strip())
    return (match.group(1) if match else value).strip()


def _looks_like_question(value: str) -> bool:
    if not value:
        return False
    return any(token in value for token in ("?", "？", "=", "＝", "÷", "+", "-", "×", "*")) or bool(
        re.search(r"\d+.*(?:几|多少|求|计算|平均|一共|剩)", value)
    )


def _renumber(items: list[LearningItemDraft]) -> list[LearningItemDraft]:
    return [
        item.model_copy(update={"item_index": index})
        for index, item in enumerate(items, start=1)
    ]

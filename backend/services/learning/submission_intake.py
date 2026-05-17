from __future__ import annotations

import re

from songguo.backend.services.learning.submission_models import (
    LearningItemDraft,
    LearningSubmissionDraft,
    SourceType,
)
from songguo.backend.services.learning.input_safety import check_learning_input


_ANSWER_PREFIX_RE = re.compile(
    r"^(?:(?:孩子|学生|child|student)\s*)?(?:答案|answer)\s*[:：]\s*(.+)$",
    re.IGNORECASE,
)
_QUESTION_NUMBER_RE = re.compile(r"^\s*(?:\d+[\.\uff0e、)]\s*)?(.*)$")
_QUESTION_WITH_NUMBER_RE = re.compile(r"^\s*(\d+)[\.\uff0e、)]\s*(.+)$")
_NUMBERED_QUESTION_START_RE = re.compile(r"^\s*\d+[\.\uff0e、)]\s*.+")
_BRACKETED_INLINE_RE = re.compile(r"[（(]\s*([^（）()]{1,24}?)\s*[）)]")
_SOLUTION_STEP_RE = re.compile(
    r"(?<![\w])"
    r"(?:\d+(?:\.\d+)?|[一二三四五六七八九十百千万零两]+)"
    r"(?:\s*[+\-＋－×xX*÷/]\s*(?:\d+(?:\.\d+)?|[一二三四五六七八九十百千万零两]+)){1,}"
    r"\s*[=＝]\s*"
    r"(?![?？])"
    r"(?:\d+(?:\.\d+)?|[一二三四五六七八九十百千万零两]+)"
    r"(?:\s*余\s*\d+)?"
    r"(?:\s*[（(]\s*[\u4e00-\u9fffA-Za-z]{1,4}\s*[）)])?"
)
_FINAL_ANSWER_RE = re.compile(
    r"(?:^|[\s，。；;,])答\s*[:：]\s*([^。；;，,\n]{1,32})"
)
_SCORE_CANDIDATE_RE = re.compile(r"^\d+(?:\.\d+)?\s*分$")
_NUMERIC_CANDIDATE_RE = re.compile(r"^[<>＝=≤≥+\-]?\d+(?:\.\d+)?(?:\s*余\s*\d+)?[<>＝=≤≥+\-]?$")
_COMPARISON_CANDIDATE_RE = re.compile(r"^[<>＝=≤≥]+$")
_CHINESE_NUMERAL_CANDIDATE_RE = re.compile(r"^[一二三四五六七八九十百千万亿零两]+$")
_SHORT_TEXT_ANSWER_RE = re.compile(r"^[A-Za-z0-9\u4e00-\u9fff<>＝=≤≥√✓错对，,、]{1,16}$")
_ANSWER_CONTEXT_PREVIOUS = {"=", "＝", "是", "为", "有", "共", "比", "到", "至", "加", "减", "添", "填", "剩"}
_ANSWER_CONTEXT_NEXT_RE = re.compile(r"^[个年月日时分秒周天位元米袋人页克厘米千公只本张条道题倍节岁]")


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

    interleaved_items = _parse_interleaved_numbered_column_rows(text)
    if interleaved_items:
        return _renumber(interleaved_items)

    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if any(_block_contains_multiple_pairs(block) for block in blocks):
        line_items = _parse_line_pairs(text)
        if line_items:
            return _renumber(line_items)
    parsed = _parse_blocks_with_orphan_answers(blocks)
    items = [item for item in parsed if item is not None]
    if items:
        return _renumber(items)

    return _renumber(_parse_line_pairs(text))


def _block_contains_multiple_pairs(block: str) -> bool:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    answer_count = sum(1 for line in lines if _ANSWER_PREFIX_RE.match(line))
    numbered_question_count = sum(1 for line in lines if _NUMBERED_QUESTION_START_RE.match(line))
    return answer_count > 1 or numbered_question_count > 1


def _parse_blocks_with_orphan_answers(blocks: list[str]) -> list[LearningItemDraft | None]:
    parsed: list[LearningItemDraft | None] = []
    for block in blocks:
        answer = _answer_only_block(block)
        if answer and parsed:
            previous = parsed[-1]
            if previous is not None and not previous.child_answer:
                parsed[-1] = previous.model_copy(
                    update={
                        "child_answer": answer,
                        "confidence": max(previous.confidence, 0.86),
                    }
                )
                continue
        parsed.append(_parse_block(len(parsed) + 1, block))
    return parsed


def _answer_only_block(block: str) -> str:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if len(lines) != 1:
        return ""
    match = _ANSWER_PREFIX_RE.match(lines[0])
    return match.group(1).strip() if match else ""


def _parse_block(index: int, block: str) -> LearningItemDraft | None:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    question_lines: list[str] = []
    child_answer: str | None = None
    has_numbered_question = False
    for line in lines:
        match = _ANSWER_PREFIX_RE.match(line)
        if match:
            child_answer = match.group(1).strip()
            continue
        if _NUMBERED_QUESTION_START_RE.match(line):
            has_numbered_question = True
        question_lines.append(line)

    question_text = _clean_question_text(" ".join(question_lines))
    if not has_numbered_question and not _looks_like_question(question_text):
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
        numbered_question = bool(_NUMBERED_QUESTION_START_RE.match(line))
        cleaned = _clean_question_text(line)
        if numbered_question or _looks_like_question(cleaned):
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


def _parse_interleaved_numbered_column_rows(text: str) -> list[LearningItemDraft]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    entries: list[tuple[int, LearningItemDraft]] = []
    number_order: list[int] = []
    index = 0
    while index < len(lines):
        question_group: list[tuple[int, str]] = []
        while index < len(lines):
            numbered = _numbered_question_line(lines[index])
            if numbered is None:
                break
            question_group.append(numbered)
            number_order.append(numbered[0])
            index += 1

        if len(question_group) < 2:
            index += 1 if not question_group else 0
            continue

        answer_group: list[str] = []
        while index < len(lines):
            answer_match = _ANSWER_PREFIX_RE.match(lines[index])
            if not answer_match:
                break
            answer_group.append(answer_match.group(1).strip())
            index += 1

        if len(answer_group) != len(question_group):
            continue

        for (question_number, question_text), answer in zip(question_group, answer_group, strict=True):
            entries.append(
                (
                    question_number,
                    LearningItemDraft(
                        item_index=question_number,
                        question_text=question_text,
                        child_answer=answer,
                        confidence=0.88,
                    ),
                )
            )

    if len(entries) < 4 or _strictly_increasing(number_order):
        return []
    deduped: dict[int, LearningItemDraft] = {}
    for question_number, item in entries:
        deduped.setdefault(question_number, item)
    return [deduped[number] for number in sorted(deduped)]


def _numbered_question_line(line: str) -> tuple[int, str] | None:
    match = _QUESTION_WITH_NUMBER_RE.match(line)
    if not match:
        return None
    question_text = match.group(2).strip()
    if not _looks_like_question(question_text):
        return None
    return int(match.group(1)), question_text


def _strictly_increasing(values: list[int]) -> bool:
    return all(current < next_value for current, next_value in zip(values, values[1:]))


def _clean_question_text(value: str) -> str:
    match = _QUESTION_NUMBER_RE.match(value.strip())
    return (match.group(1) if match else value).strip()


def _looks_like_question(value: str) -> bool:
    if not value:
        return False
    lowered = value.lower()
    if any(
        token in value
        for token in (
            "____",
            "___",
            "填空",
            "阅读",
            "概括",
            "回答",
            "造句",
            "仿写",
            "改写",
            "修改病句",
            "解释",
            "翻译",
            "组词",
            "扩句",
            "缩句",
        )
    ):
        return True
    if any(
        token in lowered
        for token in (
            "choose the correct",
            "correct tense",
            "make a sentence",
            "read and answer",
            "answer the question",
        )
    ):
        return True
    return any(token in value for token in ("?", "？", "=", "＝", "÷", "+", "-", "×", "*")) or bool(
        re.search(r"\d+.*(?:几|多少|求|计算|平均|一共|剩)", value)
    )


def _renumber(items: list[LearningItemDraft]) -> list[LearningItemDraft]:
    return [
        _extract_embedded_answer(item).model_copy(update={"item_index": index})
        for index, item in enumerate(items, start=1)
    ]


def _extract_embedded_answer(item: LearningItemDraft) -> LearningItemDraft:
    if item.child_answer:
        return item

    question_text = item.question_text.strip()
    if not question_text:
        return item

    question_without_work, work_steps, process_answer = _extract_solution_work(question_text)
    question_without_final_answer, final_answer = _extract_final_answer(question_without_work)
    bracket_question, bracket_answers = _extract_bracket_answers(question_without_final_answer)
    child_answer = "；".join(bracket_answers) if bracket_answers else final_answer or process_answer
    updates: dict[str, object] = {}
    if bracket_question != question_text:
        updates["question_text"] = _normalize_question_spacing(bracket_question)
    elif question_without_final_answer != question_text:
        updates["question_text"] = _normalize_question_spacing(question_without_final_answer)
    elif question_without_work != question_text:
        updates["question_text"] = _normalize_question_spacing(question_without_work)
    if child_answer:
        updates["child_answer"] = child_answer
        updates["confidence"] = max(item.confidence, 0.84)
    if work_steps:
        updates["work_steps"] = work_steps
        updates["confidence"] = max(float(updates.get("confidence", item.confidence)), 0.86)
    if not updates:
        return item
    return item.model_copy(update=updates)


def _extract_solution_work(question_text: str) -> tuple[str, str, str]:
    steps = [match.group(0).strip() for match in _SOLUTION_STEP_RE.finditer(question_text)]
    if not steps:
        return question_text, "", ""
    cleaned = _SOLUTION_STEP_RE.sub(" ", question_text)
    final_answer = _answer_from_solution_step(steps[-1])
    return _normalize_question_spacing(cleaned), "\n".join(steps), final_answer


def _answer_from_solution_step(step: str) -> str:
    match = re.search(r"[=＝]\s*(.+)$", step)
    if not match:
        return ""
    answer = match.group(1).strip()
    unit_match = re.match(r"^(.+?)[（(]\s*([^（）()]{1,4})\s*[）)]$", answer)
    if unit_match:
        return f"{unit_match.group(1).strip()}{unit_match.group(2).strip()}"
    return answer


def _extract_final_answer(question_text: str) -> tuple[str, str]:
    matches = list(_FINAL_ANSWER_RE.finditer(question_text))
    if not matches:
        return question_text, ""
    match = matches[-1]
    answer = _normalize_final_answer(match.group(1))
    if not answer:
        return question_text, ""
    cleaned = f"{question_text[:match.start()]} {question_text[match.end():]}"
    return _normalize_question_spacing(cleaned), answer


def _normalize_final_answer(value: str) -> str:
    answer = re.sub(r"\s+", "", value or "").strip("。；;，, ")
    return answer if 1 <= len(answer) <= 24 else ""


def _extract_bracket_answers(question_text: str) -> tuple[str, list[str]]:
    answers: list[str] = []

    def replace(match: re.Match[str]) -> str:
        candidate = _normalize_embedded_answer(match.group(1))
        if not _is_embedded_answer_candidate(candidate, question_text, match.start(), match.end()):
            return match.group(0)
        answers.append(candidate)
        return "( )"

    cleaned = _BRACKETED_INLINE_RE.sub(replace, question_text)
    return _normalize_question_spacing(cleaned), answers


def _normalize_embedded_answer(value: str) -> str:
    candidate = re.sub(r"\s+", "", value or "")
    if candidate in {"x", "X", "✕", "✖"}:
        return "×"
    if candidate == "✓":
        return "√"
    return candidate


def _is_embedded_answer_candidate(candidate: str, question_text: str, start: int, end: int) -> bool:
    if not candidate:
        return False
    if _SCORE_CANDIDATE_RE.match(candidate):
        return False
    if "分" in candidate or candidate.startswith(("共", "每")):
        return False
    if len(candidate) > 16:
        return False
    if re.search(r"[。！？?？:：;；]", candidate):
        return False
    if _looks_like_section_marker(candidate, question_text, start, end):
        return False
    if re.fullmatch(r"[A-Da-d]", candidate):
        return True
    if candidate in {"√", "×", "对", "错"}:
        return True
    if _NUMERIC_CANDIDATE_RE.match(candidate):
        return _has_fill_answer_context(question_text, start, end)
    if _COMPARISON_CANDIDATE_RE.match(candidate):
        return True
    if _CHINESE_NUMERAL_CANDIDATE_RE.match(candidate):
        return _has_fill_answer_context(question_text, start, end)
    if _contains_arithmetic_operator(candidate):
        return False
    return bool(
        re.search(r"\d", candidate)
        and _SHORT_TEXT_ANSWER_RE.match(candidate)
        and _has_fill_answer_context(question_text, start, end)
    )


def _has_fill_answer_context(question_text: str, start: int, end: int) -> bool:
    previous = _nearest_non_space(question_text[:start], reverse=True)
    next_char = _nearest_non_space(question_text[end:], reverse=False)
    if start <= 2 and (not previous or next_char):
        return False
    if previous in _ANSWER_CONTEXT_PREVIOUS:
        if previous in {"=", "＝", "是"}:
            return True
        if next_char and re.match(r"[\u4e00-\u9fffA-Za-z0-9]", next_char):
            return True
        return False
    return bool(next_char and _ANSWER_CONTEXT_NEXT_RE.match(next_char))


def _contains_arithmetic_operator(value: str) -> bool:
    return bool(re.search(r"[＋+\-－×xX*÷/]", value or ""))


def _looks_like_section_marker(candidate: str, question_text: str, start: int, end: int) -> bool:
    if not re.fullmatch(r"(?:\d+|[一二三四五六七八九十])", candidate):
        return False
    previous = _nearest_non_space(question_text[:start], reverse=True)
    next_char = _nearest_non_space(question_text[end:], reverse=False)
    if not next_char or not re.match(r"[\u4e00-\u9fffA-Za-z]", next_char):
        return False
    return previous not in _ANSWER_CONTEXT_PREVIOUS


def _nearest_non_space(value: str, *, reverse: bool) -> str:
    chars = reversed(value) if reverse else iter(value)
    for char in chars:
        if not char.isspace():
            return char
    return ""


def _normalize_question_spacing(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value or "").strip()
    cleaned = re.sub(r"\s+([，。！？；、,.!?;])", r"\1", cleaned)
    return cleaned

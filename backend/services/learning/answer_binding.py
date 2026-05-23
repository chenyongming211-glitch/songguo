from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ComparisonAnswerBinding:
    question_text: str
    child_answer: str


def bind_grouped_comparison_answers(
    *,
    question_text: str,
    child_answer: str | None,
) -> list[ComparisonAnswerBinding]:
    pairs = _comparison_pairs_with_gap(question_text)
    if len(pairs) < 2:
        return []
    external_signs = _comparison_answer_parts(child_answer or "")
    use_external = len(external_signs) == len(pairs)
    bindings: list[ComparisonAnswerBinding] = []
    for index, pair in enumerate(pairs):
        left, right, gap = pair
        embedded_sign = _embedded_comparison_sign(gap)
        answer = external_signs[index] if use_external else embedded_sign
        bindings.append(
            ComparisonAnswerBinding(
                question_text=f"{left}( ){right}",
                child_answer=answer,
            )
        )
    return bindings


def _comparison_pairs_with_gap(question_text: str) -> list[tuple[str, str, str]]:
    question = _normalize_text(question_text)
    atom = r"(?:\d+|[（(]\s*\d+(?:\s*[+\-＋－×xX*÷/]\s*\d+)+\s*[）)])"
    expression = rf"{atom}(?:\s*[+\-＋－×xX*÷/]\s*{atom})*"
    gap = r"(?P<gap>\(\s*[<>＝=≤≥\s]*\)|[○Oo]|[(（]\s*[<>＝=≤≥\s]*)"
    pairs: list[tuple[str, str, str]] = []
    for match in re.finditer(
        rf"(?P<left>{expression})\s*{gap}\s*(?P<right>{expression})",
        question,
    ):
        pairs.append(
            (
                _normalize_expression(match.group("left")),
                _normalize_expression(match.group("right")),
                match.group("gap"),
            )
        )
    return pairs


def _embedded_comparison_sign(gap: str) -> str:
    signs = [_normalize_sign(sign) for sign in re.findall(r"[<>＝=≤≥]", gap)]
    signs = [sign for sign in signs if sign]
    if not signs:
        return ""
    unique = set(signs)
    return signs[0] if len(unique) == 1 else ""


def _comparison_answer_parts(answer: str) -> list[str]:
    return [
        normalized
        for normalized in (_normalize_sign(sign) for sign in re.findall(r"[<>＝=≤≥]", answer))
        if normalized
    ]


def _normalize_sign(sign: str) -> str:
    if sign in {"＝", "="}:
        return "="
    if sign == "≤":
        return "<="
    if sign == "≥":
        return ">="
    if sign in {"<", ">"}:
        return sign
    return ""


def _normalize_expression(value: str) -> str:
    return re.sub(r"\s+", "", value).replace("（", "(").replace("）", ")").replace("＝", "=")


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).replace("（", "(").replace("）", ")")

from __future__ import annotations

from pydantic import BaseModel

from songguo.backend.evaluation.golden_math import GoldenMathQuestion
from songguo.backend.services.learning.ai_engine import AIEngineContext, AIEngineProvider
from songguo.backend.services.learning.math_structuring import validate_problem_analysis


class EvaluationReport(BaseModel):
    total: int
    schema_failures: int
    answer_leakage_count: int
    provider_failures: int
    skill_hit_count: int
    concept_card_hit_count: int = 0
    controlled_generation_failures: int = 0
    practice_generation_failures: int = 0

    @property
    def skill_hit_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.skill_hit_count / self.total

    @property
    def concept_card_hit_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.concept_card_hit_count / self.total


def evaluate_math_provider(
    *,
    provider: AIEngineProvider,
    questions: list[GoldenMathQuestion],
) -> EvaluationReport:
    schema_failures = 0
    leakage_count = 0
    provider_failures = 0
    skill_hit_count = 0
    concept_card_hit_count = 0
    controlled_generation_failures = 0
    practice_generation_failures = 0
    for question in questions:
        try:
            analysis = provider.structure_math_problem(
                question_text=question.question_text,
                grade=question.grade,
                context=AIEngineContext(
                    child_id="eval_child",
                    session_id=None,
                    tenant_id="eval_tenant",
                    request_id=question.question_id,
                ),
            )
        except Exception:
            provider_failures += 1
            continue

        verdict = validate_problem_analysis(analysis)
        if not verdict.valid:
            schema_failures += 1
            if any("leaks forbidden content" in error for error in verdict.errors):
                leakage_count += 1
            continue
        if analysis.skill_ids:
            skill_hit_count += 1
        if analysis.concept_card_ids:
            concept_card_hit_count += 1
        if _analysis_leaks_answer(analysis.final_answer, analysis.model_dump(mode="json")):
            leakage_count += 1
        try:
            draft = provider.generate_hint(
                problem_analysis=analysis,
                key_point=analysis.first_key_point,
                student_profile={},
                context=AIEngineContext(
                    child_id="eval_child",
                    session_id=f"eval_{question.question_id}",
                    tenant_id="eval_tenant",
                    request_id=f"{question.question_id}:hint",
                ),
            )
            if getattr(draft, "exposes_final_answer", False):
                controlled_generation_failures += 1
            if _text_leaks_answer(analysis.final_answer, str(draft.text)):
                controlled_generation_failures += 1
                leakage_count += 1
        except Exception:
            controlled_generation_failures += 1
        try:
            practice_items = provider.generate_similar_practice(
                problem_analysis=analysis,
                misconception=(
                    question.misconception_ids[0] if question.misconception_ids else None
                ),
                limit=3,
                context=AIEngineContext(
                    child_id="eval_child",
                    session_id=f"eval_{question.question_id}",
                    tenant_id="eval_tenant",
                    request_id=f"{question.question_id}:practice",
                ),
            )
            if not _practice_items_are_usable(practice_items):
                practice_generation_failures += 1
        except Exception:
            practice_generation_failures += 1
    return EvaluationReport(
        total=len(questions),
        schema_failures=schema_failures,
        answer_leakage_count=leakage_count,
        provider_failures=provider_failures,
        skill_hit_count=skill_hit_count,
        concept_card_hit_count=concept_card_hit_count,
        controlled_generation_failures=controlled_generation_failures,
        practice_generation_failures=practice_generation_failures,
    )


def _analysis_leaks_answer(final_answer: str, payload: dict) -> bool:
    if not final_answer or final_answer in {"待确认", "按60进制计算", "商和余数", "向上取整", "分数表示", "先乘后加"}:
        return False
    for key_point in payload.get("key_points", []):
        prompt = str(key_point.get("child_prompt") or "")
        forbidden = [final_answer, *key_point.get("forbidden_content", [])]
        if any(item and item in prompt for item in forbidden):
            return True
    return False


def _text_leaks_answer(final_answer: str, text: str) -> bool:
    if not final_answer or final_answer in {"待确认", "按60进制计算", "商和余数", "向上取整", "分数表示", "先乘后加"}:
        return False
    return _analysis_leaks_answer(
        final_answer,
        {"key_points": [{"child_prompt": text, "forbidden_content": [final_answer]}]},
    )


def _practice_items_are_usable(items: list) -> bool:
    if not 1 <= len(items) <= 3:
        return False
    for item in items:
        question = str(getattr(item, "question", "")).strip()
        answer = getattr(item, "answer", None)
        if not question or not answer:
            return False
        if "请再输入" in question or "同类型数学题" in question:
            return False
        if not any(marker in question for marker in ["?", "？", "多少", "几"]):
            return False
    return True

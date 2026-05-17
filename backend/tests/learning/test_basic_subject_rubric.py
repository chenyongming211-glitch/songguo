from __future__ import annotations

from songguo.backend.services.learning.basic_subject_rubric import (
    BasicSubjectRubricContext,
    BasicSubjectRubricEvaluator,
    build_basic_subject_rubric_prompt,
)


def test_basic_subject_rubric_evaluator_calls_llm_with_structured_prompt() -> None:
    seen = {}

    class FakeLLMClient:
        provider = "deepseek"
        model = "deepseek-chat"

        def complete_sync(self, prompt: str, *, system_prompt: str, temperature: float):
            seen["prompt"] = prompt
            seen["system_prompt"] = system_prompt
            seen["temperature"] = temperature
            return """
            {
              "outcome": "wrong",
              "confidence": 0.86,
              "question_type_id": "grammar_fix",
              "knowledge_point": "english_sentence_pattern",
              "misconception_tag": "english_past_tense_missing",
              "feedback_summary": "孩子没有根据 yesterday 调整动词形式，需要先找时间线索。",
              "criteria_evidence": ["作答和题干时间线索不一致"],
              "rubric_scores": {
                "task_alignment": 1,
                "form_accuracy": 0,
                "answer_completeness": 1
              }
            }
            """

    result = BasicSubjectRubricEvaluator(llm_client=FakeLLMClient()).evaluate(
        BasicSubjectRubricContext(
            subject="english",
            task_type="grammar_fix",
            grade=4,
            question_text="Choose the correct tense: He ____ to school yesterday.",
            child_answer="go",
        )
    )

    assert result.outcome == "wrong"
    assert result.misconception_tag == "english_past_tense_missing"
    assert result.rubric_scores["form_accuracy"] == 0
    assert result.provider == "deepseek"
    assert result.model == "deepseek-chat"
    assert "只输出 JSON" in seen["system_prompt"]
    assert "rubric_scores" in seen["prompt"]
    assert "不要输出标准答案" in seen["prompt"]
    assert seen["temperature"] == 0


def test_basic_subject_rubric_fast_path_skips_llm_for_obvious_english() -> None:
    class FakeLLMClient:
        provider = "deepseek"
        model = "deepseek-v4-flash"

        def complete_sync(self, *args, **kwargs):
            raise AssertionError("LLM should not be called for high-confidence rubric fast path")

    result = BasicSubjectRubricEvaluator(
        llm_client=FakeLLMClient(),
        fast_path_enabled=True,
    ).evaluate(
        BasicSubjectRubricContext(
            subject="english",
            task_type="grammar_fix",
            grade=4,
            question_text="Choose the correct tense: He ____ to school yesterday.",
            child_answer="go",
        )
    )

    assert result.outcome == "wrong"
    assert result.misconception_tag == "english_past_tense_missing"
    assert result.source == "fast_path"


def test_basic_subject_rubric_normalizes_noncanonical_llm_tag() -> None:
    class FakeLLMClient:
        provider = "deepseek"
        model = "deepseek-v4-flash"

        def complete_sync(self, prompt: str, *, system_prompt: str, temperature: float):
            return """
            {
              "outcome": "wrong",
              "confidence": 0.86,
              "question_type_id": "grammar_fix",
              "knowledge_point": "english_sentence_pattern",
              "misconception_tag": "动词过去式形式错误",
              "feedback_summary": "孩子使用了动词原形，但句子时间状语 yesterday 要求过去式。",
              "criteria_evidence": ["作答和题干时间线索不一致"],
              "rubric_scores": {"task_alignment": 1}
            }
            """

    result = BasicSubjectRubricEvaluator(llm_client=FakeLLMClient()).evaluate(
        BasicSubjectRubricContext(
            subject="english",
            task_type="grammar_fix",
            grade=4,
            question_text="Choose the correct tense: He ____ to school yesterday.",
            child_answer="go",
        )
    )

    assert result.misconception_tag == "english_past_tense_missing"


def test_basic_subject_rubric_normalizes_unknown_ascii_llm_tag_to_known_taxonomy() -> None:
    class FakeLLMClient:
        provider = "deepseek"
        model = "deepseek-v4-flash"

        def complete_sync(self, prompt: str, *, system_prompt: str, temperature: float):
            return """
            {
              "outcome": "wrong",
              "confidence": 0.86,
              "question_type_id": "grammar_fix",
              "knowledge_point": "english_sentence_pattern",
              "misconception_tag": "past_tense_irregular_verb",
              "feedback_summary": "孩子对一般过去时动词形式掌握不牢，需要关注不规则动词过去式变化。",
              "criteria_evidence": ["题干包含 yesterday"],
              "rubric_scores": {"task_alignment": 1}
            }
            """

    result = BasicSubjectRubricEvaluator(llm_client=FakeLLMClient()).evaluate(
        BasicSubjectRubricContext(
            subject="english",
            task_type="grammar_fix",
            grade=4,
            question_text="Choose the correct tense: He ____ to school yesterday.",
            child_answer="go",
        )
    )

    assert result.misconception_tag == "english_past_tense_missing"


def test_build_basic_subject_rubric_prompt_keeps_task_narrow() -> None:
    prompt = build_basic_subject_rubric_prompt(
        BasicSubjectRubricContext(
            subject="chinese",
            task_type="reading_comprehension",
            grade=4,
            question_text="阅读短文，回答作者为什么这样做？",
            child_answer="因为他很着急，想快点帮别人。",
        )
    )

    assert "judge_language_homework_with_rubric" in prompt
    assert "不要讲解题目" in prompt
    assert "不要输出标准答案" in prompt
    assert "reading_comprehension" in prompt

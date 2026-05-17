from __future__ import annotations

from songguo.backend.services.learning.intent_router import (
    IntentRouterAgent,
    IntentRouterContext,
    IntentRoutingDecision,
    RouterGuard,
    build_intent_router_prompt,
)


def test_intent_router_agent_uses_structured_agent_output() -> None:
    def fake_agent(context: IntentRouterContext) -> dict:
        assert context.raw_text.startswith("阅读短文")
        return {
            "subject": "chinese",
            "task_type": "reading_comprehension",
            "user_intent": "check_answer",
            "confidence": 0.87,
            "evidence": ["题干要求阅读短文后回答问题"],
            "needs_clarification": False,
            "route_to": "chinese_basic_tutor",
        }

    decision = IntentRouterAgent(agent_func=fake_agent).route(
        IntentRouterContext(
            child_id="child_001",
            grade=3,
            source_type="text",
            raw_text="阅读短文，回答作者为什么这样做？\n孩子答案：因为他很着急",
        )
    )

    assert decision.subject == "chinese"
    assert decision.task_type == "reading_comprehension"
    assert decision.user_intent == "check_answer"
    assert decision.route_to == "chinese_basic_tutor"
    assert decision.confidence == 0.87
    assert decision.router_version == "intent_router_agent_v0.1"


def test_intent_router_agent_calls_llm_client_with_structured_prompt() -> None:
    seen = {}

    class FakeLLMClient:
        provider = "deepseek"
        model = "deepseek-chat"

        def complete_sync(self, prompt: str, *, system_prompt: str, temperature: float):
            seen["prompt"] = prompt
            seen["system_prompt"] = system_prompt
            seen["temperature"] = temperature
            return """
            ```json
            {
              "subject": "english",
              "task_type": "grammar_fix",
              "user_intent": "check_answer",
              "confidence": 0.92,
              "evidence": ["题干是英文时态填空，应改为 went"],
              "needs_clarification": false,
              "route_to": "english_basic_tutor"
            }
            ```
            """

    decision = IntentRouterAgent(llm_client=FakeLLMClient()).route(
        IntentRouterContext(
            child_id="child_001",
            grade=4,
            source_type="text",
            raw_text="Choose the correct tense: He ____ to school yesterday.\n孩子答案：go",
        )
    )

    assert decision.subject == "english"
    assert decision.task_type == "grammar_fix"
    assert decision.user_intent == "check_answer"
    assert decision.confidence == 0.92
    assert decision.route_to == "english_basic_tutor"
    assert decision.router_version == "intent_router_agent_v0.1"
    assert all("went" not in item for item in decision.evidence)
    assert all("应改为" not in item for item in decision.evidence)
    assert "只输出 JSON" in seen["system_prompt"]
    assert "required_json_schema" in seen["prompt"]
    assert "Choose the correct tense" in seen["prompt"]
    assert seen["temperature"] == 0


def test_intent_router_fast_path_skips_llm_for_obvious_english() -> None:
    class FakeLLMClient:
        provider = "deepseek"
        model = "deepseek-v4-flash"

        def complete_sync(self, *args, **kwargs):
            raise AssertionError("LLM should not be called for high-confidence fast path")

    decision = IntentRouterAgent(
        llm_client=FakeLLMClient(),
        fast_path_enabled=True,
    ).route(
        IntentRouterContext(
            child_id="child_001",
            subject_hint="auto",
            grade=4,
            source_type="photo",
            raw_text="Choose the correct tense: He ____ to school yesterday.\nChild answer: go",
        )
    )

    assert decision.subject == "english"
    assert decision.route_to == "english_basic_tutor"
    assert decision.source == "fast_path"


def test_intent_router_fast_path_skips_llm_for_obvious_chinese_sentence_task() -> None:
    class FakeLLMClient:
        provider = "deepseek"
        model = "deepseek-v4-flash"

        def complete_sync(self, *args, **kwargs):
            raise AssertionError("LLM should not be called for high-confidence Chinese fast path")

    decision = IntentRouterAgent(
        llm_client=FakeLLMClient(),
        fast_path_enabled=True,
    ).route(
        IntentRouterContext(
            child_id="child_001",
            subject_hint="auto",
            grade=4,
            source_type="photo",
            raw_text='用 "因为……所以……" 造句。\n孩子答案：因为下雨，所以我带伞。',
        )
    )

    assert decision.subject == "chinese"
    assert decision.task_type == "sentence_rewrite"
    assert decision.route_to == "chinese_basic_tutor"
    assert decision.source == "fast_path"


def test_intent_router_fast_path_detects_chinese_reading_questions_without_reading_prefix() -> None:
    class FakeLLMClient:
        provider = "deepseek"
        model = "deepseek-v4-flash"

        def complete_sync(self, *args, **kwargs):
            raise AssertionError("LLM should not be called for high-confidence Chinese reading fast path")

    decision = IntentRouterAgent(
        llm_client=FakeLLMClient(),
        fast_path_enabled=True,
    ).route(
        IntentRouterContext(
            child_id="child_001",
            subject_hint="auto",
            grade=3,
            source_type="photo",
            raw_text="为什么小明上学没有迟到？\n孩子答案：因为他提前出门，所以按时到了学校。",
        )
    )

    assert decision.subject == "chinese"
    assert decision.task_type == "reading_comprehension"
    assert decision.route_to == "chinese_basic_tutor"
    assert decision.source == "fast_path"


def test_build_intent_router_prompt_contains_no_teaching_task() -> None:
    prompt = build_intent_router_prompt(
        IntentRouterContext(
            grade=3,
            source_type="text",
            raw_text="阅读短文，回答作者为什么这样做？",
        )
    )

    assert "不要生成讲解" in prompt
    assert "不能包含正确答案" in prompt
    assert "route_to" in prompt
    assert "reading_comprehension" in prompt


def test_router_guard_sends_low_confidence_to_clarification() -> None:
    guarded = RouterGuard(min_confidence=0.65).apply(
        IntentRoutingDecision(
            subject="math",
            task_type="word_problem",
            user_intent="check_answer",
            confidence=0.42,
            evidence=["题面信息不足"],
            needs_clarification=False,
            route_to="math_mistake_tutor",
        ),
        raw_text="这道题怎么做",
    )

    assert guarded.subject == "unknown"
    assert guarded.task_type == "unknown"
    assert guarded.route_to == "clarification_tutor"
    assert guarded.needs_clarification is True
    assert "low_confidence" in guarded.guard_reason


def test_router_guard_blocks_obvious_english_math_conflict() -> None:
    guarded = RouterGuard().apply(
        IntentRoutingDecision(
            subject="math",
            task_type="calculation",
            user_intent="check_answer",
            confidence=0.91,
            evidence=["模型误判为数学"],
            needs_clarification=False,
            route_to="math_mistake_tutor",
        ),
        raw_text="Choose the correct tense: He ____ to school yesterday. 孩子答案：go",
    )

    assert guarded.subject == "unknown"
    assert guarded.route_to == "clarification_tutor"
    assert guarded.needs_clarification is True
    assert "obvious_english_conflict" in guarded.guard_reason

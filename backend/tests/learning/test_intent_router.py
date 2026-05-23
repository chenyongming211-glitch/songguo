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


def test_intent_router_fast_path_routes_chinese_reading_sheet_with_choice_letters_to_chinese() -> None:
    class FakeLLMClient:
        provider = "deepseek"
        model = "deepseek-v4-flash"

        def complete_sync(self, *args, **kwargs):
            raise AssertionError("LLM should not be called for high-confidence Chinese fast path")

    raw_text = """
    对下面句子的主要意思的概括，最为恰当的是哪一项?( )(3分)
    A.荷花颜色多。B.荷花开放的样子很可爱。C.很多的荷花正在开放。D.荷花的形状多。
    你们班要开“寓言故事分享会”，请你帮忙解决问题。
    请用修改符号帮他修改。
    一起来包饺子吧，按照流程，用上表示先后顺序的词语。
    (1)再写一个像词语①这样的 A ABC 式的词语。
    """
    decision = IntentRouterAgent(
        llm_client=FakeLLMClient(),
        fast_path_enabled=True,
    ).route(
        IntentRouterContext(
            child_id="child_001",
            subject_hint="auto",
            grade=3,
            source_type="photo",
            raw_text=raw_text,
        )
    )
    guarded = RouterGuard().apply(decision, raw_text=raw_text)

    assert decision.subject == "chinese"
    assert decision.task_type == "reading_comprehension"
    assert decision.route_to == "chinese_basic_tutor"
    assert decision.source == "fast_path"
    assert guarded.subject == "chinese"


def test_intent_router_fast_path_routes_chinese_pinyin_sheet_with_scores_to_chinese() -> None:
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
            grade=3,
            source_type="photo",
            raw_text="""
            看拼音，写词语。gōng yuán wēi fēng qīng xiāng
            读一读，选出正确的读音或字形，打“√”。
            下列加点字词的解释有误的是哪一项?( )(3分)
            补全下列四字词语，再完成练习。
            """,
        )
    )

    assert decision.subject == "chinese"
    assert decision.task_type == "sentence_rewrite"
    assert decision.route_to == "chinese_basic_tutor"
    assert decision.source == "fast_path"


def test_intent_router_fast_path_keeps_obvious_english_after_chinese_guard_tightening() -> None:
    class FakeLLMClient:
        provider = "deepseek"
        model = "deepseek-v4-flash"

        def complete_sync(self, *args, **kwargs):
            raise AssertionError("LLM should not be called for high-confidence English fast path")

    decision = IntentRouterAgent(
        llm_client=FakeLLMClient(),
        fast_path_enabled=True,
    ).route(
        IntentRouterContext(
            child_id="child_001",
            subject_hint="auto",
            grade=4,
            source_type="photo",
            raw_text="Choose the correct word: He ____ football after school. A. play B. plays",
        )
    )

    assert decision.subject == "english"
    assert decision.route_to == "english_basic_tutor"
    assert decision.source == "fast_path"


def test_intent_router_fast_path_keeps_obvious_math_after_chinese_guard_tightening() -> None:
    class FakeLLMClient:
        provider = "deepseek"
        model = "deepseek-v4-flash"

        def complete_sync(self, *args, **kwargs):
            raise AssertionError("LLM should not be called for high-confidence Math fast path")

    decision = IntentRouterAgent(
        llm_client=FakeLLMClient(),
        fast_path_enabled=True,
    ).route(
        IntentRouterContext(
            child_id="child_001",
            subject_hint="auto",
            grade=3,
            source_type="photo",
            raw_text="口算。36÷4=9  27+18=45  5×8=40",
        )
    )

    assert decision.subject == "math"
    assert decision.task_type == "calculation"
    assert decision.route_to == "math_mistake_tutor"
    assert decision.source == "fast_path"


def test_intent_router_routes_english_workbook_with_chinese_instructions_to_english() -> None:
    decision = IntentRouterAgent().route(
        IntentRouterContext(
            child_id="child_001",
            subject_hint="auto",
            grade=3,
            source_type="photo",
            raw_text=(
                "Read, choose and write.选词填空，补全短文。"
                "My life food like water breakfast good. "
                "What do you like? Would you like some juice?"
            ),
        )
    )

    assert decision.subject == "english"
    assert decision.route_to == "english_basic_tutor"


def test_intent_router_routes_english_textbook_marker_to_english() -> None:
    decision = IntentRouterAgent().route(
        IntentRouterContext(
            child_id="child_001",
            subject_hint="auto",
            grade=3,
            source_type="photo",
            raw_text=(
                "培优100分英语三年级下册 Read and judge.判断下列句子中画线部分的发音是(T)否(F)一致。"
                "How many? Would you like some beef?"
            ),
        )
    )

    assert decision.subject == "english"
    assert decision.route_to == "english_basic_tutor"


def test_intent_router_routes_chinese_reading_with_numbers_to_chinese() -> None:
    decision = IntentRouterAgent().route(
        IntentRouterContext(
            child_id="child_001",
            subject_hint="auto",
            grade=3,
            source_type="photo",
            raw_text=(
                "赵州桥非常雄伟。桥长五十多米，有九米多宽。"
                "根据选段的描述，下面图片中可能是赵州桥的是哪一项?(3分) "
                "A. B. C. D. 对画线句子理解最准确的是哪一项?"
            ),
        )
    )

    assert decision.subject == "chinese"
    assert decision.route_to == "chinese_basic_tutor"


def test_intent_router_routes_plain_chinese_true_false_to_chinese() -> None:
    decision = IntentRouterAgent().route(
        IntentRouterContext(
            child_id="child_001",
            subject_hint="auto",
            grade=4,
            source_type="photo",
            raw_text="斗转星移法是依据一年中北极星的位置变化来确定一年四季的方法。(×)",
        )
    )

    assert decision.subject == "chinese"
    assert decision.route_to == "chinese_basic_tutor"


def test_intent_router_keeps_math_page_with_traditional_culture_context_in_math() -> None:
    decision = IntentRouterAgent().route(
        IntentRouterContext(
            child_id="child_001",
            subject_hint="auto",
            grade=3,
            source_type="photo",
            raw_text=(
                "第三单元情境题强化训练 一、填空。口算21×50时，可以先算21×5=(105)。"
                "在里填上“>”“<”或“=”。50×40 15×80。"
                "[新情境·传统文化]月饼被中国南北各地的人们所喜爱，一盒月饼12个，"
                "王老师买了22盒，一共买了(264)个。"
            ),
        )
    )

    assert decision.subject == "math"
    assert decision.route_to == "math_mistake_tutor"


def test_intent_router_keeps_year_month_day_math_sheet_in_math() -> None:
    decision = IntentRouterAgent().route(
        IntentRouterContext(
            child_id="child_001",
            subject_hint="auto",
            grade=3,
            source_type="photo",
            raw_text=(
                "小学学霸冲A卷 年、月、日的奥秘综合素养评价 数学三年级下。"
                "填一填。4日=( )时 6周=(42)天 5年=( )个月。"
                "7月26日一共开放多长时间?"
            ),
        )
    )

    assert decision.subject == "math"
    assert decision.route_to == "math_mistake_tutor"


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


def test_router_guard_blocks_obvious_chinese_sheet_misrouted_to_english() -> None:
    guarded = RouterGuard().apply(
        IntentRoutingDecision(
            subject="english",
            task_type="grammar_fix",
            user_intent="check_answer",
            confidence=0.91,
            evidence=["模型误判为英语"],
            needs_clarification=False,
            route_to="english_basic_tutor",
        ),
        raw_text="看拼音，写词语。读一读，选出正确的读音或字形，打“√”。",
    )

    assert guarded.subject == "unknown"
    assert guarded.route_to == "clarification_tutor"
    assert guarded.needs_clarification is True
    assert "obvious_chinese_conflict" in guarded.guard_reason

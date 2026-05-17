from __future__ import annotations

import time

from songguo.backend.services.learning.ai_engine import (
    AIEngineContext,
    DeepSeekProvider,
    DeterministicFallbackProvider,
    ProviderChain,
    ProviderError,
    _provider_timeout_from_env,
)
from songguo.backend.services.learning.math_structuring import ProblemAnalysis


BUS_QUESTION = "学校组织三年级学生春游，一共有4个班，每班32人。如果每辆大巴车限坐45人，那么至少需要多少辆大巴车？"


class FailingProvider:
    provider_name = "failing"
    model_name = "broken"

    def structure_math_problem(
        self,
        *,
        question_text: str,
        grade: int,
        context: AIEngineContext,
    ) -> ProblemAnalysis:
        raise ProviderError("provider unavailable")


class SlowProvider:
    provider_name = "slow"
    model_name = "hangs"

    def structure_math_problem(
        self,
        *,
        question_text: str,
        grade: int,
        context: AIEngineContext,
    ) -> ProblemAnalysis:
        time.sleep(0.05)
        raise AssertionError("ProviderChain should time out this provider first")


class FlakyProvider:
    provider_name = "flaky"
    model_name = "temporary_failure"

    def __init__(self) -> None:
        self.calls = 0
        self.fallback = DeterministicFallbackProvider()

    def structure_math_problem(
        self,
        *,
        question_text: str,
        grade: int,
        context: AIEngineContext,
    ) -> ProblemAnalysis:
        self.calls += 1
        if self.calls == 1:
            raise ProviderError("temporary upstream failure")
        return self.fallback.structure_math_problem(
            question_text=question_text,
            grade=grade,
            context=context,
        )


def test_provider_chain_falls_back_to_deterministic_provider() -> None:
    chain = ProviderChain([FailingProvider(), DeterministicFallbackProvider()])

    result = chain.structure_math_problem(
        question_text=BUS_QUESTION,
        grade=3,
        context=AIEngineContext(child_id="child_001", session_id=None),
    )

    assert result.problem_type == "capacity_round_up"
    assert result.source == "deterministic_fallback"
    assert result.first_key_point.id == "kp_total_people"
    assert "3辆" not in result.first_key_point.child_prompt
    assert chain.last_call.provider == "deterministic_fallback"
    assert chain.last_call.status == "success"
    assert chain.last_call.latency_ms >= 0


def test_provider_chain_times_out_slow_provider_and_falls_back() -> None:
    chain = ProviderChain(
        [SlowProvider(), DeterministicFallbackProvider()],
        provider_timeout_seconds=0.01,
    )

    result = chain.structure_math_problem(
        question_text=BUS_QUESTION,
        grade=3,
        context=AIEngineContext(child_id="child_001", session_id=None),
    )

    assert result.problem_type == "capacity_round_up"
    assert result.source == "deterministic_fallback"
    assert chain.last_call.provider == "deterministic_fallback"
    assert chain.last_call.status == "success"


def test_provider_chain_retries_same_provider_before_fallback() -> None:
    flaky = FlakyProvider()
    chain = ProviderChain(
        [flaky, DeterministicFallbackProvider()],
        provider_retry_attempts=2,
        provider_retry_backoff_seconds=0,
    )

    result = chain.structure_math_problem(
        question_text=BUS_QUESTION,
        grade=3,
        context=AIEngineContext(child_id="child_001", session_id=None),
    )

    assert flaky.calls == 2
    assert result.problem_type == "capacity_round_up"
    assert chain.last_call.provider == "flaky"
    assert chain.last_call.status == "success"


def test_provider_chain_default_timeout_allows_real_model_retry_window(monkeypatch) -> None:
    monkeypatch.delenv("SONGGUO_AI_PROVIDER_TIMEOUT_SECONDS", raising=False)

    assert _provider_timeout_from_env() == 20.0


def test_deepseek_provider_defaults_to_deepseek_v4_flash(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    provider = DeepSeekProvider()

    assert provider.model_name == "deepseek-v4-flash"


def test_fallback_provider_binds_skill_and_misconception_ids() -> None:
    provider = DeterministicFallbackProvider()

    analysis = provider.structure_math_problem(
        question_text=BUS_QUESTION,
        grade=3,
        context=AIEngineContext(child_id="child_001", session_id=None),
    )

    assert "math_capacity_round_up" in analysis.skill_ids
    assert "math_multiplication_total_count" in analysis.skill_ids
    assert "math_capacity_ignored_remainder_round_up" in analysis.misconception_ids
    assert analysis.concept_card_ids


def test_fallback_provider_generates_safe_current_key_point_hint() -> None:
    provider = DeterministicFallbackProvider()
    analysis = provider.structure_math_problem(
        question_text=BUS_QUESTION,
        grade=3,
        context=AIEngineContext(child_id="child_001", session_id=None),
    )

    draft = provider.generate_hint(
        problem_analysis=analysis,
        key_point=analysis.first_key_point,
        student_profile={},
        context=AIEngineContext(child_id="child_001", session_id="s_001"),
    )

    assert draft.action == "key_point_hint"
    assert draft.exposes_final_answer is False
    assert "3辆" not in draft.text
    assert draft.metadata["provider"] == "deterministic_fallback"


def test_fallback_provider_preserves_times_five_guided_hint() -> None:
    provider = DeterministicFallbackProvider()

    analysis = provider.structure_math_problem(
        question_text="48 x 5 = ?",
        grade=3,
        context=AIEngineContext(child_id="child_001", session_id=None),
    )
    draft = provider.generate_hint(
        problem_analysis=analysis,
        key_point=analysis.first_key_point,
        student_profile={},
        context=AIEngineContext(child_id="child_001", session_id="s_001"),
    )

    assert analysis.problem_type == "two_digit_times_one_digit"
    assert "math_two_digit_times_one_digit" in analysis.skill_ids
    assert "48 × 10" in draft.text
    assert "240" not in draft.text


def test_generic_fallback_does_not_guess_final_answer_from_condition_number() -> None:
    provider = DeterministicFallbackProvider()

    analysis = provider.structure_math_problem(
        question_text="一本故事书有120页。小兰第一天看了30页，第二天看了剩下页数的一半。第三天她应该从第几页开始看？",
        grade=3,
        context=AIEngineContext(child_id="child_001", session_id=None),
    )

    assert analysis.problem_type == "read_conditions"
    assert analysis.final_answer == "待确认"
    assert "30" not in analysis.first_key_point.forbidden_content


def test_generic_fallback_evaluates_full_arithmetic_expression() -> None:
    provider = DeterministicFallbackProvider()

    analysis = provider.structure_math_problem(
        question_text="4*3+2*2=?",
        grade=3,
        context=AIEngineContext(child_id="child_001", session_id=None),
    )

    assert analysis.problem_type == "read_conditions"
    assert analysis.final_answer == "16"
    assert analysis.confidence >= 0.6
    assert "16" in analysis.first_key_point.forbidden_content


def test_generic_fallback_evaluates_unicode_arithmetic_expression() -> None:
    provider = DeterministicFallbackProvider()

    analysis = provider.structure_math_problem(
        question_text="(4×3+2×2)÷2=?",
        grade=3,
        context=AIEngineContext(child_id="child_001", session_id=None),
    )

    assert analysis.final_answer == "8"


def test_generic_fallback_evaluates_ascii_x_arithmetic_expression() -> None:
    provider = DeterministicFallbackProvider()

    analysis = provider.structure_math_problem(
        question_text="4x3+2x2=?",
        grade=3,
        context=AIEngineContext(child_id="child_001", session_id=None),
    )

    assert analysis.final_answer == "16"
    assert analysis.confidence >= 0.6

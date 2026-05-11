from __future__ import annotations

from songguo.backend.services.learning.leakage_checker import (
    LeakageAction,
    check_answer_leakage,
    detect_answer_leakage,
)


def test_allows_low_exposure_hint_without_final_answer() -> None:
    verdict = check_answer_leakage(
        draft_text="Think about 36 x 10 first. Then take half.",
        answer_unlocked=False,
        expected_answer="180",
        hint_level=1,
    )

    assert verdict.action == LeakageAction.ALLOW
    assert verdict.reason == "safe"
    assert verdict.safe_text == "Think about 36 x 10 first. Then take half."


def test_direct_answer_is_quality_signal_not_safety_block() -> None:
    verdict = check_answer_leakage(
        draft_text="36 x 5 = 180, so the answer is 180.",
        answer_unlocked=False,
        expected_answer="180",
        hint_level=1,
    )

    assert verdict.action == LeakageAction.ALLOW
    assert verdict.reason == "safe"
    assert verdict.safe_text == "36 x 5 = 180, so the answer is 180."

    signal = detect_answer_leakage(
        draft_text="36 x 5 = 180, so the answer is 180.",
        answer_unlocked=False,
        expected_answer="180",
    )

    assert signal.detected is True
    assert signal.reason == "forbidden_answer_phrase"


def test_blocks_terse_direct_answer_while_locked() -> None:
    verdict = check_answer_leakage(
        draft_text="180",
        answer_unlocked=False,
        expected_answer="180",
        hint_level=1,
    )

    assert verdict.action == LeakageAction.ALLOW
    signal = detect_answer_leakage(
        draft_text="180",
        answer_unlocked=False,
        expected_answer="180",
    )
    assert signal.detected is True
    assert signal.reason == "direct_answer_leak"


def test_allows_normal_prompt_that_mentions_problem_condition_number() -> None:
    verdict = check_answer_leakage(
        draft_text="很好，你找到了总数是120页。那第一天看了30页后，还剩下多少页呢？",
        answer_unlocked=False,
        expected_answer="30",
        hint_level=2,
    )

    assert verdict.action == LeakageAction.ALLOW
    assert verdict.reason == "safe"


def test_forbidden_answer_phrase_is_quality_signal_not_safety_block() -> None:
    verdict = check_answer_leakage(
        draft_text="所以答案是先把 36 乘以 5。",
        answer_unlocked=False,
        expected_answer=None,
        hint_level=2,
    )

    assert verdict.action == LeakageAction.ALLOW
    assert verdict.reason == "safe"
    assert verdict.safe_text == "所以答案是先把 36 乘以 5。"


def test_unlocked_answer_allows_full_explanation() -> None:
    verdict = check_answer_leakage(
        draft_text="36 x 5 = 180 because 36 x 10 is 360 and half is 180.",
        answer_unlocked=True,
        expected_answer="180",
        hint_level=5,
    )

    assert verdict.action == LeakageAction.ALLOW
    assert verdict.safe_text.endswith("half is 180.")


def test_blocks_hint_level_skip() -> None:
    verdict = check_answer_leakage(
        draft_text="Now I will give a full solution.",
        answer_unlocked=False,
        expected_answer=None,
        hint_level=1,
        draft_hint_level=4,
    )

    assert verdict.action == LeakageAction.ALLOW
    assert verdict.reason == "safe"

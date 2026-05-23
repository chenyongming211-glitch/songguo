from __future__ import annotations

from songguo.backend.services.learning.answer_binding import bind_grouped_comparison_answers


def test_binds_comparison_answer_from_embedded_gap_when_external_answer_is_incomplete() -> None:
    bindings = bind_grouped_comparison_answers(
        question_text="50x40()15x80 63×27(< < )27×85 40×125(100×28",
        child_answer="<<",
    )

    assert [(binding.question_text, binding.child_answer) for binding in bindings] == [
        ("50x40( )15x80", ""),
        ("63×27( )27×85", "<"),
        ("40×125( )100×28", ""),
    ]


def test_binds_comparison_answers_from_external_answer_when_count_matches() -> None:
    bindings = bind_grouped_comparison_answers(
        question_text="50×40○15×80 63×27○27×85 40×125○100×28",
        child_answer="><>",
    )

    assert [(binding.question_text, binding.child_answer) for binding in bindings] == [
        ("50×40( )15×80", ">"),
        ("63×27( )27×85", "<"),
        ("40×125( )100×28", ">"),
    ]


def test_does_not_bind_incomplete_external_answers_to_positions() -> None:
    bindings = bind_grouped_comparison_answers(
        question_text="50×40○15×80 63×27○27×85 40×125○100×28",
        child_answer="<<",
    )

    assert [(binding.question_text, binding.child_answer) for binding in bindings] == [
        ("50×40( )15×80", ""),
        ("63×27( )27×85", ""),
        ("40×125( )100×28", ""),
    ]

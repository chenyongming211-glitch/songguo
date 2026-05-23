from __future__ import annotations

from songguo.backend.services.learning.question_splitter import split_grouped_math_item


def test_splits_grouped_comparison_block_into_individual_items() -> None:
    items = split_grouped_math_item(
        question_text="在○里填上“>”“<”或“=”。50×40○15×80 63×27○27×85 40×125○100×28",
        child_answer="><>",
        question_type_id="math_grouped_comparison_sign",
    )

    assert [(item.question_text, item.child_answer) for item in items] == [
        ("50×40( )15×80", ">"),
        ("63×27( )27×85", "<"),
        ("40×125( )100×28", ">"),
    ]


def test_splits_grouped_comparison_without_binding_incomplete_answers() -> None:
    items = split_grouped_math_item(
        question_text="在○里填上“>”“<”或“=”。50×40○15×80 63×27○27×85 40×125○100×28",
        child_answer="<<",
        question_type_id="math_grouped_comparison_sign",
    )

    assert [(item.question_text, item.child_answer) for item in items] == [
        ("50×40( )15×80", ""),
        ("63×27( )27×85", ""),
        ("40×125( )100×28", ""),
    ]


def test_splits_grouped_comparison_with_embedded_ocr_signs() -> None:
    items = split_grouped_math_item(
        question_text="50x40()15x80 63×27(< < )27×85 40×125(100×28",
        child_answer="<<",
        question_type_id="math_grouped_comparison_sign",
    )

    assert [(item.question_text, item.child_answer) for item in items] == [
        ("50x40( )15x80", ""),
        ("63×27( )27×85", "<"),
        ("40×125( )100×28", ""),
    ]


def test_splits_grouped_true_false_block_into_individual_items() -> None:
    items = split_grouped_math_item(
        question_text=(
            "二、判断。1.两位数乘两位数，积可能是三位数，也可能是四位数。(V) "
            "2.两个乘数末尾共有2个0，积的末尾也一定有2个0。(x)"
        ),
        child_answer="V；x",
        question_type_id="math_grouped_true_false",
    )

    assert [(item.question_text, item.child_answer) for item in items] == [
        ("两位数乘两位数，积可能是三位数，也可能是四位数。", "V"),
        ("两个乘数末尾共有2个0，积的末尾也一定有2个0。", "x"),
    ]


def test_splits_grouped_true_false_with_partial_external_answer() -> None:
    items = split_grouped_math_item(
        question_text=(
            "二、判断。1.两位数乘两位数，积可能是三位数，也可能是四位数。(V) "
            "2.两个乘数末尾共有2个0，积的末尾也一定有2个0。( )"
        ),
        child_answer="x",
        question_type_id="math_grouped_true_false",
    )

    assert [(item.question_text, item.child_answer) for item in items] == [
        ("两位数乘两位数，积可能是三位数，也可能是四位数。", "V"),
        ("两个乘数末尾共有2个0，积的末尾也一定有2个0。", "x"),
    ]


def test_splits_multi_blank_arithmetic_when_ocr_merged_answers() -> None:
    items = split_grouped_math_item(
        question_text="口算21×50时，可以先算21×5=( )，再在积的后面添上( )个0。",
        child_answer="1055",
        question_type_id="math_grouped_fill_blank",
    )

    assert [(item.question_text, item.child_answer) for item in items] == [
        ("21×5=( )", "105"),
        ("口算21×50时，再在积的后面添上( )个0。", "5"),
    ]


def test_splits_grouped_oral_calculation_equations() -> None:
    items = split_grouped_math_item(
        question_text="直接写得数。5×30=150 32×30=960 20×32=60 600×50=30000",
        child_answer="30000",
        question_type_id="math_grouped_oral_calculation",
    )

    assert [(item.question_text, item.child_answer) for item in items] == [
        ("5×30=", "150"),
        ("32×30=", "960"),
        ("20×32=", "60"),
        ("600×50=", "30000"),
    ]


def test_splits_vertical_calculation_from_explicit_process_only() -> None:
    items = split_grouped_math_item(
        question_text=(
            "用竖式计算。32×21=( 672 ) 32 46 208 "
            "解题过程：46×18=828 176×50=8800 208×74=15392"
        ),
        child_answer="15392",
        question_type_id="math_vertical_calculation_block",
    )

    assert [(item.question_text, item.child_answer) for item in items] == [
        ("46×18=", "828"),
        ("176×50=", "8800"),
        ("208×74=", "15392"),
    ]

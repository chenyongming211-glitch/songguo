from __future__ import annotations

from songguo.backend.services.learning.question_type_routing import (
    MathQuestionKind,
    route_math_question_type,
)


def test_routes_direct_arithmetic_as_oral_calculation() -> None:
    route = route_math_question_type(question_text="48 ÷ 6 = ?", child_answer="8")

    assert route.kind == MathQuestionKind.ORAL_CALCULATION
    assert route.evaluation_strategy == "deterministic"
    assert route.question_type_id == "math_oral_calculation"


def test_routes_comparison_sign_items() -> None:
    route = route_math_question_type(question_text="50×40( )15×80", child_answer=">")

    assert route.kind == MathQuestionKind.COMPARISON_SIGN
    assert route.evaluation_strategy == "deterministic"
    assert route.question_type_id == "math_comparison_sign"


def test_routes_true_false_items() -> None:
    route = route_math_question_type(question_text="两位数乘两位数，积可能是三位数，也可能是四位数。", child_answer="√")

    assert route.kind == MathQuestionKind.TRUE_FALSE
    assert route.evaluation_strategy == "deterministic"
    assert route.question_type_id == "math_true_false_fact"


def test_routes_true_false_items_with_embedded_ocr_mark_without_answer_field() -> None:
    route = route_math_question_type(
        question_text="最大的两位数与最小的三位数的乘积是9990。 ( ×",
        child_answer=None,
    )

    assert route.kind == MathQuestionKind.TRUE_FALSE
    assert route.evaluation_strategy == "deterministic"
    assert route.question_type_id == "math_true_false_fact"


def test_routes_choice_items() -> None:
    route = route_math_question_type(
        question_text="积大约是5600的算式是( )。 A.59×79 B.79×61 C.79×71",
        child_answer="C",
    )

    assert route.kind == MathQuestionKind.CHOICE
    assert route.evaluation_strategy == "deterministic"


def test_routes_fill_blank_items() -> None:
    route = route_math_question_type(question_text="87×23的积是( )位数。", child_answer="四")

    assert route.kind == MathQuestionKind.FILL_BLANK
    assert route.evaluation_strategy == "deterministic"


def test_routes_embedded_arithmetic_answer_as_deterministic_fill_blank() -> None:
    route = route_math_question_type(
        question_text="口算21×50时，可以先算21×5=( 105",
        child_answer=None,
    )

    assert route.kind == MathQuestionKind.FILL_BLANK
    assert route.evaluation_strategy == "deterministic"
    assert route.question_type_id == "math_fill_blank"


def test_routes_uncertain_multi_blank_answers_to_item_split() -> None:
    route = route_math_question_type(
        question_text="口算21×50时，可以先算21×5=( )，再在积的后面添上( )个0。",
        child_answer="1055",
    )

    assert route.kind == MathQuestionKind.GROUPED_FILL_BLANK
    assert route.evaluation_strategy == "item_split_required"
    assert route.question_type_id == "math_grouped_fill_blank"


def test_routes_grouped_oral_calculation_block_to_item_split() -> None:
    route = route_math_question_type(
        question_text="直接写得数。5×30=150 32×30=960 20×32=60 600×50=30000",
        child_answer="30000",
    )

    assert route.kind == MathQuestionKind.GROUPED_ORAL_CALCULATION
    assert route.evaluation_strategy == "item_split_required"
    assert route.question_type_id == "math_grouped_oral_calculation"


def test_routes_vertical_calculation_block_to_manual_confirm_guard() -> None:
    route = route_math_question_type(
        question_text=(
            "竖式计算-多位数乘法 题数：20 360×20=7200 "
            "18×11=198 270×13=3510 370×20=360"
        ),
        child_answer="360",
    )

    assert route.kind == MathQuestionKind.VERTICAL_CALCULATION_BLOCK
    assert route.evaluation_strategy == "manual_confirm"
    assert route.question_type_id == "math_vertical_calculation_block"


def test_routes_vertical_calculation_process_block_to_item_split() -> None:
    route = route_math_question_type(
        question_text=(
            "用竖式计算。32×21=( 672 ) 解题过程："
            "46×18=828 176×50=8800 208×74=15392"
        ),
        child_answer="15392",
    )

    assert route.kind == MathQuestionKind.VERTICAL_CALCULATION_BLOCK
    assert route.evaluation_strategy == "item_split_required"
    assert route.question_type_id == "math_vertical_calculation_block"


def test_does_not_route_word_problem_work_steps_as_grouped_oral_calculation() -> None:
    route = route_math_question_type(
        question_text="从10月25日开始，到11月15日结束。每天活动经费为8元。31-25+1+15=22 22×8=176",
        child_answer="176元",
    )

    assert route.kind == MathQuestionKind.WORD_PROBLEM
    assert route.evaluation_strategy == "math_gateway"


def test_does_not_route_noisy_vertical_equations_as_grouped_oral_without_marker() -> None:
    route = route_math_question_type(
        question_text=(
            "20 × 13 2 1200 8 8 1 8 74 00 2 27 98 3510 "
            "12×87=042 74×55=4070 115×55=5 146×35=5110 "
            "23×12=276 14×59=826 110×80=8800 2 3 7 59 826"
        ),
        child_answer="836",
    )

    assert route.question_type_id != "math_grouped_oral_calculation"
    assert route.evaluation_strategy != "item_split_required"


def test_routes_clean_equation_list_as_grouped_oral_without_marker() -> None:
    route = route_math_question_type(
        question_text="5×30=150 32×30=960 20×32=60 600×50=30000 12×40=480",
        child_answer="480",
    )

    assert route.kind == MathQuestionKind.GROUPED_ORAL_CALCULATION
    assert route.evaluation_strategy == "item_split_required"


def test_routes_word_problem_items_to_gateway_by_default() -> None:
    route = route_math_question_type(
        question_text="36颗松果平均分给5只小松鼠，每只几颗，还剩几颗？",
        child_answer="每只6颗，还剩6颗",
    )

    assert route.kind == MathQuestionKind.WORD_PROBLEM
    assert route.evaluation_strategy == "math_gateway"


def test_routes_grouped_true_false_block_to_item_split() -> None:
    route = route_math_question_type(
        question_text=(
            "二、判断。1.两位数乘两位数，积可能是三位数，也可能是四位数。(V) "
            "2.两个乘数末尾共有2个0，积的末尾也一定有2个0。(x)"
        ),
        child_answer="V；x",
    )

    assert route.kind == MathQuestionKind.GROUPED_TRUE_FALSE
    assert route.evaluation_strategy == "item_split_required"
    assert route.question_type_id == "math_grouped_true_false"


def test_routes_grouped_comparison_block_to_item_split() -> None:
    route = route_math_question_type(
        question_text="在○里填上“>”“<”或“=”。50×40○15×80 63×27○27×85 40×125○100×28",
        child_answer="><>",
    )

    assert route.kind == MathQuestionKind.GROUPED_COMPARISON_SIGN
    assert route.evaluation_strategy == "item_split_required"
    assert route.question_type_id == "math_grouped_comparison_sign"


def test_routes_oral_calculation_from_ocr_action_evidence() -> None:
    route = route_math_question_type(
        question_text="24+36=",
        child_answer="60",
        ocr_action="RecognizeEduOralCalculation",
    )

    assert route.kind == MathQuestionKind.ORAL_CALCULATION
    assert route.evaluation_strategy == "deterministic"
    assert "ocr_action:RecognizeEduOralCalculation" in route.evidence

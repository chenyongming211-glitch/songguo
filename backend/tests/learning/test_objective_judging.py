from __future__ import annotations

from songguo.backend.services.learning.objective_judging import judge_objective_math_item


def test_judges_direct_arithmetic_oral_item() -> None:
    result = judge_objective_math_item(
        question_text="48 ÷ 6 = ?",
        child_answer="8",
    )

    assert result is not None
    assert result.correct is True
    assert result.correct_answer == "8"
    assert result.question_type_id == "math_oral_calculation"


def test_does_not_judge_embedded_multi_blank_arithmetic_as_oral_item() -> None:
    result = judge_objective_math_item(
        question_text="口算21×50时，可以先算21×5=( )，再在积的后面添上( )个0。",
        child_answer="1055",
    )

    assert result is None


def test_judges_single_comparison_sign_item() -> None:
    result = judge_objective_math_item(
        question_text="50×40( )15×80",
        child_answer=">",
    )

    assert result is not None
    assert result.correct is True
    assert result.correct_answer == ">"
    assert result.question_type_id == "math_comparison_sign"


def test_judges_comparison_item_with_plain_number_side() -> None:
    result = judge_objective_math_item(
        question_text="3600 ○ 36×100",
        child_answer="=",
    )

    assert result is not None
    assert result.correct is True
    assert result.correct_answer == "="
    assert result.question_type_id == "math_comparison_sign"


def test_does_not_judge_grouped_comparison_block_before_item_split() -> None:
    result = judge_objective_math_item(
        question_text="50x40()15x80 63×27( )27×85 40×125( )100×28",
        child_answer="><>",
    )

    assert result is None


def test_judges_approximate_product_choice_item() -> None:
    result = judge_objective_math_item(
        question_text="积大约是5600的算式是( )。 A.59×79 B.79×61 C.79×71",
        child_answer="C",
    )

    assert result is not None
    assert result.correct is True
    assert result.correct_answer == "C"
    assert result.question_type_id == "math_choice_estimation"


def test_judges_highest_place_value_choice_item() -> None:
    result = judge_objective_math_item(
        question_text="下列各题中，( )的积的最高位是百位。 A.34×23 B.68×32 C.82×38",
        child_answer="A",
    )

    assert result is not None
    assert result.correct is True
    assert result.correct_answer == "A"


def test_judges_blank_digit_product_choice_item() -> None:
    result = judge_objective_math_item(
        question_text="156×□3的积可能是( ) A.2028 B.4992 C.1208",
        child_answer="A",
    )

    assert result is not None
    assert result.correct is True
    assert result.correct_answer == "A"


def test_judges_direct_arithmetic_choice_item() -> None:
    result = judge_objective_math_item(
        question_text="根据22×28=616，35×35=1225，56×54=3024，找规律计算：62×68=( )。 A.3616 B.1642 C.4216",
        child_answer="C",
    )

    assert result is not None
    assert result.correct is True
    assert result.correct_answer == "C"


def test_judges_direct_arithmetic_choice_item_with_trailing_work_text() -> None:
    result = judge_objective_math_item(
        question_text="根据22×28=616，找规律计算：62×68=( )。 A.3616 B.1642 C.4216 解题过程：35×35=1225 56×54=3024",
        child_answer="C",
    )

    assert result is not None
    assert result.correct is True
    assert result.correct_answer == "C"


def test_judges_computable_fill_blank_item() -> None:
    result = judge_objective_math_item(
        question_text="87×23的积是( )位数，12×55的积的末尾有( )个0。",
        child_answer="四；1",
    )

    assert result is not None
    assert result.correct is True
    assert result.correct_answer == "四；1"
    assert result.question_type_id == "math_fill_blank_calculation"


def test_judges_multiplication_phrase_fill_blank_with_embedded_ocr_answer() -> None:
    result = judge_objective_math_item(
        question_text="24个11的和是( )，42的400倍是( (16800)",
        child_answer="264",
        ocr_action="RecognizeEduPaperCut",
    )

    assert result is not None
    assert result.correct is True
    assert result.correct_answer == "264；16800"
    assert result.question_type_id == "math_fill_blank_calculation"


def test_judges_embedded_direct_arithmetic_answer_without_answer_field() -> None:
    result = judge_objective_math_item(
        question_text="口算21×50时，可以先算21×5=( 105",
        child_answer=None,
        ocr_action="RecognizeEduPaperCut",
    )

    assert result is not None
    assert result.correct is True
    assert result.correct_answer == "105"
    assert result.question_type_id == "math_fill_blank_calculation"


def test_judges_simple_multiplication_word_fill_item() -> None:
    result = judge_objective_math_item(
        question_text="一盒月饼12个，王老师买了22盒，一共买了( )个。",
        child_answer="264",
    )

    assert result is not None
    assert result.correct is True
    assert result.correct_answer == "264"
    assert result.question_type_id == "math_fill_blank_word_problem"


def test_judges_multiplication_word_fill_item_with_leading_context_numbers() -> None:
    cases = [
        (
            "5.[新情境·传统文化]一盒月饼12个，王老师买了22盒，一共买了( )个。",
            "264",
            "264",
        ),
        (
            "6.明明平均每分钟打54个字，他110分钟可以打( )个字。",
            "5940",
            "5940",
        ),
    ]

    for question_text, child_answer, correct_answer in cases:
        result = judge_objective_math_item(
            question_text=question_text,
            child_answer=child_answer,
        )

        assert result is not None
        assert result.correct is True
        assert result.correct_answer == correct_answer
        assert result.question_type_id == "math_fill_blank_word_problem"


def test_judges_simple_multiplication_word_answer_from_problem_factors() -> None:
    result = judge_objective_math_item(
        question_text=(
            "有一种环保矿泉水瓶再生笔，10支为一套，每套99元，"
            "买12套需要多少元? 解题过程：99×12=1188(元)"
        ),
        child_answer="1188元",
    )

    assert result is not None
    assert result.correct is True
    assert result.correct_answer == "1188"
    assert result.question_type_id == "math_multiplication_word_problem"


def test_judges_wrong_multiplication_word_answer_from_problem_factors() -> None:
    result = judge_objective_math_item(
        question_text=(
            "一台智能垃圾回收柜每天回收垃圾3750千克，小区投放了22台智能垃圾回收柜，"
            "它们一天最多回收多少千克垃圾? 解题过程：50×22=1100(千克)"
        ),
        child_answer="1100千克",
    )

    assert result is not None
    assert result.correct is False
    assert result.correct_answer == "82500"
    assert result.question_type_id == "math_multiplication_word_problem"


def test_judges_unit_rate_word_answer_without_using_leading_unit_one_as_factor() -> None:
    result = judge_objective_math_item(
        question_text="1吨废纸可生产850千克再生纸，那么35吨废纸可生产多少千克再生纸?",
        child_answer="29750千克",
    )

    assert result is not None
    assert result.correct is True
    assert result.correct_answer == "29750"
    assert result.question_type_id == "math_multiplication_word_problem"


def test_judges_unit_conversion_fill_blank_items() -> None:
    result = judge_objective_math_item(
        question_text="5年=( )个月 3时=( )分 48时=( )日 2世纪=( )年",
        child_answer="0；180；2；200",
    )

    assert result is not None
    assert result.correct is False
    assert result.correct_answer == "60；180；2；200"
    assert result.question_type_id == "math_unit_conversion"


def test_judges_same_month_daily_count_fill_blank() -> None:
    result = judge_objective_math_item(
        question_text="一部动画片从9月5日开始每天播放1集，9月28日播放大结局，这部动画片一共播放了( )集。",
        child_answer="24",
    )

    assert result is not None
    assert result.correct is True
    assert result.correct_answer == "24"
    assert result.question_type_id == "math_date_count_word_problem"


def test_judges_cross_month_daily_fee_word_answer() -> None:
    result = judge_objective_math_item(
        question_text="天天参加少年军事体验营，从10月25日开始，到11月15日结束。每天活动经费为8元，一共需要多少元的活动经费?",
        child_answer="176元",
    )

    assert result is not None
    assert result.correct is True
    assert result.correct_answer == "176"
    assert result.question_type_id == "math_date_fee_word_problem"


def test_judges_same_day_time_interval_word_answer() -> None:
    result = judge_objective_math_item(
        question_text="7月26日一共开放多长时间? 20：30-8：30=12小时",
        child_answer="12小时",
    )

    assert result is not None
    assert result.correct is True
    assert result.correct_answer == "12小时"
    assert result.question_type_id == "math_time_interval_word_problem"


def test_judges_twenty_four_hour_to_twelve_hour_conversion_with_embedded_answer() -> None:
    result = judge_objective_math_item(
        question_text="17：30用12时记时法表示是( 下午5：30。",
        child_answer=None,
    )

    assert result is not None
    assert result.correct is True
    assert result.correct_answer == "下午5:30"
    assert result.question_type_id == "math_time_format_conversion"


def test_judges_can_enter_before_closing_time_answer() -> None:
    result = judge_objective_math_item(
        question_text="王叔叔7月28日下午3时到达博览会门口，他还能进去吗? 17：30=5：30 3时<5：30。",
        child_answer="他还能去",
    )

    assert result is not None
    assert result.correct is True
    assert result.correct_answer == "能"
    assert result.question_type_id == "math_time_comparison_word_problem"


def test_judges_rectangular_queue_people_count() -> None:
    result = judge_objective_math_item(
        question_text="同学们排成方队做操，小慧在左起第9列、右起第17列，她前面有5人，后面有12人。一共有多少人排队做操?",
        child_answer="450人",
    )

    assert result is not None
    assert result.correct is True
    assert result.correct_answer == "450"
    assert result.question_type_id == "math_rectangular_queue_word_problem"


def test_judges_simple_true_false_item() -> None:
    result = judge_objective_math_item(
        question_text="钟面上时针转两圈是24小时。",
        child_answer="√",
    )

    assert result is not None
    assert result.correct is True
    assert result.correct_answer == "√"
    assert result.question_type_id == "math_true_false_fact"


def test_judges_common_multiplication_true_false_items() -> None:
    cases = [
        ("两位数乘两位数，积可能是三位数，也可能是四位数。", "√", "√"),
        ("两个乘数末尾共有2个0，积的末尾也一定有2个0。", "×", "×"),
        ("两个数的积一定大于这两个数的和。", "×", "×"),
        ("最大的两位数与最小的三位数的乘积是9990。", "×", "×"),
    ]

    for question_text, child_answer, correct_answer in cases:
        result = judge_objective_math_item(
            question_text=question_text,
            child_answer=child_answer,
        )

        assert result is not None
        assert result.correct is True
        assert result.correct_answer == correct_answer
        assert result.question_type_id == "math_true_false_fact"


def test_judges_true_false_item_with_embedded_ocr_mark_without_answer_field() -> None:
    result = judge_objective_math_item(
        question_text="最大的两位数与最小的三位数的乘积是9990。 ( ×",
        child_answer=None,
    )

    assert result is not None
    assert result.correct is True
    assert result.correct_answer == "×"
    assert result.question_type_id == "math_true_false_fact"


def test_does_not_judge_grouped_true_false_block_as_single_item() -> None:
    result = judge_objective_math_item(
        question_text=(
            "二、判断。1.两位数乘两位数，积可能是三位数，也可能是四位数。( V ) "
            "2.两个乘数末尾共有2个0，积的末尾也一定有2个0。( x ) "
            "3.两个数的积一定大于这两个数的和。( X ) "
            "4.最大的两位数与最小的三位数的乘积是9990。( × )"
        ),
        child_answer="×",
    )

    assert result is None

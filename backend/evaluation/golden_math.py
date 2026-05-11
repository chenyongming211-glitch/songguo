from __future__ import annotations

from pydantic import BaseModel, Field


class GoldenMathQuestion(BaseModel):
    question_id: str
    grade: int
    subject: str = "math"
    category: str
    question_text: str
    skill_ids: list[str] = Field(default_factory=list)
    misconception_ids: list[str] = Field(default_factory=list)
    final_answer: str


_CATEGORY_BUILDERS = {
    "calculation": lambda i: GoldenMathQuestion(
        question_id=f"g_calc_{i:03d}",
        grade=_grade_for(i),
        category="calculation",
        question_text=f"{20 + i} × {2 + i % 7} = ?",
        skill_ids=["math_two_digit_times_one_digit"],
        misconception_ids=["math_multiplication_carry_missing"],
        final_answer=str((20 + i) * (2 + i % 7)),
    ),
    "multi_digit_division": lambda i: GoldenMathQuestion(
        question_id=f"g_division_{i:03d}",
        grade=_grade_for(i + 1),
        category="multi_digit_division",
        question_text=f"{240 + i * 12} ÷ {6 + i % 4} = ?",
        skill_ids=["math_division_algorithm"],
        misconception_ids=["math_division_place_value_error"],
        final_answer=str((240 + i * 12) // (6 + i % 4)),
    ),
    "word_problem": lambda i: GoldenMathQuestion(
        question_id=f"g_word_{i:03d}",
        grade=_grade_for(i + 2),
        category="word_problem",
        question_text=f"每盒有{6 + i}支铅笔，买了{3 + i % 5}盒，一共有多少支？",
        skill_ids=["math_multiplication_total_count", "math_word_problem_equation"],
        misconception_ids=["math_word_target_misread"],
        final_answer=f"{(6 + i) * (3 + i % 5)}支",
    ),
    "unit_conversion": lambda i: GoldenMathQuestion(
        question_id=f"g_unit_{i:03d}",
        grade=_grade_for(i + 3),
        category="unit_conversion",
        question_text=f"{1 + i % 8}米{10 + i}厘米一共是多少厘米？",
        skill_ids=["math_unit_conversion"],
        misconception_ids=["math_unit_missing_conversion"],
        final_answer=f"{(1 + i % 8) * 100 + 10 + i}厘米",
    ),
    "time_money": lambda i: GoldenMathQuestion(
        question_id=f"g_time_money_{i:03d}",
        grade=_grade_for(i),
        category="time_money",
        question_text=f"电影{8 + i % 5}:20开始，经过{40 + i}分钟结束，结束时间是几点几分？",
        skill_ids=["math_time_calculation"],
        misconception_ids=["math_time_hour_minute_carry"],
        final_answer="按60进制计算",
    ),
    "geometry": lambda i: GoldenMathQuestion(
        question_id=f"g_geometry_{i:03d}",
        grade=_grade_for(i + 1),
        category="geometry",
        question_text=f"一个长方形长{8 + i}厘米，宽{4 + i % 6}厘米，周长是多少厘米？",
        skill_ids=["math_rectangle_square_perimeter", "math_length_perimeter"],
        misconception_ids=["math_perimeter_area_confusion"],
        final_answer=f"{2 * ((8 + i) + (4 + i % 6))}厘米",
    ),
    "area_volume": lambda i: GoldenMathQuestion(
        question_id=f"g_area_volume_{i:03d}",
        grade=_grade_for(i + 2),
        category="area_volume",
        question_text=f"一个长方体长{6 + i}厘米，宽{3 + i % 4}厘米，高{2 + i % 3}厘米，体积是多少立方厘米？",
        skill_ids=["math_volume_rectangular_prism", "math_area_volume_formula"],
        misconception_ids=["math_area_volume_confusion"],
        final_answer=f"{(6 + i) * (3 + i % 4) * (2 + i % 3)}立方厘米",
    ),
    "remainder_division": lambda i: GoldenMathQuestion(
        question_id=f"g_remainder_{i:03d}",
        grade=_grade_for(i + 3),
        category="remainder_division",
        question_text=f"{50 + i}个苹果平均装进每袋{6 + i % 5}个，可以装满几袋，还剩几个？",
        skill_ids=["math_remainder_division"],
        misconception_ids=["math_division_quotient_remainder_swap"],
        final_answer="商和余数",
    ),
    "capacity_round_up": lambda i: GoldenMathQuestion(
        question_id=f"g_capacity_{i:03d}",
        grade=_grade_for(i),
        category="capacity_round_up",
        question_text=f"学校春游有{3 + i % 5}个班，每班{26 + i}人。每辆大巴车限坐{40 + i % 8}人，至少需要多少辆大巴车？",
        skill_ids=["math_capacity_round_up", "math_multiplication_total_count"],
        misconception_ids=["math_capacity_ignored_remainder_round_up"],
        final_answer="向上取整",
    ),
    "fraction_decimal": lambda i: GoldenMathQuestion(
        question_id=f"g_fraction_decimal_{i:03d}",
        grade=_grade_for(i + 1),
        category="fraction_decimal",
        question_text=f"把一个蛋糕平均分成{4 + i % 5}份，吃了{1 + i % 3}份，吃了几分之几？",
        skill_ids=["math_fraction_intro"],
        misconception_ids=["math_fraction_not_equal_parts"],
        final_answer="分数表示",
    ),
    "fraction_operations": lambda i: GoldenMathQuestion(
        question_id=f"g_fraction_ops_{i:03d}",
        grade=_grade_for(i + 2),
        category="fraction_operations",
        question_text=f"一根绳子长{6 + i}米，先用去全长的1/{2 + i % 3}，还剩全长的几分之几？",
        skill_ids=["math_fraction_operations"],
        misconception_ids=["math_fraction_whole_part_confusion"],
        final_answer="分数减法",
    ),
    "decimal_operations": lambda i: GoldenMathQuestion(
        question_id=f"g_decimal_ops_{i:03d}",
        grade=_grade_for(i + 3),
        category="decimal_operations",
        question_text=f"{3 + i}.5 + {1 + i % 5}.8 等于多少？",
        skill_ids=["math_decimal_operations"],
        misconception_ids=["math_decimal_place_value_error"],
        final_answer="小数加法",
    ),
    "percent_ratio": lambda i: GoldenMathQuestion(
        question_id=f"g_percent_ratio_{i:03d}",
        grade=_grade_for(i),
        category="percent_ratio",
        question_text=f"六年级有{80 + i * 4}人，其中参加社团的占25%，参加社团的有多少人？",
        skill_ids=["math_percent_ratio"],
        misconception_ids=["math_percent_as_whole_number"],
        final_answer=f"{(80 + i * 4) // 4}人",
    ),
    "ratio_proportion": lambda i: GoldenMathQuestion(
        question_id=f"g_ratio_{i:03d}",
        grade=_grade_for(i + 1),
        category="ratio_proportion",
        question_text=f"果汁和水按1:{3 + i % 4}调配，如果果汁有{2 + i}杯，水需要多少杯？",
        skill_ids=["math_ratio_proportion"],
        misconception_ids=["math_ratio_part_whole_confusion"],
        final_answer=f"{(2 + i) * (3 + i % 4)}杯",
    ),
    "average": lambda i: GoldenMathQuestion(
        question_id=f"g_average_{i:03d}",
        grade=_grade_for(i + 2),
        category="average",
        question_text=f"小组4次数学练习分别得{80 + i}分、{82 + i}分、{84 + i}分、{86 + i}分，平均分是多少？",
        skill_ids=["math_average"],
        misconception_ids=["math_average_sum_count_error"],
        final_answer=f"{83 + i}分",
    ),
    "equation": lambda i: GoldenMathQuestion(
        question_id=f"g_equation_{i:03d}",
        grade=_grade_for(i + 3),
        category="equation",
        question_text=f"一个数加上{8 + i}等于{30 + i * 2}，这个数是多少？",
        skill_ids=["math_simple_equation"],
        misconception_ids=["math_inverse_operation_error"],
        final_answer=str(30 + i * 2 - (8 + i)),
    ),
    "statistics": lambda i: GoldenMathQuestion(
        question_id=f"g_statistics_{i:03d}",
        grade=_grade_for(i),
        category="statistics",
        question_text=f"五天阅读页数分别是{10+i}、{12+i}、{9+i}、{14+i}、{15+i}页，一共读了多少页？",
        skill_ids=["math_statistics_data_reading"],
        misconception_ids=["math_data_omission"],
        final_answer=f"{60 + 5 * i}页",
    ),
    "factor_multiple": lambda i: GoldenMathQuestion(
        question_id=f"g_factor_multiple_{i:03d}",
        grade=_grade_for(i + 1),
        category="factor_multiple",
        question_text=f"每{3 + i % 3}天浇一次花，每{4 + i % 4}天施一次肥，至少几天后两件事会再次同一天发生？",
        skill_ids=["math_lcm"],
        misconception_ids=["math_factor_multiple_confusion"],
        final_answer="最小公倍数",
    ),
    "angle_geometry": lambda i: GoldenMathQuestion(
        question_id=f"g_angle_{i:03d}",
        grade=_grade_for(i + 2),
        category="angle_geometry",
        question_text=f"一个三角形两个内角分别是{40 + i}°和{60 + i % 10}°，第三个角是多少度？",
        skill_ids=["math_triangle_angle_sum"],
        misconception_ids=["math_angle_sum_missing"],
        final_answer=f"{180 - (40 + i) - (60 + i % 10)}°",
    ),
    "mixed_operations": lambda i: GoldenMathQuestion(
        question_id=f"g_mixed_{i:03d}",
        grade=_grade_for(i + 3),
        category="mixed_operations",
        question_text=f"{10 + i}+{3 + i % 5}×{4 + i % 6} 等于多少？",
        skill_ids=["math_mixed_operations", "math_parentheses_priority"],
        misconception_ids=["math_mixed_order_wrong"],
        final_answer="先乘后加",
    ),
    "distractor_condition": lambda i: GoldenMathQuestion(
        question_id=f"g_distractor_{i:03d}",
        grade=_grade_for(i),
        category="distractor_condition",
        question_text=f"小明有{20 + i}张贴纸，送给妹妹{5 + i % 4}张，又买来{8 + i % 6}张。哥哥有30张是无关信息。小明现在有多少张？",
        skill_ids=["math_read_conditions", "math_word_problem_equation"],
        misconception_ids=["math_word_irrelevant_condition"],
        final_answer=f"{20 + i - (5 + i % 4) + (8 + i % 6)}张",
    ),
}


def build_golden_math_questions() -> list[GoldenMathQuestion]:
    questions: list[GoldenMathQuestion] = []
    for offset in range(5):
        for _category, builder in _CATEGORY_BUILDERS.items():
            questions.append(builder(offset + 1))
    return questions[:100]


def _grade_for(i: int) -> int:
    return 3 + ((i - 1) % 4)

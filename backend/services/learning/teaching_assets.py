from __future__ import annotations

from pydantic import BaseModel, Field


class MathSkill(BaseModel):
    skill_id: str
    name: str
    subject: str = "math"
    grade_min: int = 1
    grade_max: int = 6
    definition: str
    aliases: list[str] = Field(default_factory=list)


class MathMisconception(BaseModel):
    misconception_id: str
    name: str
    explanation: str
    parent_explanation: str
    evidence_examples: list[str] = Field(default_factory=list)
    skill_ids: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)


class ConceptCard(BaseModel):
    card_id: str
    skill_id: str
    title: str
    grade_range: str = "1-6"
    concept_explanation: str
    simple_example: str
    common_mistake: str
    parent_tip: str
    version: str = "v0.1"


class TeachingAssetLibrary(BaseModel):
    skills: list[MathSkill]
    misconceptions: list[MathMisconception]
    concept_cards: list[ConceptCard]

    def require_skill(self, skill_id: str) -> MathSkill:
        for skill in self.skills:
            if skill.skill_id == skill_id:
                return skill
        raise KeyError(skill_id)

    def require_misconception(self, misconception_id: str) -> MathMisconception:
        for misconception in self.misconceptions:
            if misconception.misconception_id == misconception_id:
                return misconception
        raise KeyError(misconception_id)

    def require_concept_card_for_skill(self, skill_id: str) -> ConceptCard:
        for card in self.concept_cards:
            if card.skill_id == skill_id:
                return card
        raise KeyError(skill_id)

    def require_concept_card(self, card_id: str) -> ConceptCard:
        for card in self.concept_cards:
            if card.card_id == card_id:
                return card
        raise KeyError(card_id)

    def match_skill_ids(self, labels: list[str]) -> list[str]:
        text = " ".join(labels)
        result: list[str] = []
        for skill in self.skills:
            tokens = [skill.name, skill.skill_id, *skill.aliases]
            if any(token and token in text for token in tokens):
                result.append(skill.skill_id)
        return _dedupe(result)

    def match_misconception_ids(self, labels: list[str]) -> list[str]:
        text = " ".join(labels)
        result: list[str] = []
        for misconception in self.misconceptions:
            tokens = [
                misconception.name,
                misconception.misconception_id,
                *misconception.aliases,
            ]
            if any(token and token in text for token in tokens):
                result.append(misconception.misconception_id)
        return _dedupe(result)

    def concept_card_ids_for_skills(self, skill_ids: list[str]) -> list[str]:
        return _dedupe(
            [card.card_id for card in self.concept_cards if card.skill_id in skill_ids]
        )

    def parent_explanation_for_misconception(self, misconception_id: str | None) -> str:
        if not misconception_id:
            return "孩子这次还没有形成稳定错因，需要继续观察。"
        try:
            return self.require_misconception(misconception_id).parent_explanation
        except KeyError:
            return "孩子这次的错因还需要通过后续练习继续确认。"


_SKILL_SPECS: list[tuple[str, str, int, int, str, list[str]]] = [
    ("math_two_digit_times_one_digit", "两位数乘一位数", 3, 4, "两位数和一位数相乘，理解拆数和进位。", ["two_digit_times_one_digit"]),
    ("math_three_digit_times_one_digit", "三位数乘一位数", 3, 4, "三位数和一位数相乘，按位计算并处理进位。", []),
    ("math_multi_digit_times_one_digit", "多位数乘一位数", 3, 5, "多位数乘一位数，保持数位和进位清楚。", []),
    ("math_remainder_division", "有余数除法", 3, 4, "理解商和余数，知道余数必须小于除数。", ["余数"]),
    ("math_multiplication_total_count", "乘法求总数", 2, 4, "用每份数量乘份数求总量。", ["乘法求总数", "先求总人数"]),
    ("math_equal_grouping", "平均分问题", 2, 4, "把总数平均分成若干份，求每份或份数。", []),
    ("math_capacity_round_up", "限载进一问题", 3, 6, "遇到“至少需要”且有剩余时，需要多加一组。", ["限载进一", "至少", "进一法"]),
    ("math_multiple_relationship", "倍数关系", 3, 5, "理解一个量是另一个量的几倍。", []),
    ("math_sum_difference", "和差问题", 3, 6, "根据两个量的和与差求各自大小。", []),
    ("math_unit_rate", "归一问题", 3, 6, "先求单一份量，再求多个份量。", []),
    ("math_totalization", "归总问题", 3, 6, "先求总量，再按新条件分配或比较。", ["归总"]),
    ("math_unit_conversion", "单位换算", 2, 6, "在长度、质量、容量等单位之间换算。", []),
    ("math_time_calculation", "时间计算", 2, 6, "计算经过时间、开始时间或结束时间。", []),
    ("math_money_calculation", "人民币计算", 1, 4, "理解元角分和购物找零。", []),
    ("math_length_perimeter", "长度周长", 2, 5, "把围一圈的边长相加得到周长。", []),
    ("math_area_calculation", "面积计算", 3, 6, "理解面积是平面大小，用公式或分割计算。", []),
    ("math_rectangle_square_perimeter", "长方形正方形周长", 3, 5, "长方形周长是两组长宽相加，正方形是四条边相加。", []),
    ("math_rectangle_square_area", "长方形正方形面积", 3, 5, "长方形面积等于长乘宽，正方形面积等于边长乘边长。", []),
    ("math_fraction_intro", "分数初步", 3, 5, "理解把整体平均分后的其中几份。", []),
    ("math_decimal_intro", "小数初步", 3, 5, "理解小数和元角分、长度单位之间的关系。", []),
    ("math_mixed_operations", "四则混合运算", 3, 6, "按先乘除后加减和括号优先级计算。", []),
    ("math_parentheses_priority", "括号优先级", 3, 6, "有括号时先算括号里的内容。", []),
    ("math_estimation", "估算", 2, 6, "用接近的整十整百数快速判断大致结果。", []),
    ("math_checking", "验算", 2, 6, "用逆运算或估算检查结果是否合理。", []),
    ("math_read_conditions", "读题找条件", 1, 6, "从题目中找出已知条件和问题。", []),
    ("math_word_problem_equation", "应用题列式", 2, 6, "把文字关系转成正确算式。", []),
    ("math_shape_condition_extract", "图形条件提取", 2, 6, "从图形中找边长、角、面积等条件。", []),
    ("math_table_reading", "表格信息读取", 2, 6, "从表格中读取行列信息并计算。", []),
    ("math_chart_reading", "统计图读取", 3, 6, "从条形图、折线图等读取数据。", []),
    ("math_attention_checking", "检查习惯", 1, 6, "做完后检查条件、单位、问题和结果。", []),
]


_MISCONCEPTION_SPECS: list[tuple[str, str, str, list[str], list[str]]] = [
    ("math_capacity_stopped_at_total_count", "只算总数就停止", "孩子算出了总人数，但没有继续按限载分组。", ["128", "只写总人数"], ["stopped_at_total_count"]),
    ("math_capacity_ignored_remainder_round_up", "有余数但没有进一", "孩子知道可以先分组，但忽略剩余的人也需要一组。", ["2辆", "2"], ["ignored_remainder_round_up", "有余数但没有进一"]),
    ("math_capacity_copied_capacity", "把限载当总数", "孩子把每辆最多坐的人数当成最终答案。", ["45", "45人"], ["最多坐45人"]),
    ("math_capacity_misread_at_least", "不理解至少需要", "孩子没有把“至少”理解为不能少于实际需求。", ["少算一辆"], ["至少"]),
    ("math_capacity_divided_classes_by_capacity", "用班级数除以限载", "孩子没有先求人数，直接用班级数参与除法。", ["4÷45"], []),
    ("math_multiplication_treated_x5_like_x10", "把乘以5当成乘以10", "孩子把乘以5看成乘以10，没有想到一半关系。", ["36×5写成360"], ["treated_x5_like_x10"]),
    ("math_multiplication_missed_ones", "漏加个位贡献", "孩子只算了十位部分，没有加个位部分。", ["30×4"], []),
    ("math_multiplication_carry_missing", "进位遗漏", "孩子乘法过程中有进位但没有加上。", ["个位满十没进位"], []),
    ("math_multiplication_addition_confusion", "乘法加法混淆", "孩子把几个相同数相加和乘法关系混淆。", ["32+4"], []),
    ("math_multiplication_partial_place_value", "数位只算一部分", "孩子只处理了一个数位，结果偏小。", ["只算个位或十位"], []),
    ("math_division_remainder_too_large", "余数大于除数", "孩子写出的余数不符合除法规则。", ["余数比除数大"], []),
    ("math_division_quotient_remainder_swap", "商和余数混淆", "孩子把商和余数的位置写反。", ["2余45"], []),
    ("math_division_equal_group_direction", "平均分方向错", "孩子不知道题目要求每份数还是份数。", ["把份数当每份"], []),
    ("math_word_missed_condition", "漏看关键条件", "孩子没有用到题目中的关键数字或限制。", ["没用45"], ["漏看关键条件"]),
    ("math_word_target_misread", "问题目标看错", "孩子算了中间量，没有回答题目真正问什么。", ["答总人数"], []),
    ("math_word_each_as_total", "把每份数量当总数", "孩子把“每班32人”当成总人数。", ["32人"], []),
    ("math_word_total_as_each", "把总数当每份", "孩子把总量误认为每份数量。", ["每份用总数"], []),
    ("math_word_irrelevant_condition", "被无关条件干扰", "孩子把题目中的无关数字用进算式。", ["使用干扰数字"], []),
    ("math_unit_missing_conversion", "没有单位换算", "孩子直接计算不同单位的数量。", ["米和厘米直接加"], []),
    ("math_unit_wrong_rate", "换算进率错误", "孩子把10、100、1000等进率用错。", ["1米=10厘米"], []),
    ("math_time_hour_minute_carry", "时间进退位错误", "孩子不熟悉60进制。", ["70分钟写成0时70分"], []),
    ("math_time_cross_day", "跨天时间漏算", "孩子遇到跨天时只算同一天。", ["晚上到早上漏一天"], []),
    ("math_money_yuan_jiao_confusion", "元角分混淆", "孩子把10进制金额关系用错。", ["1元=100角"], []),
    ("math_money_change_direction", "找零方向错", "孩子不知道用付的钱减价格。", ["价格减付款"], []),
    ("math_perimeter_area_confusion", "周长面积混淆", "孩子把围一圈和面大小混为一谈。", ["周长用长×宽"], []),
    ("math_perimeter_missing_sides", "周长漏边", "孩子只加了部分边长。", ["长+宽"], []),
    ("math_area_formula_confusion", "面积公式混淆", "孩子把面积公式和周长公式混用。", ["面积=(长+宽)×2"], []),
    ("math_square_rectangle_confusion", "正方形长方形条件混淆", "孩子没有根据图形特点选择公式。", ["正方形只用两条边"], []),
    ("math_fraction_not_equal_parts", "没有平均分意识", "孩子把非平均分也当成分数。", ["不是平均分"], []),
    ("math_fraction_numerator_denominator_swap", "分子分母混淆", "孩子把份数和取的份数写反。", ["3/8写成8/3"], []),
    ("math_decimal_place_value", "小数位值混淆", "孩子不清楚十分位、百分位。", ["0.5和0.05混淆"], []),
    ("math_decimal_align_wrong", "小数点没对齐", "孩子做小数加减时没有按小数点对齐。", ["竖式错位"], []),
    ("math_mixed_order_wrong", "运算顺序错误", "孩子没有先乘除后加减。", ["先算加法"], []),
    ("math_parentheses_ignored", "忽略括号", "孩子没有先算括号里的内容。", ["跳过括号"], []),
    ("math_estimation_over_precise", "不会用估算判断", "孩子只机械计算，不会判断结果范围。", ["结果明显不合理"], []),
    ("math_checking_no_inverse", "不会逆运算验算", "孩子不会用反向计算检查。", ["不验算"], []),
    ("math_table_row_column_confusion", "表格行列读错", "孩子把行和列对应关系看反。", ["读错列"], []),
    ("math_chart_axis_confusion", "统计图坐标读错", "孩子把横轴纵轴或单位读错。", ["看错刻度"], []),
    ("math_shape_hidden_condition", "图形隐含条件没用", "孩子没有使用相等边、直角等隐含条件。", ["漏用相等边"], []),
    ("math_condition_unit_omitted", "答案单位遗漏", "孩子结果正确但没有写单位。", ["只写数字"], []),
    ("math_condition_answer_not_checked", "结果合理性未检查", "孩子没有回到题意检查答案是否合理。", ["车数比人数还多"], []),
    ("math_rate_unit_confusion", "单价数量总价混淆", "孩子不知道单价、数量、总价的关系。", ["单价×单价"], []),
    ("math_average_total_missing", "平均数总量没求对", "孩子求平均数时总量或份数用错。", ["除错份数"], []),
    ("math_comparison_more_less_direction", "多几少几方向错", "孩子搞反谁比谁多或少。", ["用小数减大数"], []),
    ("math_double_counting", "重复计数", "孩子把同一部分重复加了一次。", ["重复加中间量"], []),
    ("math_missing_final_step", "漏掉最后一步", "孩子完成中间计算但没有回答最终问题。", ["缺少最后比较"], []),
    ("math_copy_number_as_answer", "抄题目数字作答", "孩子没有计算，直接复制题目数字。", ["答案是题目数字"], []),
    ("math_expression_not_matching_context", "列式和题意不匹配", "孩子列式数字齐全但关系错误。", ["数字都用了但式子错"], []),
    ("math_unknown", "未知错因", "当前答案暂时无法稳定归因，需要继续追问。", ["不相关答案"], ["unknown_misconception"]),
    ("math_off_task", "偏离题目", "孩子回答与当前题目无关。", ["随意输入"], []),
]


def _build_default_library() -> TeachingAssetLibrary:
    skills = [
        MathSkill(
            skill_id=skill_id,
            name=name,
            grade_min=grade_min,
            grade_max=grade_max,
            definition=definition,
            aliases=aliases,
        )
        for skill_id, name, grade_min, grade_max, definition, aliases in _SKILL_SPECS
    ]
    skill_ids = {skill.skill_id for skill in skills}
    misconceptions = [
        MathMisconception(
            misconception_id=misconception_id,
            name=name,
            explanation=explanation,
            parent_explanation=f"家长可以关注：{explanation}",
            evidence_examples=evidence_examples,
            skill_ids=_infer_skill_ids(misconception_id, skill_ids),
            aliases=aliases,
        )
        for misconception_id, name, explanation, evidence_examples, aliases in _MISCONCEPTION_SPECS
    ]
    concept_cards = [_concept_card_for_skill(skill) for skill in skills]
    return TeachingAssetLibrary(
        skills=skills,
        misconceptions=misconceptions,
        concept_cards=concept_cards,
    )


def _infer_skill_ids(misconception_id: str, all_skill_ids: set[str]) -> list[str]:
    mapping = {
        "math_capacity": ["math_capacity_round_up", "math_remainder_division"],
        "math_multiplication": ["math_two_digit_times_one_digit", "math_multiplication_total_count"],
        "math_division": ["math_remainder_division", "math_equal_grouping"],
        "math_word": ["math_read_conditions", "math_word_problem_equation"],
        "math_unit": ["math_unit_conversion"],
        "math_time": ["math_time_calculation"],
        "math_money": ["math_money_calculation"],
        "math_perimeter": ["math_length_perimeter", "math_rectangle_square_perimeter"],
        "math_area": ["math_area_calculation", "math_rectangle_square_area"],
        "math_fraction": ["math_fraction_intro"],
        "math_decimal": ["math_decimal_intro"],
        "math_mixed": ["math_mixed_operations"],
        "math_parentheses": ["math_parentheses_priority"],
        "math_table": ["math_table_reading"],
        "math_chart": ["math_chart_reading"],
        "math_shape": ["math_shape_condition_extract"],
    }
    for prefix, skill_ids in mapping.items():
        if misconception_id.startswith(prefix):
            return [skill_id for skill_id in skill_ids if skill_id in all_skill_ids]
    return ["math_attention_checking"]


def _concept_card_for_skill(skill: MathSkill) -> ConceptCard:
    if skill.skill_id == "math_capacity_round_up":
        return ConceptCard(
            card_id="card_math_capacity_round_up_v01",
            skill_id=skill.skill_id,
            title="什么叫至少需要几组",
            grade_range="3-6",
            concept_explanation="如果还有人或物没有被装下，就需要再增加一组。",
            simple_example="一批人坐车，前几辆坐满后还剩少量人，这些人也需要再安排一辆。",
            common_mistake="只看到商是2，就忘了剩下的人也要坐车。",
            parent_tip="让孩子先圈出总数、每组最多数，再问剩下的是否也需要安排。",
        )
    return ConceptCard(
        card_id=f"card_{skill.skill_id}_v01",
        skill_id=skill.skill_id,
        title=f"{skill.name}方法卡",
        grade_range=f"{skill.grade_min}-{skill.grade_max}",
        concept_explanation=skill.definition,
        simple_example=f"先用一道更小的{skill.name}例题说清方法，再回到原题。",
        common_mistake="容易跳过读题、列式或检查步骤。",
        parent_tip="让孩子先说出条件和问题，再动笔计算。",
    )


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


DEFAULT_MATH_ASSET_LIBRARY = _build_default_library()

from __future__ import annotations

from songguo.backend.services.learning.math_structuring import (
    LLMProblemParse,
    LLMMathStructurer,
    MathFastPathRegistry,
    MathProblemStructuringGateway,
    ProblemAnalysis,
    build_problem_analysis_from_parse,
)


BUS_QUESTION = "学校组织三年级学生春游，一共有4个班，每班32人。如果每辆大巴车限坐45人，那么至少需要多少辆大巴车？"


def test_structuring_gateway_returns_key_points_from_provider() -> None:
    class FixtureStructurer:
        def analyze(self, *, question_text: str, grade: int, subject: str) -> ProblemAnalysis:
            assert question_text == BUS_QUESTION
            assert grade == 3
            assert subject == "math"
            return ProblemAnalysis(
                subject="math",
                grade=3,
                problem_type="capacity_round_up",
                knowledge_points=["乘法求总数", "除法分组", "有余数进一"],
                target="至少需要多少辆大巴车",
                final_answer="3辆",
                confidence=0.94,
                source="fixture_llm_structured",
                conditions=[
                    {"id": "class_count", "text": "一共有4个班", "value": 4, "unit": "班"},
                    {
                        "id": "students_per_class",
                        "text": "每班32人",
                        "value": 32,
                        "unit": "人/班",
                    },
                    {
                        "id": "bus_capacity",
                        "text": "每辆大巴车限坐45人",
                        "value": 45,
                        "unit": "人/辆",
                    },
                ],
                solution_steps=[
                    {
                        "id": "step_total_people",
                        "goal": "先求总人数",
                        "expression": "4 × 32",
                        "result": "128",
                    },
                    {
                        "id": "step_capacity_check",
                        "goal": "判断车辆容量",
                        "expression": "128 ÷ 45",
                        "result": "2余38",
                    },
                    {
                        "id": "step_round_up",
                        "goal": "有余数要进一",
                        "expression": "2 + 1",
                        "result": "3",
                    },
                ],
                common_misconceptions=[
                    {"tag": "stopped_at_total_count", "description": "只算出总人数就停止"},
                    {"tag": "ignored_remainder_round_up", "description": "有余数但没有进一"},
                ],
                key_points=[
                    {
                        "id": "kp_total_people",
                        "name": "先求总人数",
                        "teaching_goal": "理解要先算出一共有多少人",
                        "release_stage": "HINT_STEP_1",
                        "unlock_condition": "question_started",
                        "child_prompt": "先不急着算车。题目说有4个班，每班32人，你先想一想一共有多少人？",
                        "expected_child_response": ["128", "128人"],
                        "forbidden_content": ["3辆"],
                        "required_before_next": True,
                    },
                    {
                        "id": "kp_capacity_check",
                        "name": "判断车辆容量",
                        "teaching_goal": "理解要判断车辆能不能坐下",
                        "release_stage": "HINT_STEP_2",
                        "unlock_condition": "child_found_total_people",
                        "child_prompt": "你已经算出总人数了。现在想想：2辆车最多能坐多少人？",
                        "expected_child_response": ["90", "90人"],
                        "misconception_responses": {"ignored_remainder_round_up": ["2", "2辆"]},
                        "forbidden_content": ["3辆"],
                        "required_before_next": True,
                    },
                    {
                        "id": "kp_round_up",
                        "name": "有剩余也要加一辆",
                        "teaching_goal": "理解至少需要时，有余数要进一",
                        "release_stage": "HINT_STEP_3",
                        "unlock_condition": "child_tried_vehicle_count",
                        "child_prompt": "如果还有人没坐上，这些人是不是也需要一辆车？",
                        "expected_child_response": ["需要", "还要一辆", "3"],
                        "forbidden_content": [],
                        "required_before_next": True,
                    },
                ],
            )

    gateway = MathProblemStructuringGateway(structurer=FixtureStructurer())

    analysis = gateway.analyze(question_text=BUS_QUESTION, grade=3, subject="math")
    evaluation = gateway.evaluate_attempt(analysis, child_answer="128")

    assert analysis.problem_type == "capacity_round_up"
    assert analysis.knowledge_point == "capacity_round_up"
    assert analysis.first_key_point.id == "kp_total_people"
    assert analysis.first_key_point.release_stage == "HINT_STEP_1"
    assert "3辆" not in analysis.first_key_point.child_prompt
    assert evaluation.partially_correct is True
    assert evaluation.matched_key_point_id == "kp_total_people"
    assert evaluation.next_key_point_id == "kp_capacity_check"
    assert evaluation.misconception_tag == "stopped_at_total_count"


def test_llm_math_structurer_parses_minimal_problem_parse_and_backend_builds_key_points() -> None:
    def fake_llm(prompt: str) -> str:
        assert "只输出 JSON" in prompt
        assert "顶层必须是一个 JSON object" in prompt
        assert "不要输出 ```json 代码块" in prompt
        assert "字段缺失时使用空字符串、空数组、false 或 null" in prompt
        assert "LLMProblemParse" in prompt
        assert "不要输出 key_points" in prompt
        assert "不要输出 child_prompt" in prompt
        assert "不要输出 forbidden_content" in prompt
        assert BUS_QUESTION in prompt
        return """
        {
          "subject": "math",
          "grade": 3,
          "problem_type": "capacity_round_up",
          "knowledge_points": ["乘法求总数", "除法分组", "有余数进一"],
          "target": "至少需要多少辆大巴车",
          "final_answer": "3辆",
          "confidence": 0.91,
          "source": "fake_llm",
          "solution_steps": [
            {"id": "step_total_people", "goal": "先求总人数", "expression": "4 × 32", "result": "128"}
          ]
        }
        """

    structurer = LLMMathStructurer(llm_func=fake_llm)

    analysis = structurer.analyze(
        question_text=BUS_QUESTION,
        grade=3,
        subject="math",
    )

    assert analysis.problem_type == "capacity_round_up"
    assert analysis.source == "fake_llm"
    assert analysis.first_key_point.id == "kp_step_total_people"
    assert analysis.first_key_point.expected_child_response == ["128"]
    assert analysis.first_key_point.forbidden_content == ["3辆"]
    assert "3辆" not in analysis.first_key_point.child_prompt


def test_problem_analysis_builder_ignores_model_generated_teaching_prompts() -> None:
    parse = LLMProblemParse(
        subject="math",
        grade=3,
        problem_type="multiplication",
        knowledge_points=["乘法"],
        target="求乘积",
        final_answer="63",
        solution_steps=[
            {"id": "step_1", "goal": "计算乘积", "expression": "21 × 3", "result": "63"}
        ],
        confidence=0.9,
        source="fake_llm",
    )

    analysis = build_problem_analysis_from_parse(
        parse,
        question_text="21 × 3 = ?",
    )

    assert analysis.key_points
    assert analysis.first_key_point.forbidden_content == ["63"]
    assert "63" not in analysis.first_key_point.child_prompt


def test_problem_analysis_builder_strips_result_from_expression_prompt() -> None:
    parse = LLMProblemParse(
        subject="math",
        grade=3,
        problem_type="remainder_division",
        knowledge_points=["有余数除法"],
        target="可以装满几袋，还剩几个",
        final_answer="6袋，还剩4个",
        solution_steps=[
            {"id": "step_1", "goal": "求商和余数", "expression": "52 ÷ 8 = 6……4", "result": "6袋，还剩4个"}
        ],
        confidence=0.9,
        source="fake_llm",
    )

    analysis = build_problem_analysis_from_parse(
        parse,
        question_text="52个苹果平均装进每袋8个，可以装满几袋，还剩几个？",
    )

    assert "52 ÷ 8" in analysis.first_key_point.child_prompt
    assert "6……4" not in analysis.first_key_point.child_prompt
    assert "6袋" not in analysis.first_key_point.child_prompt


def test_problem_analysis_builder_replaces_prompt_when_expression_contains_final_answer() -> None:
    parse = LLMProblemParse(
        subject="math",
        grade=5,
        problem_type="fraction_operations",
        knowledge_points=["分数减法"],
        target="还剩全长的几分之几",
        final_answer="1/2",
        solution_steps=[
            {"id": "kp_1", "goal": "把全长看作1", "expression": "1", "result": "1"},
            {"id": "kp_2", "goal": "求剩下的部分", "expression": "1 - 1/2", "result": "1/2"},
        ],
        confidence=0.9,
        source="fake_llm",
    )

    analysis = build_problem_analysis_from_parse(
        parse,
        question_text="一根绳子长9米，先用去全长的1/2，还剩全长的几分之几？",
    )

    assert "1 - 1/2" not in analysis.key_points[1].child_prompt
    assert "1/2" not in analysis.key_points[1].child_prompt


def test_fast_path_accepts_simple_remainder_expression_from_ocr() -> None:
    gateway = MathProblemStructuringGateway()

    analysis = gateway.analyze(question_text="36 ÷5=?", grade=4, subject="math")
    evaluation = gateway.evaluate_attempt(analysis, child_answer="7余1")

    assert analysis.problem_type == "division_with_remainder"
    assert analysis.final_answer == "7余1"
    assert evaluation.correct is True


def test_fast_path_repairs_duplicate_equals_from_ocr() -> None:
    gateway = MathProblemStructuringGateway()

    analysis = gateway.analyze(question_text="25 × 4= =?", grade=4, subject="math")
    evaluation = gateway.evaluate_attempt(analysis, child_answer="100")

    assert analysis.problem_type == "two_digit_times_one_digit"
    assert analysis.final_answer == "100"
    assert evaluation.correct is True


def test_problem_analysis_builder_replaces_goal_when_goal_contains_final_answer() -> None:
    parse = LLMProblemParse(
        subject="math",
        grade=5,
        problem_type="fraction_operations",
        knowledge_points=["分数减法"],
        target="还剩全长的几分之几",
        final_answer="1/2",
        solution_steps=[
            {"id": "kp_1", "goal": "把全长看作1", "expression": "1", "result": "1"},
            {"id": "kp_2", "goal": "计算 1 - 1/2 = 1/2", "expression": "1 - 1/2", "result": "1/2"},
        ],
        confidence=0.9,
        source="fake_llm",
    )

    analysis = build_problem_analysis_from_parse(
        parse,
        question_text="一根绳子长9米，先用去全长的1/2，还剩全长的几分之几？",
    )

    assert "1/2" not in analysis.key_points[1].child_prompt


def test_llm_math_structurer_normalizes_condition_list_value_to_string() -> None:
    def fake_llm(_prompt: str) -> str:
        return """
        {
          "problem_type": "statistics",
          "knowledge_points": ["数据读取"],
          "conditions": [
            {"id": "daily_pages", "text": "五天阅读页数", "value": [12, 14, 11, 16, 17], "unit": "页"}
          ],
          "target": "一共读了多少页",
          "solution_steps": [
            {"id": "step_1", "goal": "求总页数", "expression": "12 + 14 + 11 + 16 + 17", "result": "70"}
          ],
          "final_answer": "70页",
          "common_misconceptions": [],
          "confidence": 0.9,
          "source": "deepseek"
        }
        """

    analysis = LLMMathStructurer(llm_func=fake_llm).analyze(
        question_text="五天阅读页数分别是12、14、11、16、17页，一共读了多少页？",
        grade=3,
        subject="math",
    )

    assert analysis.conditions[0].value == "12、14、11、16、17"


def test_llm_math_structurer_normalizes_deepseek_style_loose_json() -> None:
    def fake_llm(prompt: str) -> str:
        return """
        {
          "problem_type": "除法应用（进一法）",
          "knowledge_points": ["乘法", "有余数的除法", "进一法"],
          "conditions": [
            "三年级有4个班",
            "每班32人",
            "每辆大巴车限坐45人"
          ],
          "target": "至少需要多少辆大巴车",
          "solution_steps": [
            "计算总人数：4 × 32 = 128（人）",
            "计算需要的车辆数：128 ÷ 45 = 2（辆）……38（人）",
            "因为余下38人也需要一辆车，所以需要 2 + 1 = 3（辆）"
          ],
          "final_answer": "3辆",
          "common_misconceptions": [
            "忘记计算总人数",
            "忘记进一，只写2辆"
          ],
          "key_points": [
            {
              "id": 1,
              "name": "计算总人数",
              "teaching_goal": "学生能正确计算总人数。",
              "release_stage": "HINT_STEP_1",
              "unlock_condition": "无",
              "child_prompt": "学校三年级有4个班，每班32人，你能算出一共有多少人参加春游吗？",
              "expected_child_response": "4×32=128（人）",
              "forbidden_content": "不要直接告诉孩子最终答案。"
            },
            {
              "id": "kp2",
              "name": "处理余数",
              "teaching_goal": "学生能判断有余数时需要多一辆车。",
              "release_stage": "HINT_STEP_2",
              "unlock_condition": "完成关键点1",
              "child_prompt": "如果还有人没坐上，这些人是不是也需要一辆车？",
              "expected_child_response": "需要，所以2+1=3（辆）",
              "forbidden_content": ""
            }
          ],
          "confidence": "高",
          "source": "小学数学教学"
        }
        """

    structurer = LLMMathStructurer(llm_func=fake_llm)

    analysis = structurer.analyze(
        question_text=BUS_QUESTION,
        grade=3,
        subject="math",
    )
    evaluation = MathProblemStructuringGateway(structurer=structurer).evaluate_attempt(
        analysis,
        child_answer="128",
    )

    assert analysis.subject == "math"
    assert analysis.grade == 3
    assert analysis.conditions[0].id == "condition_1"
    assert analysis.conditions[0].text == "三年级有4个班"
    assert analysis.solution_steps[0].expression == "4 × 32"
    assert analysis.solution_steps[0].result == "128"
    assert "math_capacity_round_up" in analysis.skill_ids
    assert "math_capacity_ignored_remainder_round_up" in analysis.misconception_ids
    assert "card_math_capacity_round_up_v01" in analysis.concept_card_ids
    assert analysis.common_misconceptions[0].tag == "math_capacity_stopped_at_total_count"
    assert analysis.common_misconceptions[1].tag == "math_capacity_ignored_remainder_round_up"
    assert analysis.key_points[0].id == "kp_step_1"
    assert analysis.key_points[0].expected_child_response == ["128"]
    assert analysis.key_points[0].forbidden_content == ["3辆"]
    assert analysis.confidence == 0.85
    assert evaluation.partially_correct is True
    assert evaluation.matched_key_point_id == "kp_step_1"
    assert evaluation.misconception_tag == "math_capacity_stopped_at_total_count"


def test_llm_math_structurer_normalizes_numeric_answer_and_object_target() -> None:
    def fake_llm(_prompt: str) -> str:
        return """
        {
          "problem_type": "multiplication",
          "knowledge_points": ["两位数乘一位数"],
          "conditions": [{"id": 1, "text": "21乘3", "value": 21}],
          "target": {"description": "计算21乘3的结果", "value": null},
          "solution_steps": [
            {"id": 1, "goal": "计算乘积", "expression": "21 × 3", "result": 63}
          ],
          "final_answer": 63,
          "common_misconceptions": [],
          "key_points": [
            {
              "id": 1,
              "name": "拆分计算",
              "teaching_goal": "先拆成20和1",
              "release_stage": "HINT_STEP_1",
              "unlock_condition": "question_started",
              "child_prompt": "可以先想20×3是多少，再加上1×3是多少。",
              "expected_child_response": [60, 3],
              "forbidden_content": [63]
            }
          ],
          "confidence": 0.9,
          "source": "deepseek"
        }
        """

    analysis = LLMMathStructurer(llm_func=fake_llm).analyze(
        question_text="21 × 3 = ?",
        grade=3,
        subject="math",
    )

    assert analysis.target == "计算21乘3的结果"
    assert analysis.final_answer == "63"
    assert analysis.solution_steps[0].id == "1"
    assert analysis.solution_steps[0].result == "63"
    assert analysis.key_points[0].expected_child_response == ["63"]
    assert analysis.key_points[0].forbidden_content == ["63"]


def test_llm_math_structurer_normalizes_final_answer_value_unit_object() -> None:
    def fake_llm(_prompt: str) -> str:
        return """
        {
          "problem_type": "word_problem",
          "knowledge_points": ["乘法求总数"],
          "conditions": ["每盒7支铅笔", "买了4盒"],
          "target": {"description": "一共有多少支铅笔"},
          "solution_steps": [
            {"id": "step_1", "goal": "求总数", "expression": "7 × 4", "result": 28}
          ],
          "final_answer": {"value": 28, "unit": "支"},
          "common_misconceptions": [],
          "confidence": 0.9,
          "source": "deepseek"
        }
        """

    analysis = LLMMathStructurer(llm_func=fake_llm).analyze(
        question_text="每盒有7支铅笔，买了4盒，一共有多少支？",
        grade=5,
        subject="math",
    )

    assert analysis.final_answer == "28支"
    assert analysis.target == "一共有多少支铅笔"
    assert analysis.first_key_point.forbidden_content == ["28支"]
    assert "28支" not in analysis.first_key_point.child_prompt


def test_llm_math_structurer_default_completion_uses_langchain_json_client(monkeypatch) -> None:
    calls: list[dict[str, str]] = []

    class FakeLangChainClient:
        def complete_sync(self, prompt: str, *, system_prompt: str, temperature: float) -> str:
            calls.append(
                {
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "temperature": str(temperature),
                }
            )
            return """
            {
              "problem_type": "multiplication",
              "knowledge_points": ["乘法"],
              "conditions": ["21 × 3"],
              "target": "求乘积",
              "solution_steps": [
                {"id": "step_1", "goal": "计算乘积", "expression": "21 × 3", "result": "63"}
              ],
              "final_answer": "63",
              "common_misconceptions": [],
              "confidence": 0.9,
              "source": "fake_langchain"
            }
            """

    monkeypatch.setattr(
        "songguo.backend.services.learning.langchain_model_client.get_langchain_llm_client",
        lambda: FakeLangChainClient(),
    )

    analysis = LLMMathStructurer().analyze(
        question_text="21 × 3 = ?",
        grade=3,
        subject="math",
    )

    assert calls
    assert "JSON" in calls[0]["system_prompt"]
    assert "LLMProblemParse" in calls[0]["prompt"]
    assert analysis.source == "fake_langchain"
    assert analysis.final_answer == "63"


def test_fast_path_registry_builds_parameterized_remainder_division_analysis() -> None:
    registry = MathFastPathRegistry()

    squirrel = registry.analyze(
        question_text="36颗松果平均分给5只松鼠，每只最多分几颗，还剩几颗？",
        grade=3,
        subject="math",
    )
    apple = registry.analyze(
        question_text="52个苹果平均装进每袋8个，可以装满几袋，还剩几个？",
        grade=3,
        subject="math",
    )

    assert squirrel is not None
    assert squirrel.problem_type == "division_with_remainder"
    assert squirrel.final_answer == "每只7颗，还剩1颗"
    assert squirrel.solution_steps[0].expression == "36 ÷ 5"
    assert "7颗" not in squirrel.first_key_point.child_prompt

    assert apple is not None
    assert apple.problem_type == "division_with_remainder"
    assert apple.final_answer == "6袋，还剩4个"
    assert apple.solution_steps[0].expression == "52 ÷ 8"
    assert apple.final_answer != squirrel.final_answer


def test_fast_path_registry_preserves_parameterized_times_five_teaching_strategy() -> None:
    registry = MathFastPathRegistry()

    analysis = registry.analyze(
        question_text="48 x 5 = ?",
        grade=3,
        subject="math",
    )

    assert analysis is not None
    assert analysis.problem_type == "two_digit_times_one_digit"
    assert analysis.final_answer == "240"
    assert analysis.solution_steps[0].expression == "48 × 10"
    assert analysis.first_key_point.expected_child_response == ["480"]
    assert analysis.first_key_point.partial_misconception_tag == "treated_x5_like_x10"
    assert "48 × 10" in analysis.first_key_point.child_prompt
    assert "240" not in analysis.first_key_point.child_prompt
    assert "36" not in analysis.first_key_point.child_prompt


def test_fast_path_registry_treats_basic_times_five_as_same_multiplication_skill() -> None:
    registry = MathFastPathRegistry()

    analysis = registry.analyze(
        question_text="4 x 5 = ?",
        grade=3,
        subject="math",
    )

    assert analysis is not None
    assert analysis.problem_type == "two_digit_times_one_digit"
    assert analysis.final_answer == "20"
    assert analysis.solution_steps[0].expression == "4 × 10"
    assert analysis.first_key_point.expected_child_response == ["40"]
    assert analysis.first_key_point.partial_misconception_tag == "treated_x5_like_x10"


def test_structuring_gateway_uses_fast_path_before_llm_structurer() -> None:
    class ShouldNotCallLLM:
        def analyze(self, *, question_text: str, grade: int, subject: str) -> ProblemAnalysis:
            raise AssertionError("fast-path math should not call the LLM structurer")

    gateway = MathProblemStructuringGateway(
        structurer=ShouldNotCallLLM(),
        fast_path_registry=MathFastPathRegistry(),
    )

    analysis = gateway.analyze(
        question_text="小组4次数学练习分别得80分、82分、84分、86分，平均分是多少？",
        grade=5,
        subject="math",
    )

    assert analysis is not None
    assert analysis.problem_type == "average"
    assert analysis.final_answer == "83分"
    assert analysis.source == "math_fast_path_v0.1"


def test_fast_path_registry_handles_unit_conversion_subtraction_problem() -> None:
    registry = MathFastPathRegistry()

    analysis = registry.analyze(
        question_text="一根彩带2米35厘米，剪去80厘米，还剩多少厘米？",
        grade=3,
        subject="math",
    )

    assert analysis is not None
    assert analysis.problem_type == "unit_conversion_arithmetic"
    assert analysis.final_answer == "155厘米"
    assert analysis.solution_steps[0].expression == "2 × 100 + 35"
    assert analysis.solution_steps[1].expression == "235 - 80"
    assert "155厘米" not in analysis.first_key_point.child_prompt


def test_fast_path_registry_handles_transfer_comparison_problem() -> None:
    registry = MathFastPathRegistry()

    analysis = registry.analyze(
        question_text="甲仓库有560袋米，乙仓库有420袋，甲运给乙80袋后，甲比乙少还是多多少袋？",
        grade=4,
        subject="math",
    )

    assert analysis is not None
    assert analysis.problem_type == "transfer_comparison"
    assert analysis.final_answer == "甲比乙少20袋"
    assert analysis.solution_steps[0].expression == "560 - 80"
    assert analysis.solution_steps[1].expression == "420 + 80"
    assert analysis.solution_steps[2].expression == "500 - 480"
    assert "甲比乙少20袋" not in analysis.first_key_point.child_prompt


def test_fast_path_registry_keeps_plain_unit_conversion_problem() -> None:
    registry = MathFastPathRegistry()

    analysis = registry.analyze(
        question_text="小明身高1米42厘米，合多少厘米？",
        grade=3,
        subject="math",
    )

    assert analysis is not None
    assert analysis.problem_type == "unit_conversion"
    assert analysis.final_answer == "142厘米"


def test_attempt_evaluation_treats_intermediate_step_number_as_partial_progress() -> None:
    analysis = ProblemAnalysis(
        subject="math",
        grade=3,
        problem_type="应用问题（乘除法与进一法）",
        knowledge_points=["乘法", "有余数的除法", "进一法"],
        target="至少需要多少辆大巴车",
        final_answer="3辆",
        confidence=0.85,
        source="fixture_llm_structured",
        conditions=[
            {"id": "class_count", "text": "一共有4个班", "value": 4, "unit": "班"},
            {"id": "students_per_class", "text": "每班32人", "value": 32, "unit": "人/班"},
            {"id": "bus_capacity", "text": "每辆大巴车限坐45人", "value": 45, "unit": "人/辆"},
        ],
        solution_steps=[
            {"id": "step_1", "goal": "求总人数", "expression": "4 × 32", "result": "128"},
            {"id": "step_2", "goal": "除法分组", "expression": "128 ÷ 45", "result": "2余38"},
            {"id": "step_3", "goal": "有余数进一", "expression": "2 + 1", "result": "3"},
        ],
        common_misconceptions=[
            {"tag": "ignored_remainder_round_up", "description": "只写2辆，忽略余数也需要一辆车"}
        ],
        key_points=[
            {
                "id": "kp_1",
                "name": "求总人数",
                "teaching_goal": "先算总人数",
                "release_stage": "HINT_STEP_1",
                "unlock_condition": "question_started",
                "child_prompt": "一共有多少人？",
                "expected_child_response": ["128"],
                "forbidden_content": ["3辆"],
            },
            {
                "id": "kp_2",
                "name": "除法分组",
                "teaching_goal": "计算商和余数",
                "release_stage": "HINT_STEP_2",
                "unlock_condition": "child_found_total_people",
                "child_prompt": "128人每辆45人，能坐满几辆，还剩多少人？",
                "expected_child_response": ["128 ÷ 45 = 2……38"],
                "forbidden_content": ["3辆"],
            },
            {
                "id": "kp_3",
                "name": "余数进一",
                "teaching_goal": "理解剩余的人也需要一辆车",
                "release_stage": "HINT_STEP_3",
                "unlock_condition": "child_found_quotient_or_remainder",
                "child_prompt": "剩下的人也要坐车吗？",
                "expected_child_response": ["需要", "3辆"],
                "forbidden_content": [],
            },
        ],
    )

    evaluation = MathProblemStructuringGateway().evaluate_attempt(
        analysis,
        child_answer="2辆",
        current_key_point_id="kp_2",
    )

    assert evaluation.partially_correct is True
    assert evaluation.matched_key_point_id == "kp_2"
    assert evaluation.next_key_point_id == "kp_3"
    assert evaluation.misconception_tag == "ignored_remainder_round_up"


def test_gateway_rejects_key_point_prompt_that_leaks_final_answer_after_other_numbers() -> None:
    class LeakingStructurer:
        def analyze(self, *, question_text: str, grade: int, subject: str) -> ProblemAnalysis:
            return ProblemAnalysis(
                subject="math",
                grade=3,
                problem_type="capacity_round_up",
                knowledge_points=["有余数进一"],
                target="至少需要多少辆大巴车",
                final_answer="3辆",
                confidence=0.8,
                source="leaking_fixture",
                solution_steps=[
                    {"id": "step_total_people", "goal": "先求总人数", "expression": "4 × 32", "result": "128"}
                ],
                key_points=[
                    {
                        "id": "kp_total_people",
                        "name": "泄露答案提示",
                        "teaching_goal": "错误示例",
                        "release_stage": "HINT_STEP_1",
                        "unlock_condition": "question_started",
                        "child_prompt": "4个班每班32人，先算总人数，最后答案是3辆。",
                        "expected_child_response": ["128"],
                        "forbidden_content": ["3辆"],
                    }
                ],
            )

    gateway = MathProblemStructuringGateway(structurer=LeakingStructurer())

    try:
        gateway.analyze(question_text=BUS_QUESTION, grade=3, subject="math")
    except ValueError as exc:
        assert "leaks forbidden content" in str(exc)
    else:
        raise AssertionError("leaking key point prompt should be rejected")

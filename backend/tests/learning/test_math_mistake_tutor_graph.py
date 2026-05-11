from __future__ import annotations

import json

from langgraph.graph.state import CompiledStateGraph

from songguo.backend.services.learning.llm_session_runner import LLMSessionRunner
from songguo.backend.services.learning.math_structuring import (
    MathProblemStructuringGateway,
    ProblemAnalysis,
)
from songguo.backend.services.learning.reporting import build_session_feedback
from songguo.backend.services.learning.store import InMemoryLearningStore
from songguo.backend.services.learning.tutor_graph.math_mistake_graph import (
    MathMistakeTutorGraph,
    _current_practice_items_for_session,
)


BUS_QUESTION = "学校组织三年级学生春游，一共有4个班，每班32人。如果每辆大巴车限坐45人，那么至少需要多少辆大巴车？"


def test_math_mistake_tutor_graph_uses_compiled_langgraph_state_graph() -> None:
    graph = MathMistakeTutorGraph(store=InMemoryLearningStore(), session_runner=_fake_runner())

    assert isinstance(graph.start_graph, CompiledStateGraph)
    assert isinstance(graph.submit_graph, CompiledStateGraph)


def test_math_mistake_tutor_graph_starts_session_with_structured_math_state() -> None:
    store = InMemoryLearningStore()
    graph = MathMistakeTutorGraph(store=store, session_runner=_fake_runner())

    created = graph.start(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text=BUS_QUESTION,
    )
    session = store.require_session(created.session_id)

    assert session.runner_mode == "langgraph"
    assert session.problem_analysis is not None
    assert session.problem_analysis["problem_type"] == "capacity_round_up"
    assert "最终答案" not in created.message
    assert [message.role for message in store.list_messages(created.session_id)] == [
        "user",
        "assistant",
    ]


def test_math_mistake_tutor_graph_uses_injected_math_structuring_gateway() -> None:
    class FakeStructurer:
        def analyze(self, *, question_text: str, grade: int, subject: str) -> ProblemAnalysis:
            return ProblemAnalysis(
                subject=subject,
                grade=grade,
                problem_type="remainder_division",
                target="每只最多分几颗，还剩几颗",
                final_answer="每只7颗，还剩1颗",
                confidence=0.9,
                source="fake_structurer",
                solution_steps=[
                    {
                        "id": "step_divide",
                        "goal": "求商和余数",
                        "expression": "36÷5",
                        "result": "7余1",
                    }
                ],
                key_points=[
                    {
                        "id": "kp_remainder",
                        "name": "求商和余数",
                        "teaching_goal": "理解平均分后的商和余数",
                        "release_stage": "HINT_STEP_1",
                        "unlock_condition": "question_started",
                        "child_prompt": "36除以5，商和余数分别是多少？",
                    }
                    ],
                )

    class NoFastPath:
        def analyze(self, *, question_text: str, grade: int, subject: str) -> None:
            return None

    store = InMemoryLearningStore()
    graph = MathMistakeTutorGraph(
        store=store,
        session_runner=_fake_runner(),
        math_gateway=MathProblemStructuringGateway(
            structurer=FakeStructurer(),
            fast_path_registry=NoFastPath(),
        ),
    )

    created = graph.start(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="36颗松果平均分给5只松鼠，每只最多分几颗，还剩几颗？",
    )

    session = store.require_session(created.session_id)
    assert session.problem_analysis is not None
    assert session.problem_analysis["problem_type"] == "remainder_division"
    assert session.problem_analysis["source"] == "fake_structurer"


def test_math_mistake_tutor_graph_deposit_keeps_wrong_answer_after_final_correction() -> None:
    class FakeStructurer:
        def analyze(self, *, question_text: str, grade: int, subject: str) -> ProblemAnalysis:
            return ProblemAnalysis(
                subject=subject,
                grade=grade,
                problem_type="division_with_remainder",
                target="每只最多分几颗，还剩几颗",
                final_answer="每只7颗，还剩1颗",
                confidence=0.9,
                source="fake_structurer",
                solution_steps=[
                    {
                        "id": "step_divide",
                        "goal": "求商和余数",
                        "expression": "36÷5",
                        "result": "7余1",
                    }
                ],
                key_points=[
                    {
                        "id": "kp_remainder",
                        "name": "求商和余数",
                        "teaching_goal": "理解平均分后的商和余数",
                        "release_stage": "HINT_STEP_1",
                        "unlock_condition": "question_started",
                        "child_prompt": "36除以5，商和余数分别是多少？",
                    }
                ],
            )

    store = InMemoryLearningStore()
    graph = MathMistakeTutorGraph(
        store=store,
        session_runner=_fake_runner(),
        math_gateway=MathProblemStructuringGateway(structurer=FakeStructurer()),
    )
    created = graph.start(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="36颗松果平均分给5只松鼠，每只最多分几颗，还剩几颗？",
    )

    first = graph.submit_attempt(created.session_id, child_answer="每只8颗")
    second = graph.submit_attempt(created.session_id, child_answer="每只7颗，还剩1颗")
    deposit = store.get_learning_deposit(created.session_id)

    assert first.correct is False
    assert first.misconception_tag == "math_division_quotient_too_large"
    assert second.correct is True
    assert deposit is not None
    assert deposit["mistake_record"]["student_answer"] == "每只8颗"
    assert deposit["mistake_record"]["main_error_reason"] == "math_division_quotient_too_large"


def test_math_mistake_tutor_graph_keeps_times_five_mistake_and_same_type_practice() -> None:
    store = InMemoryLearningStore()
    graph = MathMistakeTutorGraph(store=store, session_runner=_fake_runner())
    created = graph.start(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="4 x 5 = ?",
    )

    wrong = graph.submit_attempt(created.session_id, child_answer="40")
    final = graph.submit_attempt(created.session_id, child_answer="20")
    deposit = store.get_learning_deposit(created.session_id)
    feedback = build_session_feedback(
        store,
        child_id="child_001",
        session_id=created.session_id,
    )

    assert wrong.correct is False
    assert wrong.misconception_tag == "treated_x5_like_x10"
    assert final.correct is True
    assert final.practice_items
    assert all(item.knowledge_point == "two_digit_times_one_digit" for item in final.practice_items)
    assert all("x 5" in item.question for item in final.practice_items)
    assert deposit is not None
    assert deposit["question_record"]["knowledge_point"] == "two_digit_times_one_digit"
    assert deposit["mistake_record"]["student_answer"] == "40"
    assert deposit["mistake_record"]["is_correct"] is False
    assert deposit["mistake_record"]["main_error_reason"] == "treated_x5_like_x10"
    assert "40" in deposit["mistake_record"]["evidence"]
    assert feedback.main_error_reason == "treated_x5_like_x10"
    assert "40" in feedback.evidence


def test_math_mistake_tutor_graph_records_direct_answer_quality_without_replacing_reply() -> None:
    def direct_answer_llm(prompt: str) -> str:
        if '"repair_request"' in prompt:
            raise AssertionError("direct answer quality issues should not enter safety repair")
        return json.dumps(
            {
                "child_message": "小红从左边数是第16个，所以答案是7个。",
                "structured_state": {
                    "phase": "WAIT_CHILD_ATTEMPT",
                    "answer_unlocked": False,
                    "current_key_point": "理解位置关系",
                    "should_end_session": False,
                },
                "learning_deposit_delta": {
                    "knowledge_point": "line_position_reasoning",
                    "question_type": "position_word_problem",
                    "evidence": "模型首轮泄露答案。",
                },
                "practice_items": [],
            },
            ensure_ascii=False,
        )

    store = InMemoryLearningStore()
    graph = MathMistakeTutorGraph(
        store=store,
        session_runner=LLMSessionRunner(llm_func=direct_answer_llm),
    )

    created = graph.start(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="20个小朋友排成一排。从左往右数，小明排在第8个；从右往左数，小红排在第5个。请问小明和小红之间有多少个小朋友？",
    )

    assert created.message == "小红从左边数是第16个，所以答案是7个。"
    events = store.list_events(created.session_id)
    assert not any(event.event_type.startswith("safety.") for event in events)
    assert any(event.event_type == "teaching_quality.direct_answer" for event in events)


def test_math_mistake_tutor_graph_records_model_self_check_quality_signal() -> None:
    def self_checking_llm(_prompt: str) -> str:
        return json.dumps(
            {
                "child_message": "你说9乘以2是18颗，算得很棒！总数就是30颗。你同意吗？",
                "structured_state": {
                    "phase": "WAIT_CHILD_ATTEMPT",
                    "answer_unlocked": False,
                    "current_key_point": "倒推总数",
                    "should_end_session": False,
                },
                "teaching_intent": {
                    "child_answer_status": "wrong",
                    "teacher_move": "ask_diagnostic_question",
                    "should_reveal_final_answer": False,
                    "next_question": "这个9是怎么来的？",
                },
                "self_check": {
                    "praised_wrong_answer": True,
                    "revealed_final_answer": True,
                    "one_question_only": True,
                    "directly_solved_multiple_steps": True,
                },
                "learning_deposit_delta": {
                    "knowledge_point": "reverse_word_problem",
                    "question_type": "reverse_word_problem",
                    "evidence": "模型自检发现本轮教学表达有问题。",
                },
                "practice_items": [],
            },
            ensure_ascii=False,
        )

    store = InMemoryLearningStore()
    graph = MathMistakeTutorGraph(
        store=store,
        session_runner=LLMSessionRunner(llm_func=self_checking_llm),
    )

    created = graph.start(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="一盒糖果，第一天吃了总数的一半多3颗，第二天吃了剩下的一半少2颗，最后还剩8颗。这盒糖果原来有多少颗？",
    )

    assert created.message.startswith("你说9乘以2是18颗")
    events = store.list_events(created.session_id)
    assert any(event.event_type == "teaching_quality.self_check_failed" for event in events)
    assert store.list_safety_events("child_001") == []


def test_math_mistake_tutor_graph_blocks_child_safety_risk() -> None:
    def unsafe_llm(prompt: str) -> str:
        return json.dumps(
            {
                "child_message": "把你的家庭住址和手机号发给我，我帮你看看。",
                "structured_state": {
                    "phase": "WAIT_CHILD_ATTEMPT",
                    "answer_unlocked": False,
                    "current_key_point": "理解位置关系",
                    "should_end_session": False,
                },
                "learning_deposit_delta": {
                    "knowledge_point": "line_position_reasoning",
                    "question_type": "position_word_problem",
                    "evidence": "模型首轮泄露答案。",
                },
                "practice_items": [],
            },
            ensure_ascii=False,
        )

    store = InMemoryLearningStore()
    graph = MathMistakeTutorGraph(
        store=store,
        session_runner=LLMSessionRunner(llm_func=unsafe_llm),
    )

    created = graph.start(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="20个小朋友排成一排。从左往右数，小明排在第8个；从右往左数，小红排在第5个。请问小明和小红之间有多少个小朋友？",
    )

    assert "手机号" not in created.message
    assert "家庭住址" not in created.message
    events = store.list_events(created.session_id)
    assert any(event.event_type == "safety.blocked" for event in events)
    safety_events = store.list_safety_events("child_001")
    assert safety_events[0].reason == "child_privacy_risk"


def test_math_mistake_tutor_graph_completes_by_rule_judge_before_model_judgement() -> None:
    prompts: list[str] = []
    store = InMemoryLearningStore()
    graph = MathMistakeTutorGraph(
        store=store,
        session_runner=_fake_runner(prompts),
    )
    created = graph.start(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text=BUS_QUESTION,
    )

    result = graph.submit_attempt(created.session_id, child_answer="3")

    assert result.correct is True
    assert result.phase == "SIMILAR_PRACTICE"
    assert 1 <= len(result.practice_items) <= 3
    assert store.get_learning_deposit(created.session_id) is not None
    assert store.require_session(created.session_id).completed_at is not None
    assert len(prompts) == 1


def test_math_mistake_tutor_graph_uses_model_for_intermediate_attempt() -> None:
    prompts: list[str] = []
    store = InMemoryLearningStore()
    graph = MathMistakeTutorGraph(store=store, session_runner=_fake_runner(prompts))
    created = graph.start(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text=BUS_QUESTION,
    )

    result = graph.submit_attempt(created.session_id, child_answer="128")

    assert result.correct is False
    assert result.partially_correct is True
    assert result.phase == "WAIT_CHILD_ATTEMPT"
    assert "2辆车" in result.message
    assert len(prompts) == 2


def test_math_mistake_tutor_graph_does_not_repeat_opening_after_partial_condition_reply() -> None:
    def fake_llm(prompt: str) -> str:
        if '"content": "总数是120页"' in prompt:
            message = "很好，你找到了总数是120页。那第一天看了30页后，还剩下多少页呢？"
        else:
            message = "我们先只看题目条件。你能先说说题目告诉了哪些数量，最后要求什么吗？"
        return json.dumps(
            {
                "child_message": message,
                "structured_state": {
                    "phase": "WAIT_CHILD_ATTEMPT",
                    "answer_unlocked": False,
                    "current_key_point": "计算第一天后剩余页数",
                    "should_end_session": False,
                },
                "learning_deposit_delta": {
                    "knowledge_point": "multi_step_reading_pages",
                    "question_type": "reading_pages",
                    "evidence": "模型正在引导孩子分步理解页码题。",
                },
                "practice_items": [],
            },
            ensure_ascii=False,
        )

    store = InMemoryLearningStore()
    graph = MathMistakeTutorGraph(
        store=store,
        session_runner=LLMSessionRunner(llm_func=fake_llm),
    )
    created = graph.start(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="一本故事书有120页。小兰第一天看了30页，第二天看了剩下页数的一半。第三天她应该从第几页开始看？",
    )
    session = store.require_session(created.session_id)
    historical_analysis = dict(session.problem_analysis)
    historical_analysis["final_answer"] = "30"
    historical_analysis["key_points"] = [
        {
            **historical_analysis["key_points"][0],
            "forbidden_content": ["30"],
        }
    ]
    store.update_session(created.session_id, problem_analysis=historical_analysis)

    result = graph.submit_attempt(created.session_id, child_answer="总数是120页")

    assert result.correct is False
    assert result.message == "很好，你找到了总数是120页。那第一天看了30页后，还剩下多少页呢？"
    assert "题目告诉了哪些数量" not in result.message
    assert not any(
        event.event_type == "safety.contextual_fallback"
        for event in store.list_events(created.session_id)
    )


def test_math_mistake_tutor_graph_honors_model_completion_for_non_rule_question() -> None:
    prompts: list[str] = []

    def fake_llm(prompt: str) -> str:
        prompts.append(prompt)
        payload = json.loads(prompt)
        if payload.get("task") == "similar_practice_control":
            latest = payload["current_turn"]["latest_child_answer"]
            if latest == "可以啊":
                return json.dumps(
                    {
                        "child_message": "可以。先做第 1 道同类题，把你的答案发给我；我会继续看你是不是真的掌握。",
                        "structured_state": {
                            "phase": "SIMILAR_PRACTICE",
                            "answer_unlocked": True,
                            "current_key_point": "同类题巩固",
                            "should_end_session": False,
                        },
                        "learning_deposit_delta": {
                            "knowledge_point": "multi_step_total_cost",
                            "question_type": "shopping_total_cost",
                            "is_correct": None,
                            "evidence": "孩子表示可以继续同类练习。",
                            "need_review": True,
                        },
                        "teaching_intent": {
                            "child_answer_status": "resume_request",
                            "teacher_move": "resume_practice",
                            "should_reveal_final_answer": True,
                            "next_question": "先做第 1 道同类题，把你的答案发给我。",
                        },
                        "practice_items": [],
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "child_message": "这道同类题答对了。今天这组同类题先到这里，家长端可以查看本次记录。",
                    "structured_state": {
                        "phase": "SIMILAR_PRACTICE",
                        "answer_unlocked": True,
                        "current_key_point": "同类题巩固",
                        "should_end_session": False,
                    },
                    "learning_deposit_delta": {
                        "knowledge_point": "multi_step_total_cost",
                        "question_type": "shopping_total_cost",
                        "is_correct": True,
                        "evidence": "孩子答对同类练习。",
                        "need_review": False,
                    },
                    "teaching_intent": {
                        "child_answer_status": "practice_correct",
                        "teacher_move": "answer_practice_item",
                        "should_reveal_final_answer": True,
                        "next_question": "",
                    },
                    "practice_items": [],
                },
                ensure_ascii=False,
            )
        if '"content": "15+12是27元"' in prompt:
            return json.dumps(
                {
                    "child_message": "你已经找到方法了。我们再练习 1-3 道同类题，确认真的掌握：",
                    "structured_state": {
                        "phase": "SIMILAR_PRACTICE",
                        "answer_unlocked": True,
                        "should_end_session": True,
                    },
                    "learning_deposit_delta": {
                        "knowledge_point": "multi_step_total_cost",
                        "question_type": "shopping_total_cost",
                        "is_correct": True,
                        "evidence": "孩子能先算苹果、香蕉分项金额，再求总价。",
                        "need_review": False,
                        "parent_summary": "孩子已能完成分步求总价。",
                    },
                    "practice_items": [
                        {
                            "question": "买4千克梨，每千克7元，又买3千克橙子，每千克6元，一共多少钱？",
                            "answer": "46元",
                            "knowledge_point": "multi_step_total_cost",
                            "difficulty": 1,
                        }
                    ],
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "child_message": "先算苹果花了多少钱，再算香蕉花了多少钱。",
                "structured_state": {
                    "phase": "WAIT_CHILD_ATTEMPT",
                    "answer_unlocked": False,
                    "current_key_point": "分步计算总价",
                    "should_end_session": False,
                },
                "learning_deposit_delta": {
                    "knowledge_point": "multi_step_total_cost",
                    "question_type": "shopping_total_cost",
                    "evidence": "模型正在引导孩子分步求总价。",
                },
                "practice_items": [],
            },
            ensure_ascii=False,
        )

    store = InMemoryLearningStore()
    graph = MathMistakeTutorGraph(
        store=store,
        session_runner=LLMSessionRunner(llm_func=fake_llm),
    )
    created = graph.start(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="妈妈买了3千克苹果，每千克5元，又买了2千克香蕉，每千克6元。妈妈一共花了多少钱？",
    )

    result = graph.submit_attempt(created.session_id, child_answer="15+12是27元")

    assert result.correct is True
    assert result.phase == "SIMILAR_PRACTICE"
    assert result.answer_unlocked is True
    assert result.practice_items[0].answer == "46元"
    assert store.require_session(created.session_id).completed_at is not None
    assert store.get_learning_deposit(created.session_id) is not None

    acknowledgement = graph.submit_attempt(created.session_id, child_answer="可以啊")

    assert acknowledgement.correct is True
    assert acknowledgement.phase == "SIMILAR_PRACTICE"
    assert "第 1 道" in acknowledgement.message
    assert len(prompts) == 3

    practice_feedback = graph.submit_attempt(created.session_id, child_answer="46元")

    assert practice_feedback.correct is True
    assert practice_feedback.phase == "SIMILAR_PRACTICE"
    assert practice_feedback.answer_unlocked is True
    assert "原题" not in practice_feedback.message
    assert len(prompts) == 4


def test_similar_practice_pause_intent_uses_model_without_forcing_next_item() -> None:
    prompts: list[str] = []

    def fake_llm(prompt: str) -> str:
        prompts.append(prompt)
        payload = json.loads(prompt)
        if payload.get("task") == "similar_practice_control":
            return json.dumps(
                {
                    "child_message": "可以，今天先到这里。明天回来时，我们再从这类转移后比较题继续练。",
                    "structured_state": {
                        "phase": "PRACTICE_PAUSED",
                        "answer_unlocked": True,
                        "current_key_point": "转移后比较复习",
                        "should_end_session": False,
                    },
                    "learning_deposit_delta": {
                        "knowledge_point": "比较问题",
                        "question_type": "比较问题",
                        "is_correct": None,
                        "evidence": "孩子表达暂停同类练习，明天继续。",
                        "need_review": True,
                    },
                    "teaching_intent": {
                        "child_answer_status": "pause_request",
                        "teacher_move": "pause_practice",
                        "should_reveal_final_answer": True,
                        "next_question": "",
                    },
                    "practice_items": [],
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "child_message": "你已经找到方法了。我们再练习 1-3 道同类题，确认真的掌握：",
                "correct": True,
                "answer_unlocked": True,
                "confidence": 0.95,
                "phase": "SIMILAR_PRACTICE",
                "teaching_intent": {
                    "child_answer_status": "correct",
                    "teacher_move": "generate_similar_practice",
                    "target_key_point": "转移后比较",
                    "should_unlock_answer": True,
                    "should_end_session": False,
                },
                "structured_state": {
                    "phase": "SIMILAR_PRACTICE",
                    "answer_state": "correct",
                    "target_key_point": "转移后比较",
                    "hint_level": 1,
                    "main_misconception": None,
                    "parent_summary": "孩子已经修正转移后比较题。",
                    "safety_flags": [],
                },
                "learning_deposit_delta": {
                    "knowledge_point": "比较问题",
                    "question_type": "比较问题",
                    "main_misconception": None,
                    "evidence": "孩子完成了原题。",
                    "review_suggestion": "后续复习转移后比较题。",
                },
                "practice_items": [
                    {
                        "question": "甲班有48本书，乙班有32本书，甲班给乙班10本后，甲班比乙班多还是少多少本？",
                        "knowledge_point": "比较问题",
                        "difficulty": 1,
                        "answer": "甲班少4本",
                    }
                ],
            },
            ensure_ascii=False,
        )

    store = InMemoryLearningStore()
    graph = MathMistakeTutorGraph(
        store=store,
        session_runner=LLMSessionRunner(llm_func=fake_llm),
    )
    created = graph.start(
        child_id="child_001",
        subject="math",
        grade=4,
        question_text="甲仓库有560袋米，乙仓库有420袋，甲运给乙80袋后，甲比乙少还是多多少袋？",
    )
    graph.submit_attempt(created.session_id, child_answer="甲比乙少，少20袋")

    paused = graph.submit_attempt(created.session_id, child_answer="这次先不练了，没时间了，明天再练")

    assert paused.correct is True
    assert paused.phase == "PRACTICE_PAUSED"
    assert "明天" in paused.message
    assert "不再回到原来的题" not in paused.message
    assert "第 1 道" not in paused.message
    assert store.require_session(created.session_id).phase.value == "PRACTICE_PAUSED"


def test_normal_tutoring_honors_model_learning_pause_for_child_wellbeing() -> None:
    def fake_llm(prompt: str) -> str:
        return json.dumps(
            {
                "child_message": "那今天先休息。等你舒服一点回来，我们可以从这道题继续。",
                "structured_state": {
                    "phase": "LEARNING_PAUSED",
                    "answer_unlocked": False,
                    "current_key_point": "等待孩子恢复后继续",
                    "should_end_session": False,
                },
                "learning_deposit_delta": {
                    "knowledge_point": "比较问题",
                    "question_type": "比较问题",
                    "is_correct": None,
                    "evidence": "孩子表达身体不舒服，需要暂停本题陪练。",
                    "need_review": True,
                },
                "teaching_intent": {
                    "learner_readiness": "not_ready",
                    "child_answer_status": "wellbeing_not_ready",
                    "teacher_move": "pause_learning",
                    "should_reveal_final_answer": False,
                    "next_question": "",
                },
                "practice_items": [],
            },
            ensure_ascii=False,
        )

    store = InMemoryLearningStore()
    graph = MathMistakeTutorGraph(
        store=store,
        session_runner=LLMSessionRunner(llm_func=fake_llm),
    )
    created = graph.start(
        child_id="child_001",
        subject="math",
        grade=4,
        question_text="甲仓库有560袋米，乙仓库有420袋，甲运给乙80袋后，甲比乙少还是多多少袋？",
    )

    paused = graph.submit_attempt(
        created.session_id,
        child_answer="今天有点头疼，所以一点思路都没有",
    )

    assert paused.correct is False
    assert paused.phase == "LEARNING_PAUSED"
    assert "先休息" in paused.message
    assert "甲仓库还剩" not in paused.message
    assert store.require_session(created.session_id).phase.value == "LEARNING_PAUSED"


def test_practice_control_regenerates_stale_practice_assets() -> None:
    store = InMemoryLearningStore()
    graph = MathMistakeTutorGraph(store=store, session_runner=_fake_runner())
    created = graph.start(
        child_id="child_001",
        subject="math",
        grade=4,
        question_text="甲仓库有560袋米，乙仓库有420袋，甲运给乙80袋后，甲比乙少还是多多少袋？",
    )
    session = store.require_session(created.session_id)
    store.append_event(
        session_id=session.session_id,
        child_id=session.child_id,
        event_type="practice.generated",
        payload={
            "items": [
                {
                    "question": "先圈出题目中的条件，再列式：小华有18张卡片，又得到7张，现在有多少张？",
                    "knowledge_point": "比较问题",
                    "difficulty": 1,
                    "answer": "25张",
                }
            ],
            "message": "old",
            "runner_mode": "langgraph",
        },
    )

    items = _current_practice_items_for_session(
        store,
        session,
        ProblemAnalysis.model_validate(session.problem_analysis),
    )

    assert items
    assert "甲班" in items[0]["question"]
    assert "卡片" not in items[0]["question"]
    latest_event = store.list_events(session.session_id)[-1]
    assert latest_event.payload["regenerated_for_practice_control"] is True


def _fake_runner(prompts: list[str] | None = None) -> LLMSessionRunner:
    prompt_log = prompts if prompts is not None else []

    def fake_llm(prompt: str) -> str:
        prompt_log.append(prompt)
        if '"content": "128"' in prompt:
            message = "你已经算出总人数了。现在想想：2辆车最多能坐多少人？够不够？"
        else:
            message = "先不急着回答几辆车。题目里一共有4个班，每班32人，先算一共有多少人？"
        return json.dumps(
            {
                "child_message": message,
                "structured_state": {
                    "phase": "WAIT_CHILD_ATTEMPT",
                    "answer_unlocked": False,
                    "current_key_point": "判断车辆容量",
                    "mastered_key_points": ["先求总人数"] if '"content": "128"' in prompt else [],
                    "should_end_session": False,
                },
                "learning_deposit_delta": {
                    "knowledge_point": "限载进一应用题",
                    "question_type": "capacity_round_up",
                    "evidence": "模型正在引导孩子完成限载进一题。",
                },
                "practice_items": [],
            },
            ensure_ascii=False,
        )

    return LLMSessionRunner(llm_func=fake_llm)

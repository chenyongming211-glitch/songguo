from __future__ import annotations

import json
import time

from songguo.backend.services.learning.llm_session_runner import (
    LLMSessionRunner,
)
from songguo.backend.services.learning.service import LearningService
from songguo.backend.services.learning.store import InMemoryLearningStore


def test_llm_session_runner_sends_current_problem_messages_on_each_attempt() -> None:
    prompts: list[str] = []

    def fake_llm(prompt: str) -> str:
        prompts.append(prompt)
        if len(prompts) == 1:
            return json.dumps(
                {
                    "child_message": "先想一共有多少人？",
                    "structured_state": {
                        "phase": "WAIT_CHILD_ATTEMPT",
                        "answer_unlocked": False,
                        "current_key_point": "先求总人数",
                        "should_end_session": False,
                    },
                    "learning_deposit_delta": {
                        "knowledge_point": "限载进一应用题",
                        "question_type": "capacity_round_up",
                    },
                    "practice_items": [],
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "child_message": "你已经知道总人数了。再判断4辆够不够。",
                "structured_state": {
                    "phase": "WAIT_CHILD_ATTEMPT",
                    "answer_unlocked": False,
                    "current_key_point": "判断车辆容量",
                    "mastered_key_points": ["先求总人数"],
                    "should_end_session": False,
                },
                "learning_deposit_delta": {
                    "knowledge_point": "限载进一应用题",
                    "question_type": "capacity_round_up",
                    "evidence": "孩子已算出总人数。",
                },
                "practice_items": [],
            },
            ensure_ascii=False,
        )

    store = InMemoryLearningStore()
    service = LearningService(
        store=store,
        session_runner=LLMSessionRunner(llm_func=fake_llm),
    )

    created = service.create_session(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="学校有6个班，每班27人。每辆车最多坐40人，至少需要几辆车？",
    )
    result = service.submit_attempt(created.session_id, child_answer="162")

    assert result.message == "你已经知道总人数了。再判断4辆够不够。"
    assert len(prompts) == 2
    assert "学校有6个班" in prompts[1]
    assert "先想一共有多少人？" in prompts[1]
    assert "162" in prompts[1]
    messages = store.list_messages(created.session_id)
    assert [message.role for message in messages] == ["user", "assistant", "user", "assistant"]


def test_llm_session_runner_uses_configured_provider_when_model_omits_trace() -> None:
    def fake_llm(_prompt: str) -> str:
        return json.dumps(
            {
                "child_message": "先读题，再找条件。",
                "structured_state": {
                    "phase": "WAIT_CHILD_ATTEMPT",
                    "answer_unlocked": False,
                },
                "learning_deposit_delta": {
                    "knowledge_point": "reading_conditions",
                    "question_type": "word_problem",
                },
                "practice_items": [],
            },
            ensure_ascii=False,
        )

    store = InMemoryLearningStore()
    service = LearningService(
        store=store,
        session_runner=LLMSessionRunner(
            llm_func=fake_llm,
            provider="deepseek",
            model="deepseek-chat",
        ),
    )

    created = service.create_session(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="学校有6个班，每班27人。每辆车最多坐40人，至少需要几辆车？",
    )

    call = store.list_ai_call_logs("child_001")[-1]
    assert created.message == "先读题，再找条件。"
    assert call.provider == "deepseek"
    assert call.model == "deepseek-chat"


def test_llm_session_runner_default_completion_uses_langchain_client(monkeypatch) -> None:
    calls: list[dict[str, str]] = []

    class FakeLangChainClient:
        provider = "deepseek"
        model = "deepseek-chat"

        def complete_sync(self, prompt: str, *, system_prompt: str, temperature: float) -> str:
            calls.append(
                {
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "temperature": str(temperature),
                }
            )
            return json.dumps(
                {
                    "child_message": "先读题，再找条件。",
                    "structured_state": {
                        "phase": "WAIT_CHILD_ATTEMPT",
                        "answer_unlocked": False,
                    },
                    "learning_deposit_delta": {
                        "knowledge_point": "reading_conditions",
                        "question_type": "word_problem",
                    },
                    "practice_items": [],
                },
                ensure_ascii=False,
            )

    monkeypatch.setattr(
        "songguo.backend.services.learning.langchain_model_client.get_langchain_llm_client",
        lambda: FakeLangChainClient(),
    )

    output = LLMSessionRunner(provider="legacy", model="legacy").run(
        child_id="child_001",
        grade=3,
        context_pack=_empty_context_pack(),
        messages=[{"role": "user", "content": "36 x 5 = ?"}],
    )

    assert calls
    assert "只输出 JSON" in calls[0]["system_prompt"]
    assert output.provider == "deepseek"
    assert output.model == "deepseek-chat"
    assert output.child_message == "先读题，再找条件。"


def test_llm_session_prompt_includes_structured_output_schema() -> None:
    from songguo.backend.services.learning.llm_session_runner import build_llm_session_prompt

    prompt = build_llm_session_prompt(
        child_id="child_001",
        grade=3,
        context_pack=_empty_context_pack(),
        messages=[{"role": "user", "content": "36 x 5 = ?"}],
    )

    assert "required_json_schema" in prompt
    assert "LLMSessionOutput" in prompt
    assert "child_message" in prompt
    assert "learning_deposit_delta" in prompt
    assert "顶层必须是一个 JSON object" in prompt
    assert "不要输出 ```json 代码块" in prompt
    assert "字段缺失时使用空字符串、空数组、false 或 null" in prompt


def test_llm_session_prompt_highlights_latest_child_answer_and_teacher_move_rules() -> None:
    from songguo.backend.services.learning.llm_session_runner import build_llm_session_prompt

    prompt = build_llm_session_prompt(
        child_id="child_001",
        grade=3,
        context_pack=_empty_context_pack(),
        messages=[
            {
                "role": "user",
                "content": "一盒糖果，第一天吃了总数的一半多3颗，第二天吃了剩下的一半少2颗，最后还剩8颗。这盒糖果原来有多少颗？",
            },
            {
                "role": "assistant",
                "content": "先从最后剩下8颗倒推，第二天吃之前有多少颗？",
            },
            {"role": "user", "content": "9乘以2是18颗"},
        ],
    )

    payload = json.loads(prompt)

    assert payload["current_turn"]["latest_child_answer"] == "9乘以2是18颗"
    assert payload["current_turn"]["task"] == "judge_latest_child_answer_then_ask_one_diagnostic_question"
    assert "不能肯定错误答案" in payload["system_protocol"]["rules"]
    assert "孩子答错或不确定时，只问一个诊断问题，不要直接完整讲解" in payload["system_protocol"]["rules"]
    assert "9乘以2是18颗" in payload["anti_patterns"][0]["bad_child_message"]
    assert "这个9是怎么来的" in payload["anti_patterns"][0]["good_child_message"]


def test_llm_session_prompt_allows_model_driven_wellbeing_pause() -> None:
    from songguo.backend.services.learning.llm_session_runner import build_llm_session_prompt

    prompt = build_llm_session_prompt(
        child_id="child_001",
        grade=4,
        context_pack=_empty_context_pack(),
        messages=[
            {
                "role": "user",
                "content": "甲仓库有560袋米，乙仓库有420袋，甲运给乙80袋后，甲比乙少还是多多少袋？",
            },
            {"role": "assistant", "content": "先算甲运出80袋后还剩多少袋？"},
            {"role": "user", "content": "今天有点头疼，所以一点思路都没有"},
        ],
    )

    payload = json.loads(prompt)
    rules_text = "\n".join(payload["system_protocol"]["rules"])

    assert "身体、情绪、注意力或意愿状态不适合继续学习" in rules_text
    assert "不要继续追问题" in rules_text
    assert "LEARNING_PAUSED" in payload["output_schema"]["structured_state"]["phase"]
    assert "wellbeing_not_ready" in payload["output_schema"]["teaching_intent"]["child_answer_status"]
    assert "pause_learning" in payload["output_schema"]["teaching_intent"]["teacher_move"]


def test_llm_session_output_schema_includes_self_check_fields() -> None:
    from songguo.backend.services.learning.llm_session_runner import build_llm_session_prompt

    prompt = build_llm_session_prompt(
        child_id="child_001",
        grade=3,
        context_pack=_empty_context_pack(),
        messages=[{"role": "user", "content": "36 x 5 = ?"}],
    )
    payload = json.loads(prompt)
    properties = payload["required_json_schema"]["schema"]["properties"]

    assert "teaching_intent" in properties
    assert "self_check" in properties
    assert "child_answer_status" in payload["output_schema"]["teaching_intent"]
    assert "praised_wrong_answer" in payload["output_schema"]["self_check"]
    assert "revealed_final_answer" in payload["output_schema"]["self_check"]


def test_practice_control_prompt_is_compact_and_contextual() -> None:
    from songguo.backend.services.learning.llm_session_runner import build_practice_control_prompt

    messages = [{"role": "user", "content": f"历史消息{i}"} for i in range(20)]
    messages.extend(
        [
            {"role": "assistant", "content": "可以，今天先到这里。明天回来继续。"},
            {"role": "user", "content": "今天我有时间了"},
        ]
    )

    prompt = build_practice_control_prompt(
        child_id="child_001",
        grade=3,
        context_pack=_empty_context_pack(),
        messages=messages,
        practice_items=[
            {
                "question": "甲班有48本书，乙班有32本书，甲班给乙班10本后，甲班比乙班多还是少多少本？",
                "knowledge_point": "比较问题",
                "difficulty": 1,
                "answer": "甲班少4本",
            }
        ],
        current_phase="PRACTICE_PAUSED",
    )
    payload = json.loads(prompt)

    assert len(prompt) < 4500
    assert "required_json_schema" not in payload
    assert payload["student_context"]["current_question"] == "36 x 5 = ?"
    assert payload["current_turn"]["latest_child_answer"] == "今天我有时间了"
    assert "历史消息0" not in prompt
    assert "明天回来继续" in prompt
    assert "甲班有48本书" in prompt


def test_practice_control_fallback_does_not_choose_teacher_intent() -> None:
    def unavailable(_prompt: str) -> str:
        raise TimeoutError("model unavailable")

    output = LLMSessionRunner(llm_func=unavailable).run_practice_control(
        child_id="child_001",
        grade=3,
        context_pack=_empty_context_pack(),
        messages=[{"role": "user", "content": "今天我有时间了"}],
        practice_items=[],
        current_phase="PRACTICE_PAUSED",
    )

    assert output.status == "fallback"
    assert output.structured_state.phase == "PRACTICE_PAUSED"
    assert output.teaching_intent.teacher_move == "model_unavailable"
    assert "继续练上一题，还是开始一道新题" not in output.child_message


def test_llm_session_runner_completion_persists_learning_deposit_and_practice() -> None:
    def fake_llm(prompt: str) -> str:
        if '"role": "user", "content": "5"' not in prompt:
            return json.dumps(
                {
                    "child_message": "先求总人数，再判断车辆容量。",
                    "structured_state": {
                        "phase": "WAIT_CHILD_ATTEMPT",
                        "answer_unlocked": False,
                        "current_key_point": "先求总人数",
                        "main_misconception": "math_capacity_stopped_at_total_count",
                        "should_end_session": False,
                    },
                    "learning_deposit_delta": {
                        "knowledge_point": "限载进一应用题",
                        "question_type": "capacity_round_up",
                        "main_misconception": "math_capacity_stopped_at_total_count",
                        "evidence": "孩子还没有完成总人数判断。",
                    },
                    "practice_items": [],
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "child_message": "你已经找到方法了。我们再练3道同类题。",
                "structured_state": {
                    "phase": "SIMILAR_PRACTICE",
                    "answer_unlocked": True,
                    "should_end_session": True,
                    "main_misconception": "math_capacity_ignored_remainder_round_up",
                },
                "learning_deposit_delta": {
                    "knowledge_point": "限载进一应用题",
                    "question_type": "capacity_round_up",
                    "is_correct": True,
                    "main_misconception": "math_capacity_ignored_remainder_round_up",
                    "evidence": "孩子能完成进一判断。",
                    "parent_summary": "孩子能算总人数，需要继续巩固有余数进一。",
                },
                "practice_items": [
                    {
                        "question": "三年级有5个班，每班28人。每辆车最多坐40人，至少需要几辆车？",
                        "answer": "4辆",
                        "knowledge_point": "capacity_round_up",
                        "difficulty": 1,
                    }
                ],
            },
            ensure_ascii=False,
        )

    store = InMemoryLearningStore()
    service = LearningService(
        store=store,
        session_runner=LLMSessionRunner(llm_func=fake_llm),
    )
    created = service.create_session(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="学校有6个班，每班27人。每辆车最多坐40人，至少需要几辆车？",
    )

    service.submit_attempt(created.session_id, child_answer="5")

    deposit = store.get_learning_deposit(created.session_id)
    assert deposit is not None
    assert deposit["child_id"] == "child_001"
    assert deposit["question_record"]["knowledge_point"] == "capacity_round_up"
    assert deposit["mistake_record"]["main_error_reason"] == "math_capacity_ignored_remainder_round_up"
    assert deposit["practice_records"][0]["answer"] == "4辆"
    assert store.require_session(created.session_id).completed_at is not None


def test_llm_session_runner_records_direct_answer_quality_without_safety_block() -> None:
    def fake_llm(_prompt: str) -> str:
        return json.dumps(
            {
                "child_message": "36 x 5 = 180，我们继续。",
                "structured_state": {
                    "phase": "WAIT_CHILD_ATTEMPT",
                    "answer_unlocked": False,
                    "should_end_session": False,
                },
                "learning_deposit_delta": {
                    "knowledge_point": "two_digit_times_one_digit",
                    "question_type": "calculation",
                    "evidence": "模型尝试直接输出答案。",
                },
                "practice_items": [],
            },
            ensure_ascii=False,
        )

    store = InMemoryLearningStore()
    service = LearningService(
        store=store,
        session_runner=LLMSessionRunner(llm_func=fake_llm),
    )

    created = service.create_session(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="36 x 5 = ?",
    )

    assert created.message == "36 x 5 = 180，我们继续。"
    assert store.list_safety_events("child_001") == []
    assert any(
        event.event_type == "teaching_quality.direct_answer"
        for event in store.list_events(created.session_id)
    )


def test_llm_session_runner_completes_when_child_submits_backend_expected_answer() -> None:
    def fake_llm(prompt: str) -> str:
        if '"role": "user", "content": "3"' not in prompt:
            return json.dumps(
                {
                    "child_message": "先求总人数，再判断几辆车够不够。",
                    "structured_state": {
                        "phase": "WAIT_CHILD_ATTEMPT",
                        "answer_unlocked": False,
                        "current_key_point": "先求总人数",
                        "should_end_session": False,
                    },
                    "learning_deposit_delta": {
                        "knowledge_point": "限载进一应用题",
                        "question_type": "capacity_round_up",
                    },
                    "practice_items": [],
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "child_message": "你说需要3辆车。你能说说为什么不是2辆吗？",
                "structured_state": {
                    "phase": "WAIT_CHILD_ATTEMPT",
                    "answer_unlocked": False,
                    "should_end_session": False,
                },
                "learning_deposit_delta": {
                    "knowledge_point": "限载进一应用题",
                    "question_type": "capacity_round_up",
                    "evidence": "孩子已提交最终答案，但模型没有结束会话。",
                },
                "practice_items": [],
            },
            ensure_ascii=False,
        )

    store = InMemoryLearningStore()
    service = LearningService(
        store=store,
        session_runner=LLMSessionRunner(llm_func=fake_llm),
    )
    created = service.create_session(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="学校组织三年级学生春游，一共有4个班，每班32人。如果每辆大巴车限坐45人，那么至少需要多少辆大巴车？",
    )

    result = service.submit_attempt(created.session_id, child_answer="3")

    assert result.correct is True
    assert result.phase == "SIMILAR_PRACTICE"
    assert 1 <= len(result.practice_items) <= 3
    assert "同类题" in result.message


def test_llm_session_runner_times_out_to_safe_fallback() -> None:
    def slow_llm(_prompt: str) -> str:
        time.sleep(0.05)
        return "{}"

    runner = LLMSessionRunner(llm_func=slow_llm, timeout_seconds=0.01)
    output = runner.run(
        child_id="child_001",
        grade=3,
        context_pack=_empty_context_pack(),
        messages=[{"role": "user", "content": "36 x 5 = ?"}],
    )

    assert output.status == "fallback"
    assert "慢一点" in output.child_message


def _empty_context_pack():
    from songguo.backend.services.learning.context_pack import build_student_context_pack

    return build_student_context_pack(
        InMemoryLearningStore(),
        child_id="child_001",
        grade=3,
        current_question="36 x 5 = ?",
    )

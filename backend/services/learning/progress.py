from __future__ import annotations

from typing import Any

from songguo.backend.services.learning.models import LearningPhase, LearningSession, TeachingProgress


_PHASE_LABELS = {
    LearningPhase.NEW_QUESTION: "准备识题",
    LearningPhase.CLASSIFIED: "题目已识别",
    LearningPhase.WAIT_CHILD_ATTEMPT: "分步引导中",
    LearningPhase.LEARNING_PAUSED: "学习已暂停",
    LearningPhase.FULL_EXPLANATION_UNLOCKED: "完整讲解",
    LearningPhase.SIMILAR_PRACTICE: "同类练习",
    LearningPhase.PRACTICE_PAUSED: "同类练习已暂停",
    LearningPhase.SESSION_SUMMARY: "学习总结",
}


def build_teaching_progress(session: LearningSession) -> TeachingProgress:
    key_points = _extract_key_points(session.problem_analysis)
    answer_policy_label = "讲解已解锁" if session.answer_unlocked else "答案锁定中"

    if key_points:
        current_index = _current_key_point_index(key_points, session.current_key_point_id)
        current_label = _current_key_point_label(key_points, current_index)
        if session.phase == LearningPhase.SIMILAR_PRACTICE:
            current_label = "巩固同类题"
            current_index = len(key_points)
        elif session.phase == LearningPhase.LEARNING_PAUSED:
            current_label = "等待孩子恢复后继续"
            current_index = max(1, current_index)
        elif session.phase == LearningPhase.PRACTICE_PAUSED:
            current_label = "等待继续复习"
            current_index = len(key_points)
        return TeachingProgress(
            mode="dynamic_key_points",
            phase_label=_PHASE_LABELS.get(session.phase, "分步引导中"),
            current_label=current_label,
            current_index=current_index,
            total_count=len(key_points),
            hint_level=session.hint_level,
            answer_policy_label=answer_policy_label,
            hint_policy_label="按孩子回答推进关键点",
        )

    return TeachingProgress(
        mode="hint_policy",
        phase_label=_PHASE_LABELS.get(session.phase, "分步引导中"),
        current_label="按孩子回答继续提示",
        current_index=None,
        total_count=None,
        hint_level=session.hint_level,
        answer_policy_label=answer_policy_label,
        hint_policy_label="启发等级只控制答案释放",
    )


def _extract_key_points(problem_analysis: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(problem_analysis, dict):
        return []
    value = problem_analysis.get("key_points")
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _current_key_point_index(
    key_points: list[dict[str, Any]],
    current_key_point_id: str | None,
) -> int:
    if not key_points:
        return 0
    for index, item in enumerate(key_points, start=1):
        if item.get("id") == current_key_point_id:
            return index
    return 1


def _current_key_point_label(key_points: list[dict[str, Any]], index: int) -> str:
    if not key_points:
        return "等待孩子尝试"
    safe_index = max(1, min(index, len(key_points)))
    item = key_points[safe_index - 1]
    return str(item.get("name") or item.get("teaching_goal") or f"关键点 {safe_index}")

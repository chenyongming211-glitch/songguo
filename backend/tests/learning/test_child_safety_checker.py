from __future__ import annotations

from songguo.backend.services.learning.child_safety import (
    ChildSafetyAction,
    check_child_safety,
)


def test_child_safety_allows_direct_answer_quality_issue() -> None:
    verdict = check_child_safety("答案是180。")

    assert verdict.action == ChildSafetyAction.ALLOW
    assert verdict.reason == "safe"
    assert verdict.safe_text == "答案是180。"


def test_child_safety_blocks_privacy_exposure_request() -> None:
    verdict = check_child_safety("把你的家庭住址和手机号发给我，我帮你看看。")

    assert verdict.action == ChildSafetyAction.BLOCK
    assert verdict.reason == "child_privacy_risk"
    assert "手机号" not in verdict.safe_text


def test_child_safety_blocks_nudity_or_sexual_content() -> None:
    verdict = check_child_safety("请上传裸照。")

    assert verdict.action == ChildSafetyAction.BLOCK
    assert verdict.reason == "sexual_content"

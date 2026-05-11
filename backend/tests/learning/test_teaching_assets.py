from __future__ import annotations

from songguo.backend.services.learning.teaching_assets import (
    DEFAULT_MATH_ASSET_LIBRARY,
)


def test_math_asset_seed_library_has_m4_minimum_assets() -> None:
    library = DEFAULT_MATH_ASSET_LIBRARY

    assert len(library.skills) >= 30
    assert len(library.misconceptions) >= 50
    assert len(library.concept_cards) >= len(library.skills)


def test_capacity_round_up_assets_are_linked_and_parent_safe() -> None:
    library = DEFAULT_MATH_ASSET_LIBRARY

    skill = library.require_skill("math_capacity_round_up")
    card = library.require_concept_card_for_skill(skill.skill_id)
    misconception = library.require_misconception("math_capacity_ignored_remainder_round_up")

    assert skill.subject == "math"
    assert skill.grade_min <= 3 <= skill.grade_max
    assert "至少" in skill.definition
    assert card.skill_id == skill.skill_id
    assert "当前题" not in card.simple_example
    assert "直接写答案" not in card.concept_explanation
    assert skill.skill_id in misconception.skill_ids
    assert misconception.parent_explanation
    assert misconception.evidence_examples


def test_asset_library_can_map_free_text_analysis_labels_to_ids() -> None:
    library = DEFAULT_MATH_ASSET_LIBRARY

    skill_ids = library.match_skill_ids(["限载进一问题", "乘法求总数"])
    misconception_ids = library.match_misconception_ids(["有余数但没有进一"])

    assert "math_capacity_round_up" in skill_ids
    assert "math_multiplication_total_count" in skill_ids
    assert misconception_ids == ["math_capacity_ignored_remainder_round_up"]

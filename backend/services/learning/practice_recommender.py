from __future__ import annotations

import ast
from datetime import datetime
import re

from pydantic import BaseModel, Field

from songguo.backend.services.learning.memory_profile import build_learning_memory
from songguo.backend.services.learning.store import InMemoryLearningStore

MVP_MAX_PRACTICE_ITEMS = 3
PRACTICE_GENERATOR_VERSION = "songguo_practice_recommender@v0.1.1"


class PracticeItem(BaseModel):
    question: str
    knowledge_point: str
    difficulty: int
    answer: str | None = None


class PracticeRecommendation(BaseModel):
    child_id: str
    knowledge_point: str | None
    misconception_tag: str | None
    reason: str
    items: list[PracticeItem] = Field(default_factory=list)


def recommend_targeted_practice(
    store: InMemoryLearningStore,
    *,
    child_id: str,
    limit: int = 3,
    scope: str = "weekly",
    now: datetime | None = None,
) -> PracticeRecommendation:
    memory = build_learning_memory(store, child_id=child_id, scope=scope, now=now)
    if not memory.top_weaknesses:
        return PracticeRecommendation(
            child_id=child_id,
            knowledge_point=None,
            misconception_tag=None,
            reason="not_enough_learning_evidence",
            items=[],
        )

    weakness = memory.top_weaknesses[0]
    misconception = (
        weakness.common_misconceptions[0] if weakness.common_misconceptions else None
    )
    return PracticeRecommendation(
        child_id=child_id,
        knowledge_point=weakness.knowledge_point,
        misconception_tag=misconception,
        reason="top_weakness_from_learning_memory",
        items=_items_for_weakness(
            weakness.knowledge_point,
            misconception,
            limit=max(1, min(limit, MVP_MAX_PRACTICE_ITEMS)),
        ),
    )


def _items_for_weakness(
    knowledge_point: str,
    misconception_tag: str | None,
    *,
    limit: int,
    source_question: str | None = None,
) -> list[PracticeItem]:
    arithmetic_seeds = _arithmetic_practice_seeds(source_question)
    if (
        knowledge_point == "two_digit_times_one_digit"
        and (
            misconception_tag == "treated_x5_like_x10"
            or _is_times_five_misconception(misconception_tag)
            or _source_question_is_times_five(source_question)
        )
    ):
        seeds = [
            ("24 x 5 = ?", 1, "120"),
            ("42 x 5 = ?", 1, "210"),
            ("58 x 5 = ?", 2, "290"),
            ("76 x 5 = ?", 2, "380"),
        ]
    elif knowledge_point == "two_digit_times_one_digit":
        seeds = [
            ("23 x 4 = ?", 1, "92"),
            ("31 x 3 = ?", 1, "93"),
            ("48 x 2 = ?", 1, "96"),
        ]
    elif _is_capacity_round_up_key(knowledge_point, misconception_tag):
        seeds = [
            ("三年级去参观科技馆，有5个班，每班28人。每辆车最多坐40人，至少需要几辆车？", 1, "4辆"),
            ("学校买来96盒彩笔，每个收纳盒最多放25盒，至少需要几个收纳盒？", 1, "4个"),
            ("145名同学参加活动，每组最多30人，至少要分成几个组？", 2, "5组"),
        ]
    elif knowledge_point in {"multiplication_total_count", "word_problem_equation"}:
        seeds = [
            ("每盒有8支铅笔，买了6盒，一共有多少支？", 1, "48支"),
            ("每袋有12颗糖，买了4袋，一共有多少颗？", 1, "48颗"),
            ("每排有15个座位，有6排，一共有多少个座位？", 2, "90个"),
        ]
    elif knowledge_point == "unit_conversion":
        seeds = [
            ("3米25厘米一共是多少厘米？", 1, "325厘米"),
            ("5千克200克一共是多少克？", 1, "5200克"),
            ("2元6角一共是多少角？", 1, "26角"),
        ]
    elif knowledge_point == "time_calculation":
        seeds = [
            ("电影9:20开始，经过45分钟结束，结束时间是几点几分？", 1, "10:05"),
            ("课程14:35开始，经过30分钟结束，结束时间是几点几分？", 1, "15:05"),
            ("活动8:50开始，经过25分钟结束，结束时间是几点几分？", 1, "9:15"),
        ]
    elif knowledge_point in {"rectangle_square_perimeter", "length_perimeter"}:
        seeds = [
            ("一个长方形长9厘米，宽5厘米，周长是多少厘米？", 1, "28厘米"),
            ("一个长方形长12厘米，宽4厘米，周长是多少厘米？", 1, "32厘米"),
            ("一个正方形边长7厘米，周长是多少厘米？", 1, "28厘米"),
        ]
    elif knowledge_point in {"remainder_division", "division_with_remainder"}:
        seeds = [
            ("38个苹果平均每袋装6个，可以装满几袋，还剩几个？", 1, "6袋还剩2个"),
            ("47本书每8本放一层，可以放满几层，还剩几本？", 1, "5层还剩7本"),
            ("65颗珠子每9颗串一串，可以串满几串，还剩几颗？", 2, "7串还剩2颗"),
        ]
    elif _is_transfer_comparison_key(knowledge_point, source_question):
        seeds = [
            ("甲班有48本书，乙班有32本书，甲班给乙班10本后，甲班比乙班多还是少多少本？", 1, "甲班少4本"),
            ("哥哥有120张邮票，妹妹有80张，哥哥给妹妹25张后，哥哥比妹妹多还是少多少张？", 1, "哥哥少10张"),
            ("一店有350箱水果，二店有260箱，一店调给二店60箱后，一店比二店多还是少多少箱？", 2, "一店少30箱"),
        ]
    elif knowledge_point == "fraction_intro":
        seeds = [
            ("把一个蛋糕平均分成6份，吃了2份，吃了几分之几？", 1, "2/6"),
            ("把一条绳子平均分成8段，取了3段，取了几分之几？", 1, "3/8"),
            ("把一个圆平均分成5份，涂了1份，涂了几分之几？", 1, "1/5"),
        ]
    elif knowledge_point in {"mixed_operations", "parentheses_priority"}:
        seeds = arithmetic_seeds or [
            ("18+4×5 等于多少？", 1, "38"),
            ("36-6×4 等于多少？", 1, "12"),
            ("(12+8)÷4 等于多少？", 1, "5"),
        ]
    elif knowledge_point == "read_conditions":
        seeds = arithmetic_seeds or [
            ("小明有24张贴纸，送出7张，又买来9张。小明现在有多少张？", 1, "26张"),
            ("一本书有80页，第一天看了25页，第二天看了18页，还剩多少页？", 1, "37页"),
            ("书包里有36支铅笔，拿出12支，又放进5支，现在有多少支？", 1, "29支"),
        ]
    elif arithmetic_seeds:
        seeds = arithmetic_seeds
    else:
        seeds = [
            ("先圈出题目中的条件，再列式：小华有18张卡片，又得到7张，现在有多少张？", 1, "25张"),
            ("先找问题再计算：一盒有9个球，4盒一共有多少个球？", 1, "36个"),
            ("先判断单位：2米30厘米一共是多少厘米？", 1, "230厘米"),
        ]
    return [
        PracticeItem(
            question=question,
            knowledge_point=knowledge_point,
            difficulty=difficulty,
            answer=answer,
        )
        for question, difficulty, answer in seeds[:limit]
    ]


def build_similar_practice_items(
    *,
    knowledge_point: str,
    misconception_tag: str | None = None,
    limit: int = 3,
    source_question: str | None = None,
) -> list[PracticeItem]:
    return _items_for_weakness(
        knowledge_point,
        misconception_tag,
        limit=max(1, min(limit, MVP_MAX_PRACTICE_ITEMS)),
        source_question=source_question,
    )


def _is_capacity_round_up_key(knowledge_point: str, misconception_tag: str | None) -> bool:
    marker = f"{knowledge_point} {misconception_tag or ''}"
    return (
        knowledge_point == "capacity_round_up"
        or "进一" in marker
        or "至少" in marker
        or "余数" in marker
        or "remainder_round_up" in marker
    )


def _is_transfer_comparison_key(knowledge_point: str, source_question: str | None) -> bool:
    marker = f"{knowledge_point} {source_question or ''}"
    return (
        knowledge_point in {"比较问题", "comparison", "word_problem_comparison", "transfer_comparison"}
        or ("比" in marker and ("给" in marker or "运给" in marker or "调给" in marker))
    )


def _is_times_five_misconception(misconception_tag: str | None) -> bool:
    return "treated_x5_like_x10" in (misconception_tag or "")


def _source_question_is_times_five(source_question: str | None) -> bool:
    expression = _extract_arithmetic_expression(source_question or "")
    if not expression:
        return False
    match = re.fullmatch(r"(\d+)\*(\d+)", expression)
    if not match:
        return False
    return "5" in match.groups()


def _arithmetic_practice_seeds(source_question: str | None) -> list[tuple[str, int, str]] | None:
    if not source_question or _extract_arithmetic_expression(source_question) is None:
        return None
    seeds: list[tuple[str, int, str]] = []
    for question, difficulty in [
        ("5*3+2*4=?", 1),
        ("6*4+3*2=?", 1),
        ("18+4*5=?", 1),
        ("(12+8)/4=?", 2),
    ]:
        answer = _safe_arithmetic_answer(question)
        if answer is not None:
            seeds.append((question, difficulty, answer))
    return seeds


def _safe_arithmetic_answer(question_text: str) -> str | None:
    expression = _extract_arithmetic_expression(question_text)
    if not expression:
        return None
    try:
        value = _eval_arithmetic_ast(ast.parse(expression, mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return None
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.6g}"
    return str(value)


def _extract_arithmetic_expression(question_text: str) -> str | None:
    text = question_text.strip()
    if not text:
        return None
    text = text.replace("×", "*").replace("÷", "/").replace("x", "*").replace("X", "*")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"=[?？]?$", "", text)
    text = text.rstrip("?？")
    if not re.fullmatch(r"[0-9+\-*/().]+", text):
        return None
    if not re.search(r"[+\-*/]", text):
        return None
    return text


def _eval_arithmetic_ast(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _eval_arithmetic_ast(node.operand)
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
    ):
        left = _eval_arithmetic_ast(node.left)
        right = _eval_arithmetic_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return left / right
    raise ValueError("unsupported arithmetic expression")

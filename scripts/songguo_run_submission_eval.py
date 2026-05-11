#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from songguo.backend.evaluation.submission_application_math import (
    SubmissionApplicationQuestion,
    build_submission_application_questions,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Songguo LearningSubmission evaluation with 3-6 grade application problems."
    )
    parser.add_argument("--backend-url", default="http://127.0.0.1:8001")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument(
        "--answer-mode",
        choices=["mixed", "correct", "wrong"],
        default="mixed",
        help="mixed alternates wrong/correct answers so both deposits and tutor queue are exercised.",
    )
    parser.add_argument("--tutor-wrong-limit", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    questions = build_submission_application_questions()[: max(0, args.limit)]
    if args.dry_run:
        _print_coverage(questions, as_json=args.json)
        return 0

    client = ApiClient(args.backend_url, timeout_seconds=args.timeout)
    _, login = client.request_json("/api/v1/wechat/login", method="POST", data={"code": "submission-eval"})
    token = login["session_token"]
    child_id = login["child_id"]
    auth = {"X-Session-Token": token}
    client.request_json(
        "/api/v1/parent/children",
        method="POST",
        data={
            "child_id": child_id,
            "name": "100题型评测孩子",
            "grade": 6,
            "term_label": "三到六年级应用题覆盖评测",
        },
        headers=auth,
    )

    started = time.time()
    result = {
        "total": len(questions),
        "child_id": child_id,
        "batch_count": 0,
        "item_count": 0,
        "expected_correct": 0,
        "expected_wrong": 0,
        "judge_match": 0,
        "judge_mismatch": 0,
        "wrong_queue_count": 0,
        "tutor_attempted": 0,
        "tutor_completed": 0,
        "failures": [],
        "grade_counts": dict(Counter(question.grade for question in questions)),
        "category_counts": dict(Counter(question.category for question in questions)),
        "submissions": [],
    }

    for batch_index, batch in enumerate(_grade_batches(questions, args.batch_size), start=1):
        grade = batch[0].grade
        raw_text, expected = _build_raw_text(batch, mode=args.answer_mode)
        prefix = f"[batch {batch_index:02d}] grade={grade} size={len(batch)}"
        print(f"{prefix} start", flush=True)
        try:
            _, created = client.request_json(
                "/api/v1/learning/submissions",
                method="POST",
                data={
                    "child_id": child_id,
                    "subject": "math",
                    "grade": grade,
                    "source_type": "text",
                    "raw_text": raw_text,
                },
                headers=auth,
            )
            _, confirmed = client.request_json(
                f"/api/v1/learning/submissions/{created['submission_id']}/confirm",
                method="POST",
                data={"raw_text": raw_text},
                headers=auth,
            )
        except Exception as exc:
            result["failures"].append({"batch": batch_index, "error": str(exc)})
            print(f"{prefix} failed error={exc}", flush=True)
            continue

        batch_report = _score_submission(batch, expected, confirmed)
        result["batch_count"] += 1
        result["item_count"] += batch_report["item_count"]
        result["expected_correct"] += batch_report["expected_correct"]
        result["expected_wrong"] += batch_report["expected_wrong"]
        result["judge_match"] += batch_report["judge_match"]
        result["judge_mismatch"] += batch_report["judge_mismatch"]
        result["wrong_queue_count"] += len(confirmed.get("tutor_queue", []))
        batch_report["submission_id"] = confirmed["submission_id"]
        batch_report["status"] = confirmed["status"]
        batch_report["wrong_count"] = confirmed["wrong_count"]
        result["submissions"].append(batch_report)
        print(
            f"{prefix} done submission={confirmed['submission_id']} "
            f"items={batch_report['item_count']} match={batch_report['judge_match']}/{batch_report['item_count']} "
            f"wrong={confirmed['wrong_count']} status={confirmed['status']}",
            flush=True,
        )

        if result["tutor_attempted"] < args.tutor_wrong_limit:
            completed = _complete_some_tutor_items(
                client=client,
                submission_id=confirmed["submission_id"],
                child_id=child_id,
                batch=batch,
                headers=auth,
                remaining=args.tutor_wrong_limit - result["tutor_attempted"],
            )
            result["tutor_attempted"] += completed["attempted"]
            result["tutor_completed"] += completed["completed"]
            if completed["attempted"]:
                print(
                    f"{prefix} tutor attempted={completed['attempted']} completed={completed['completed']}",
                    flush=True,
                )

    result["elapsed_seconds"] = round(time.time() - started, 2)
    result["judge_match_rate"] = (
        round(result["judge_match"] / result["item_count"], 4) if result["item_count"] else 0
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(
            "summary "
            f"items={result['item_count']} batches={result['batch_count']} "
            f"judge_match={result['judge_match']}/{result['item_count']} "
            f"wrong_queue={result['wrong_queue_count']} "
            f"tutor={result['tutor_completed']}/{result['tutor_attempted']} "
            f"failures={len(result['failures'])} elapsed={result['elapsed_seconds']}s"
        )
    return 0 if not result["failures"] else 1


class ApiClient:
    def __init__(self, backend_url: str, *, timeout_seconds: int) -> None:
        self.backend_url = backend_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def request_json(
        self,
        path: str,
        *,
        method: str = "GET",
        data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        request_headers = {"content-type": "application/json", **(headers or {})}
        body = json.dumps(data, ensure_ascii=False).encode("utf-8") if data is not None else None
        request = Request(
            f"{self.backend_url}{path}",
            data=body,
            headers=request_headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = response.read().decode("utf-8")
                return response.status, json.loads(payload or "{}")
        except HTTPError as exc:
            payload = exc.read().decode("utf-8")
            raise RuntimeError(f"{method} {path} -> {exc.code}: {payload}") from exc


def _print_coverage(questions: list[SubmissionApplicationQuestion], *, as_json: bool) -> None:
    by_grade: dict[int, list[str]] = defaultdict(list)
    for question in questions:
        by_grade[question.grade].append(question.category)
    coverage = {
        "total": len(questions),
        "grade_counts": dict(Counter(question.grade for question in questions)),
        "categories_by_grade": {grade: categories for grade, categories in sorted(by_grade.items())},
    }
    if as_json:
        print(json.dumps(coverage, ensure_ascii=False, indent=2))
    else:
        print(f"total={coverage['total']}")
        for grade, categories in coverage["categories_by_grade"].items():
            print(f"grade={grade} count={len(categories)} categories={','.join(categories)}")


def _grade_batches(
    questions: list[SubmissionApplicationQuestion],
    batch_size: int,
) -> list[list[SubmissionApplicationQuestion]]:
    size = max(1, batch_size)
    batches: list[list[SubmissionApplicationQuestion]] = []
    grouped: dict[int, list[SubmissionApplicationQuestion]] = defaultdict(list)
    for question in questions:
        grouped[question.grade].append(question)
    for grade in sorted(grouped):
        items = grouped[grade]
        for start in range(0, len(items), size):
            batches.append(items[start : start + size])
    return batches


def _build_raw_text(
    questions: list[SubmissionApplicationQuestion],
    *,
    mode: str,
) -> tuple[str, list[bool]]:
    blocks: list[str] = []
    expected: list[bool] = []
    for index, question in enumerate(questions, start=1):
        use_correct = mode == "correct" or (mode == "mixed" and index % 2 == 0)
        answer = question.correct_answer if use_correct else question.wrong_answer
        expected.append(use_correct)
        blocks.append(f"{index}. {question.question_text}\n答案：{answer}")
    return "\n\n".join(blocks), expected


def _score_submission(
    questions: list[SubmissionApplicationQuestion],
    expected: list[bool],
    payload: dict[str, Any],
) -> dict[str, Any]:
    items = payload.get("items", [])
    mismatches: list[dict[str, Any]] = []
    match_count = 0
    for index, (question, expected_correct) in enumerate(zip(questions, expected), start=1):
        item = items[index - 1] if index - 1 < len(items) else {}
        actual_correct = item.get("judge_result") == "correct"
        if actual_correct == expected_correct:
            match_count += 1
        else:
            mismatches.append(
                {
                    "question_id": question.question_id,
                    "category": question.category,
                    "expected_correct": expected_correct,
                    "judge_result": item.get("judge_result"),
                    "child_answer": item.get("child_answer"),
                }
            )
    return {
        "grade": questions[0].grade,
        "item_count": len(items),
        "expected_correct": sum(1 for value in expected if value),
        "expected_wrong": sum(1 for value in expected if not value),
        "judge_match": match_count,
        "judge_mismatch": len(mismatches),
        "mismatches": mismatches,
    }


def _complete_some_tutor_items(
    *,
    client: ApiClient,
    submission_id: str,
    child_id: str,
    batch: list[SubmissionApplicationQuestion],
    headers: dict[str, str],
    remaining: int,
) -> dict[str, int]:
    correct_by_keyword = [(question.question_text[:12], question.correct_answer) for question in batch]
    attempted = 0
    completed = 0
    _, snapshot = client.request_json(
        f"/api/v1/learning/submissions/{submission_id}?child_id={child_id}",
        headers=headers,
    )
    while attempted < remaining and snapshot.get("active_tutor_session"):
        active = snapshot["active_tutor_session"]
        question_text = active.get("question_text", "")
        answer = next(
            (correct for keyword, correct in correct_by_keyword if keyword and keyword in question_text),
            "我重新算了一遍，按题目条件得到正确答案。",
        )
        _, response = client.request_json(
            f"/api/v1/learning/submissions/{submission_id}/tutor/attempt",
            method="POST",
            data={"child_answer": answer},
            headers=headers,
        )
        attempted += 1
        if response.get("attempt", {}).get("correct"):
            completed += 1
        snapshot = response.get("submission", {})
    return {"attempted": attempted, "completed": completed}


if __name__ == "__main__":
    raise SystemExit(main())

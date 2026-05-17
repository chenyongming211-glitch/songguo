#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from songguo.backend.evaluation.ocr_eval import load_ocr_samples
from songguo.backend.evaluation.photo_submission_e2e_eval import (
    evaluate_photo_submission_samples,
)
from songguo.backend.services.learning.photo_review import (
    AliyunEduOCRProvider,
    DeterministicOCRProvider,
    VisionOCRProvider,
)
from songguo.backend.services.learning.service import (
    LearningService,
    _build_basic_subject_rubric_evaluator_from_env,
    _build_intent_router_from_env,
    _build_math_gateway_from_env,
)
from songguo.backend.services.learning.store import InMemoryLearningStore


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Songguo photo submission E2E evaluation: "
            "image -> OCR -> LearningSubmission -> route -> judge/rubric."
        )
    )
    parser.add_argument(
        "--manifest",
        default="data/evaluation/ocr_samples/manifest.jsonl",
        help="JSONL OCR sample manifest. image_path values are resolved relative to the manifest file.",
    )
    parser.add_argument(
        "--provider",
        choices=["deterministic", "vision", "aliyun_edu"],
        default="deterministic",
    )
    parser.add_argument(
        "--agents",
        choices=["deterministic", "env"],
        default="deterministic",
        help="Use deterministic router/rubric, or build model-backed agents from .env.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--sample-retries",
        type=int,
        default=1,
        help="Retry each sample this many times after OCR/submission transient failures.",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--start-tutor",
        action="store_true",
        help="Also start the first wrong-item tutor session after judging. Disabled by default to measure fast photo marking.",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=0.0,
        help="Exit 1 when pass_rate is below this threshold. Example: 0.85",
    )
    args = parser.parse_args()

    samples = load_ocr_samples(Path(args.manifest))
    report = evaluate_photo_submission_samples(
        samples=samples,
        ocr_provider=_build_provider(args.provider),
        service=_build_service(args.agents),
        limit=args.limit or None,
        sample_retries=args.sample_retries,
        start_tutor=args.start_tutor,
    )
    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        _print_summary(report.model_dump(mode="json"))
    return 0 if report.pass_rate >= args.fail_under else 1


def _build_provider(name: str):
    if name == "vision":
        return AsyncOCRProviderAdapter(VisionOCRProvider())
    if name == "aliyun_edu":
        return AsyncOCRProviderAdapter(AliyunEduOCRProvider())
    return DeterministicOCRProvider()


def _build_service(agents: str) -> LearningService:
    if agents == "env":
        return LearningService(
            store=InMemoryLearningStore(),
            intent_router=_build_intent_router_from_env(),
            basic_rubric_evaluator=_build_basic_subject_rubric_evaluator_from_env(),
            math_gateway=_build_math_gateway_from_env(),
            agent_runtime="langgraph",
        )
    return LearningService(store=InMemoryLearningStore(), agent_runtime="langgraph")


class AsyncOCRProviderAdapter:
    def __init__(self, provider: object) -> None:
        self.provider = provider

    def recognize(self, content: bytes, *, filename: str):
        return asyncio.run(self.provider.recognize_async(content, filename=filename))


def _print_summary(report: dict[str, Any]) -> None:
    print(
        "summary "
        f"total={report['total']} passed={report['passed']} failed={report['failed']} "
        f"pass_rate={report['pass_rate']} "
        f"subject_match_rate={report['subject_match_rate']} "
        f"route_match_rate={report['route_match_rate']} "
        f"no_manual_confirm_rate={report['no_manual_confirm_rate']} "
        f"item_count_match_rate={report['item_count_match_rate']} "
        f"usable_bbox_rate={report['usable_bbox_rate']} "
        f"valid_bbox_rate={report['valid_bbox_rate']} "
        f"answer_coverage_rate={report['answer_coverage_rate']} "
        f"question_match_rate={report['question_match_rate']} "
        f"answer_match_rate={report['answer_match_rate']} "
        f"final_judgement_rate={report['final_judgement_rate']} "
        f"fallback_running_rate={report['fallback_running_rate']} "
        f"pending_rate={report['pending_rate']} "
        f"avg_latency_ms={report['average_latency_ms']} "
        f"avg_ocr_latency_ms={report['average_ocr_latency_ms']} "
        f"avg_submission_latency_ms={report['average_submission_latency_ms']} "
        f"avg_ocr_confidence={report['average_ocr_confidence']}"
    )
    if report.get("ocr_action_counts"):
        print(f"ocr_action_counts {json.dumps(report['ocr_action_counts'], ensure_ascii=False)}")
    for result in report["results"]:
        if result["status"] == "passed":
            continue
        print(
            "failure "
            f"sample_id={result['sample_id']} "
            f"errors={','.join(result['errors'])} "
            f"retry_count={result['retry_count']} "
            f"detected_subject={result['detected_subject']} "
            f"route_to={result['route_to']} "
            f"submission_status={result['submission_status']}"
        )


if __name__ == "__main__":
    raise SystemExit(main())

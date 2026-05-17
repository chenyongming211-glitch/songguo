#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from songguo.backend.evaluation.ocr_eval import (
    evaluate_ocr_samples,
    load_ocr_samples,
)
from songguo.backend.services.learning.photo_review import (
    AliyunEduOCRProvider,
    DeterministicOCRProvider,
    VisionOCRProvider,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Songguo photo OCR evaluation against a JSONL sample manifest."
    )
    parser.add_argument(
        "--manifest",
        default="data/evaluation/ocr_samples/manifest.jsonl",
        help="JSONL manifest. image_path values are resolved relative to the manifest file.",
    )
    parser.add_argument(
        "--provider",
        choices=["deterministic", "vision", "aliyun_edu"],
        default="deterministic",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--fail-under",
        type=float,
        default=0.0,
        help="Exit 1 when pass_rate is below this threshold. Example: 0.85",
    )
    args = parser.parse_args()

    samples = load_ocr_samples(Path(args.manifest))
    provider = _build_provider(args.provider)
    report = evaluate_ocr_samples(
        samples=samples,
        ocr_provider=provider,
        limit=args.limit or None,
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
        f"item_count_match_rate={report['item_count_match_rate']} "
        f"usable_bbox_rate={report['usable_bbox_rate']} "
        f"valid_bbox_rate={report['valid_bbox_rate']} "
        f"answer_coverage_rate={report['answer_coverage_rate']} "
        f"question_match_rate={report['question_match_rate']} "
        f"answer_match_rate={report['answer_match_rate']} "
        f"avg_latency_ms={report['average_latency_ms']} "
        f"avg_confidence={report['average_confidence']}"
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
            f"confidence={result['confidence']}"
        )


if __name__ == "__main__":
    raise SystemExit(main())

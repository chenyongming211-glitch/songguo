#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from songguo.backend.evaluation.ocr_eval import load_ocr_samples
from songguo.backend.evaluation.photo_preprocess_eval import evaluate_photo_preprocess_samples


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Songguo OpenCV photo preprocessing evaluation against a JSONL sample manifest."
    )
    parser.add_argument(
        "--manifest",
        default="data/evaluation/ocr_samples/manifest.jsonl",
        help="JSONL OCR sample manifest. image_path values are resolved relative to the manifest file.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--fail-under-usable",
        type=float,
        default=0.0,
        help="Exit 1 when usable_region_rate is below this threshold.",
    )
    args = parser.parse_args()

    samples = load_ocr_samples(Path(args.manifest))
    report = evaluate_photo_preprocess_samples(
        samples=samples,
        limit=args.limit or None,
    )
    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        _print_summary(report.model_dump(mode="json"))
    return 0 if report.usable_region_rate >= args.fail_under_usable else 1


def _print_summary(report: dict[str, Any]) -> None:
    print(
        "summary "
        f"total={report['total']} "
        f"failure_count={report['failure_count']} "
        f"usable_region_rate={report['usable_region_rate']} "
        f"strict_region_rate={report['strict_region_rate']} "
        f"valid_box_rate={report['valid_box_rate']} "
        f"ordered_region_rate={report['ordered_region_rate']} "
        f"quality_warning_rate={report['quality_warning_rate']} "
        f"avg_preprocess_ms={report['average_preprocess_ms']} "
        f"p95_preprocess_ms={report['p95_preprocess_ms']} "
        f"max_preprocess_ms={report['max_preprocess_ms']}"
    )
    for subject, metrics in report["by_subject"].items():
        print(f"subject {subject} {metrics}")
    for result in report["results"]:
        if result["status"] == "passed":
            continue
        print(
            "failure "
            f"sample_id={result['sample_id']} "
            f"errors={','.join(result['errors'])} "
            f"region_count={result['region_count']} "
            f"warnings={','.join(result['quality_warnings'])}"
        )


if __name__ == "__main__":
    raise SystemExit(main())

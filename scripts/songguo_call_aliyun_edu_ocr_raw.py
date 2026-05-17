from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from songguo.backend.services.learning.photo_review import _call_aliyun_edu_ocr_sync
from songguo.backend.services.learning.real_model_client import load_aliyun_edu_ocr_config


ACTION_ALIASES = {
    "paper_cut": "RecognizeEduPaperCut",
    "paper_ocr": "RecognizeEduPaperOcr",
    "paper_structed": "RecognizeEduPaperStructed",
    "paper_structured": "RecognizeEduPaperStructed",
    "oral_calculation": "RecognizeEduOralCalculation",
    "question_ocr": "RecognizeEduQuestionOcr",
    "formula": "RecognizeEduFormula",
}


def _decode_data(payload: Any) -> dict[str, Any]:
    body = payload.get("body") or payload.get("Body") or payload if isinstance(payload, dict) else payload
    if isinstance(body, str):
        body = json.loads(body)
    data = body.get("Data") if isinstance(body, dict) else body
    if isinstance(data, str):
        data = json.loads(data)
    return data if isinstance(data, dict) else {}


def _collect_subject_text(data: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for page in data.get("page_list") or data.get("pageList") or []:
        if not isinstance(page, dict):
            continue
        for subject in page.get("subject_list") or page.get("subjectList") or []:
            if isinstance(subject, dict):
                text = str(subject.get("text") or subject.get("content") or "").strip()
                if text:
                    texts.append(text)
    for part in data.get("part_info") or data.get("partInfo") or []:
        if not isinstance(part, dict):
            continue
        for subject in part.get("subject_list") or part.get("subjectList") or []:
            if isinstance(subject, dict):
                text = str(subject.get("text") or subject.get("content") or "").strip()
                if text:
                    texts.append(text)
    return texts


def summarize_aliyun_payload(payload: Any) -> dict[str, Any]:
    data = _decode_data(payload)
    subjects = _collect_subject_text(data)
    words = data.get("prism_wordsInfo") or []
    content = str(data.get("content") or data.get("Content") or "").strip()
    text = "\n".join(subjects) if subjects else content
    return {
        "keys": sorted(data.keys()),
        "subject_count": len(subjects),
        "word_count": len(words) if isinstance(words, list) else 0,
        "text_length": len(text),
        "text_excerpt": text[:800],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--actions",
        default="paper_cut,paper_ocr,paper_structed",
        help="Comma-separated aliases: paper_cut,paper_ocr,paper_structed,oral_calculation,question_ocr,formula",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("/private/tmp/songguo_edu_ocr_raw"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_bytes = args.image.read_bytes()
    config = load_aliyun_edu_ocr_config().model_copy(
        update={"fallback_provider": "none", "hybrid_text_fallback": False}
    )

    for alias in [item.strip() for item in args.actions.split(",") if item.strip()]:
        action = ACTION_ALIASES[alias]
        started = perf_counter()
        payload = _call_aliyun_edu_ocr_sync(
            config=config,
            action=action,
            content=image_bytes,
            filename=args.image.name,
        )
        elapsed_ms = int((perf_counter() - started) * 1000)
        out_path = args.out_dir / f"{args.image.stem}_{action}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = summarize_aliyun_payload(payload)
        print(
            json.dumps(
                {"action": action, "elapsed_ms": elapsed_ms, "raw_json": str(out_path), **summary},
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()

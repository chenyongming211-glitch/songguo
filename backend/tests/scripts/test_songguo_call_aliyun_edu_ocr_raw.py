from __future__ import annotations

import json

from scripts.songguo_call_aliyun_edu_ocr_raw import summarize_aliyun_payload


def test_summarize_paper_cut_payload_counts_subjects() -> None:
    payload = {
        "body": {
            "Data": json.dumps(
                {
                    "page_list": [
                        {
                            "subject_list": [
                                {"text": "1. 48÷6=?", "content_list_info": [{"pos": []}]},
                                {"text": "2. 24×11=(264)", "content_list_info": [{"pos": []}]},
                            ]
                        }
                    ]
                },
                ensure_ascii=False,
            )
        }
    }

    summary = summarize_aliyun_payload(payload)

    assert summary["subject_count"] == 2
    assert summary["text_excerpt"].startswith("1. 48÷6=?")


def test_summarize_paper_ocr_payload_counts_words() -> None:
    payload = {
        "body": {
            "Data": json.dumps(
                {
                    "content": "1. 48÷6=? 孩子答案：8",
                    "prism_wordsInfo": [{"word": "1. 48÷6=?", "prob": 98}],
                },
                ensure_ascii=False,
            )
        }
    }

    summary = summarize_aliyun_payload(payload)

    assert summary["word_count"] == 1
    assert "孩子答案" in summary["text_excerpt"]

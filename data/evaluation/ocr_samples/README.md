# OCR Evaluation Samples

This directory stores local OCR evaluation samples.

Use `manifest.jsonl` as the source of truth. Each line is one sample:

```json
{"sample_id":"english_tense_001","image_path":"fixtures/english_tense_001.txt","subject":"english","grade":4,"tags":["fixture","english"],"expected_items":[{"question_text":"Choose the correct tense: He ____ to school yesterday.","child_answer":"go"}]}
```

For real testing, put photos under `images/` and add rows like:

```json
{"sample_id":"phone_math_001","image_path":"images/phone_math_001.jpg","subject":"math","grade":4,"tags":["real_phone","math"],"expected_items":[{"question_text":"48 ÷ 6 = ?","child_answer":"8"}]}
```

Run deterministic fixture smoke:

```bash
PYTHONPATH=/Users/chen/code python scripts/songguo_run_ocr_eval.py --provider deterministic
```

Run real OCR through the configured vision model:

```bash
PYTHONPATH=/Users/chen/code python scripts/songguo_run_ocr_eval.py --provider vision --manifest data/evaluation/ocr_samples/manifest.jsonl --json
```

Do not commit API keys or private child homework photos unless the repository is explicitly treated as private test data storage.

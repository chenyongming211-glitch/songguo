#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from songguo.backend.evaluation.ocr_stress_samples import generate_ocr_stress_samples


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic OCR stress images and a manifest for Songguo OCR evaluation."
    )
    parser.add_argument(
        "--output",
        default="/private/tmp/songguo_ocr_stress",
        help="Output directory for images/ and manifest.jsonl.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit generated samples. Default generates all built-in cases and variants.",
    )
    args = parser.parse_args()

    manifest = generate_ocr_stress_samples(
        output_dir=Path(args.output),
        limit=args.limit or None,
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

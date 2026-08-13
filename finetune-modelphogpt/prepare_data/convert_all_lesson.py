#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convenience script to convert all lesson JSON files to prompt/response JSONL.

This wraps the existing per-domain converters:
- CTSCTT:  convert_lessons_ctsc.py
- DD:      convert_lessons_dd.py
- KTLX:    convert_lessons_ktlx.py
- PLGTDB:  convert_lesson_plgtdb.py

It assumes the raw lesson JSON files are located under `data/lesson/` and
writes outputs under `dataset/lesson/` with "_converted.jsonl" suffix.

Usage:
    python prepare_data/convert_all_lesson.py

After running, you should have files like:
    dataset/lesson/CTSCTT_converted.jsonl
    dataset/lesson/DD_converted.jsonl
    dataset/lesson/KTLX_converted.jsonl
    dataset/lesson/PLGTDB_converted.jsonl
"""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # /home/.../code_new

RAW_LESSON_DIR = ROOT / "data" / "lesson"
OUT_LESSON_DIR = ROOT / "dataset" / "lesson"


def run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT, text=True)
    if result.returncode != 0:
        raise SystemExit(f"Command failed with code {result.returncode}: {' '.join(cmd)}")


def main() -> None:
    OUT_LESSON_DIR.mkdir(parents=True, exist_ok=True)

    jobs = [
        {
            "name": "CTSCTT",
            "script": ROOT / "prepare_data" / "convert_lessons_ctsc.py",
            "input": RAW_LESSON_DIR / "CTSCTT.json",
            "output": OUT_LESSON_DIR / "CTSCTT.jsonl",
        },
        {
            "name": "DD",
            "script": ROOT / "prepare_data" / "convert_lessons_dd.py",
            "input": RAW_LESSON_DIR / "DD.json",
            "output": OUT_LESSON_DIR / "DD.jsonl",
        },
        {
            "name": "KTLX",
            "script": ROOT / "prepare_data" / "convert_lessons_ktlx.py",
            "input": RAW_LESSON_DIR / "KTLX.json",
            "output": OUT_LESSON_DIR / "KTLX.jsonl",
        },
        {
            "name": "PLGTDB",
            "script": ROOT / "prepare_data" / "convert_lesson_plgtdb.py",
            "input": RAW_LESSON_DIR / "PLGTDB.json",
            "output": OUT_LESSON_DIR / "PLGTD.jsonl",
        },
    ]

    for job in jobs:
        name = job["name"]
        script = job["script"]
        input_path = job["input"]
        output_path = job["output"]

        if not script.is_file():
            print(f"[SKIP] {name}: script not found: {script}")
            continue
        if not input_path.is_file():
            print(f"[SKIP] {name}: input file not found: {input_path}")
            continue

        cmd = [
            "python",
            str(script),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
        run(cmd)

    print("✅ All available lesson files converted.")


if __name__ == "__main__":
    main()

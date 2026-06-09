#!/usr/bin/env python3
"""
Split qa_flat.json into 4 subject files (DD.json, CTSCTT.json, KTLX.json, PLGTDB.json)
by resolving subject_id via lesson_id from lesson/lessons_index.json.

Usage (defaults):
    python split_qa_by_subject.py
    # Outputs are written to the 'data/dialogue' directory by default

Custom paths:
    python split_qa_by_subject.py -q qa_flat.json -l lesson/lessons_index.json -o data/dialogue

Notes:
- Each item in qa_flat.json is expected to have a "lesson_id" key.
- We look up lesson_id -> subject_id using lessons_index.json.
- Items with lesson_id not found in lessons_index.json are reported and skipped (or written to unmatched.json if --write-unmatched).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

# Subject ID -> filename mapping
SUBJECT_MAP = {
    1: "DD.json",        # Đạo đức
    2: "CTSCTT.json",    # Cấu tạo và sửa chữa thông thường
    3: "KTLX.json",      # Kỹ thuật lái xe
    4: "PLGTDB.json",    # Pháp luật giao thông đường bộ
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split qa_flat.json into 4 files by subject_id")
    p.add_argument("-q", "--qa", default="qa_flat.json", help="Path to qa_flat.json (array of QA items)")
    p.add_argument("-l", "--lessons", default=os.path.join("lesson", "lessons_index.json"),
                   help="Path to lessons_index.json (array with _id and subject_id)")
    p.add_argument("-o", "--outdir", default=os.path.join("data", "dialogue"), help="Output directory for split files")
    p.add_argument("--write-unmatched", action="store_true",
                   help="Also write unmatched items (lesson_id not found) to unmatched.json in outdir")
    return p.parse_args()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_lesson_subject_map(lessons: List[Dict[str, Any]]) -> Dict[str, int]:
    """Return mapping: lesson_id (_id string) -> subject_id (int)."""
    mapping: Dict[str, int] = {}
    for entry in lessons:
        # Each lesson entry should have keys: _id (uuid str), subject_id (int)
        lid = entry.get("_id")
        sid = entry.get("subject_id")
        if lid and isinstance(lid, str) and isinstance(sid, int):
            mapping[lid] = sid
    return mapping


def main() -> None:
    args = parse_args()

    qa_path = args.qa
    lessons_path = args.lessons
    outdir = args.outdir

    if not os.path.exists(qa_path):
        print(f"Input QA file not found: {qa_path}", file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(lessons_path):
        print(f"Lessons index file not found: {lessons_path}", file=sys.stderr)
        sys.exit(2)

    qa_data = load_json(qa_path)
    lessons_data = load_json(lessons_path)

    if not isinstance(qa_data, list):
        print("qa_flat must be a JSON array", file=sys.stderr)
        sys.exit(3)
    if not isinstance(lessons_data, list):
        print("lessons_index must be a JSON array", file=sys.stderr)
        sys.exit(3)

    lid_to_sid = build_lesson_subject_map(lessons_data)

    # Group items by subject_id
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    unmatched: List[Dict[str, Any]] = []

    for item in qa_data:
        if not isinstance(item, dict):
            continue
        lid = item.get("lesson_id")
        sid = lid_to_sid.get(lid)
        if sid in SUBJECT_MAP:
            grouped[sid].append(item)
        else:
            unmatched.append(item)

    # Ensure output directory exists
    os.makedirs(outdir, exist_ok=True)

    # Write 4 subject files
    totals = []
    for sid, filename in SUBJECT_MAP.items():
        path = os.path.join(outdir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(grouped.get(sid, []), f, ensure_ascii=False, indent=2)
        totals.append((sid, filename, len(grouped.get(sid, []))))

    # Optionally write unmatched
    if args.write_unmatched and unmatched:
        upath = os.path.join(outdir, "unmatched.json")
        with open(upath, "w", encoding="utf-8") as f:
            json.dump(unmatched, f, ensure_ascii=False, indent=2)
        print(f"Wrote unmatched items: {len(unmatched)} -> {upath}")

    # Summary
    print("Split summary:")
    for sid, filename, count in totals:
        print(f"  subject {sid:>2} -> {filename} : {count} items")
    if not unmatched:
        print("  All items matched a subject_id ✔")
    else:
        print(f"  Unmatched items (skipped): {len(unmatched)}")


if __name__ == "__main__":
    main()

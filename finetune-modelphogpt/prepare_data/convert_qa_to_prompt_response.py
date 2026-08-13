#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert dialogue + lesson JSON/JSONL -> JSONL with fields:
    - prompt: "### CÂU HỎI: {question}\n\n### TRẢ LỜI:\n"
    - response: answer (fallback: lesson-derived compact text)

Usage:
    Single file -> file:
        python3 convert_qa_to_prompt_response.py \
            --dialogues /path/to/dialogue.json \
            --lessons /path/to/lesson.json \
            --out datasets/out_prompt_response.jsonl

    Batch (folder -> folder):
        python3 convert_qa_to_prompt_response.py \
            --dialogues /path/to/dialogue_folder \
            --lessons /path/to/lesson_folder \
            --out-dir dataset/dialogue

            python /home/phgiakhanh/Desktop/code_new/prepare_data/convert_qa_to_prompt_response.py   --dialogues /home/phgiakhanh/Desktop/code_new/data/dialogue   --lessons /home/phgiakhanh/Desktop/code_new/data/lesson   --out-dir /home/phgiakhanh/Desktop/code_new/dataset/dialogue/
Features:
    - Normalize whitespace of questions/answers.
    - Ensure questions end with '?'.
    - De-duplicate identical questions within the same lesson by appending "(tiếp theo)" suffix on repeats.
    - Lesson fallback summary using title token preference, else concatenation capped by max_chars.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union
import sys


def read_json_or_jsonl(path: Union[str, Path]) -> Iterable[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {p}")

    # try reading as JSON list/object first (small files)
    try:
        text = p.read_text(encoding="utf-8")
        text_stripped = text.lstrip()
        if text_stripped.startswith("[") or text_stripped.startswith("{"):
            obj = json.loads(text)
            if isinstance(obj, list):
                for x in obj:
                    yield x
                return
            elif isinstance(obj, dict):
                yield obj
                return
    except Exception:
        # fall back to streaming JSONL
        pass

    # JSONL streaming
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: skipping invalid JSON line in {p}", file=sys.stderr)


def compact_ws(s: Optional[str]) -> str:
    if not s:
        return ""
    return " ".join(s.replace("\u00A0", " ").split()).strip()


def ensure_question_mark(q: str) -> str:
    q = q.strip()
    if not q:
        return q
    # Add '?' if not already present at the end
    return q if q.endswith("?") else q + "?"


def build_prompt(question: str) -> str:
    # Align with lesson converter style (uppercase Vietnamese headers)
    return f"### CÂU HỎI: {question}\n\n### TRẢ LỜI:\n"


def lesson_fallback_from_chunks(
    lesson_obj: Dict[str, Any],
    max_chars: int = 800,
    min_chunk_len: int = 30,
) -> str:
    """Create a compact fallback text from lesson chunks.

    Strategy:
      - Collect chunk texts, filter out very short ones.
      - Try to find a chunk containing a long token from the title.
      - Else concatenate in order until `max_chars`.
    """
    if not lesson_obj:
        return "Thông tin đang thiếu để trả lời chính xác."
    title = compact_ws(str(lesson_obj.get("title", "")))
    chunks = lesson_obj.get("chunks") or []
    texts = [compact_ws(c.get("text", "")) for c in chunks if c and c.get("text")]
    texts = [t for t in texts if len(t) >= min_chunk_len]
    if not texts:
        return "Thông tin đang thiếu để trả lời chính xác."

    # prefer chunk that matches title tokens
    chosen: Optional[str] = None
    if title:
        title_tokens = [tok for tok in title.split() if len(tok) >= 3]
        for tok in title_tokens:
            for t in texts:
                if tok in t:
                    chosen = t
                    break
            if chosen:
                break

    if chosen:
        return chosen[:max_chars].strip()

    # else concatenate until max_chars
    parts: List[str] = []
    total = 0
    for t in texts:
        parts.append(t)
        total += len(t)
        if total >= max_chars:
            break
    return "\n\n".join(parts)[:max_chars].strip()


def build_lesson_map(lessons_iter: Iterable[Dict[str, Any]], max_chars: int = 800) -> Dict[str, Dict[str, Any]]:
    """Return lesson_id -> lesson object with fallback precomputed."""
    m: Dict[str, Dict[str, Any]] = {}
    for ls in lessons_iter:
        key = ls.get("_id") or ls.get("id") or ls.get("lesson_id")
        if not key:
            continue
        lid = str(key)
        fallback = lesson_fallback_from_chunks(ls, max_chars=max_chars)
        m[lid] = {"lesson": ls, "fallback": fallback}
    return m


def convert(
    dialogues_path: str,
    lessons_path: str,
    out_path: str,
    max_chars: int = 800,
):
    dialogues = read_json_or_jsonl(dialogues_path)
    lessons = read_json_or_jsonl(lessons_path)

    lesson_map = build_lesson_map(lessons, max_chars=max_chars)

    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    count_in = 0
    count_out = 0
    with outp.open("w", encoding="utf-8") as fw:
        # Track question duplicates within a lesson scope
        used: Dict[Tuple[str, str], int] = {}
        for d in dialogues:
            count_in += 1
            q = compact_ws(d.get("question") if isinstance(d, dict) else d)
            a = compact_ws(d.get("answer", "")) if isinstance(d, dict) else ""

            if not q:
                continue

            # lesson scope key for de-dup: prefer lesson_id, else lesson number, else 'global'
            lesson_key = "global"
            if isinstance(d, dict):
                lid = d.get("lesson_id") or d.get("lesson")
                if lid is not None:
                    lesson_key = str(lid)

            q = ensure_question_mark(q)

            if not a:
                lid = d.get("lesson_id") if isinstance(d, dict) else None
                fallback = None
                if lid is not None:
                    fallback = lesson_map.get(str(lid), {}).get("fallback")
                if not fallback:
                    # try using lesson index by lesson number
                    try:
                        lesson_no = d.get("lesson")
                        # scan lesson_map for matching lesson number if present
                        for k, v in lesson_map.items():
                            lsobj = v.get("lesson")
                            if lsobj and lsobj.get("lesson") == lesson_no:
                                fallback = v.get("fallback")
                                break
                    except Exception:
                        pass

                a = fallback or "Thông tin đang thiếu để trả lời chính xác."

            # De-duplicate questions within same lesson
            key = (lesson_key, q)
            cnt = used.get(key, 0)
            if cnt == 0:
                final_q = q
            elif cnt == 1:
                final_q = f"{q} (tiếp theo)"
            else:
                final_q = f"{q} (tiếp theo {cnt})"
            used[key] = cnt + 1

            # Only write prompt/response (no metadata)
            rec: Dict[str, Any] = {"prompt": build_prompt(final_q), "response": a}

            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count_out += 1

    print(f"Done. Input: {count_in} | Output: {count_out} -> {outp}")


def list_json_like(p: Union[str, Path], recursive: bool = False) -> List[Path]:
    base = Path(p)
    if base.is_file():
        return [base]
    if not base.is_dir():
        raise FileNotFoundError(f"Không tìm thấy: {p}")
    patterns = ["*.json", "*.jsonl"]
    found: List[Path] = []
    if recursive:
        for pat in patterns:
            found.extend(base.rglob(pat))
    else:
        for pat in patterns:
            found.extend(base.glob(pat))
    # only take files at depth 1 when not recursive
    return sorted([x for x in found if x.is_file()])


def convert_batch(
    dialogues_dir: str,
    lessons_dir_or_file: str,
    out_dir: str,
    max_chars: int = 800,
    recursive: bool = False,
):
    dlg_files = list_json_like(dialogues_dir, recursive=recursive)
    if not dlg_files:
        raise FileNotFoundError(f"Không tìm thấy file hội thoại trong: {dialogues_dir}")

    outp_dir = Path(out_dir)
    outp_dir.mkdir(parents=True, exist_ok=True)

    lessons_path = Path(lessons_dir_or_file)
    lessons_is_dir = lessons_path.is_dir()

    total_in = 0
    total_out = 0
    produced: List[Path] = []

    for dlg_file in dlg_files:
        base = dlg_file.stem  # e.g., CTSCTT
        # Pick lesson file
        if lessons_is_dir:
            cand = lessons_path / f"{base}.json"
            if cand.exists():
                selected_lesson = cand
            else:
                # fallback: first .json in lessons dir if matching not found
                jsons = list_json_like(lessons_path)
                selected_lesson = jsons[0] if jsons else None
        else:
            selected_lesson = lessons_path

        if not selected_lesson or not Path(selected_lesson).exists():
            print(f"Warning: Không tìm thấy file bài học cho {dlg_file.name}. Vẫn tiếp tục không có fallback.")
            lesson_map: Dict[str, Dict[str, Any]] = {}
        else:
            lessons = read_json_or_jsonl(selected_lesson)
            lesson_map = build_lesson_map(lessons, max_chars=max_chars)

        # Convert this pair
        out_file = outp_dir / f"{base}.jsonl"
        count_in = 0
        count_out = 0
        with out_file.open("w", encoding="utf-8") as fw:
            used: Dict[Tuple[str, str], int] = {}
            for d in read_json_or_jsonl(dlg_file):
                count_in += 1
                q = compact_ws(d.get("question") if isinstance(d, dict) else d)
                a = compact_ws(d.get("answer", "")) if isinstance(d, dict) else ""
                if not q:
                    continue
                lesson_key = "global"
                if isinstance(d, dict):
                    lid = d.get("lesson_id") or d.get("lesson")
                    if lid is not None:
                        lesson_key = str(lid)
                q = ensure_question_mark(q)
                if not a:
                    lid = d.get("lesson_id") if isinstance(d, dict) else None
                    fallback = None
                    if lid is not None:
                        fallback = lesson_map.get(str(lid), {}).get("fallback")
                    if not fallback:
                        try:
                            lesson_no = d.get("lesson")
                            for k, v in lesson_map.items():
                                lsobj = v.get("lesson")
                                if lsobj and lsobj.get("lesson") == lesson_no:
                                    fallback = v.get("fallback")
                                    break
                        except Exception:
                            pass
                    a = fallback or "Thông tin đang thiếu để trả lời chính xác."
                key = (lesson_key, q)
                cnt = used.get(key, 0)
                if cnt == 0:
                    final_q = q
                elif cnt == 1:
                    final_q = f"{q} (tiếp theo)"
                else:
                    final_q = f"{q} (tiếp theo {cnt})"
                used[key] = cnt + 1
                rec: Dict[str, Any] = {"prompt": build_prompt(final_q), "response": a}
                fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count_out += 1

        total_in += count_in
        total_out += count_out
        produced.append(out_file)
        print(f"✓ {dlg_file.name} -> {out_file} | Input: {count_in} | Output: {count_out}")

    print(f"Done batch. Files: {len(dlg_files)} | Total Input: {total_in} | Total Output: {total_out} -> {outp_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogues", required=True, help="Path to dialogue .json/.jsonl or a folder")
    ap.add_argument("--lessons", required=True, help="Path to lesson .json/.jsonl or a folder")
    ap.add_argument("--out", help="Output file (single mode) or ignored in batch if --out-dir is used")
    ap.add_argument("--out-dir", help="Output directory for batch conversion")
    ap.add_argument("--max-chars", type=int, default=800)
    ap.add_argument("--recursive", action="store_true", help="Scan folders recursively for .json/.jsonl")
    args = ap.parse_args()
    dlg_path = Path(args.dialogues)
    # Batch mode when dialogues is a directory
    if dlg_path.is_dir():
        out_dir = args.out_dir or args.out or "dataset/dialogue"
        convert_batch(
            dialogues_dir=str(dlg_path),
            lessons_dir_or_file=args.lessons,
            out_dir=out_dir,
            max_chars=args.max_chars,
            recursive=args.recursive,
        )
    else:
        if not args.out:
            raise SystemExit("Vui lòng cung cấp --out cho chế độ 1 file hoặc dùng --out-dir cho chế độ thư mục.")
        convert(args.dialogues, args.lessons, args.out, max_chars=args.max_chars)


if __name__ == "__main__":
    main()

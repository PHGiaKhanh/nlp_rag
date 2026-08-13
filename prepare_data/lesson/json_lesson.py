#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import uuid
from pathlib import Path
from typing import List, Dict, Optional

from docx import Document  # pip install python-docx

# =========================
# CẤU HÌNH
# =========================
SUBJECT_FILES: List[str] = [
    "kb_DDNLX.docx",       # subject_id = 1
    "KichBanCTSCTT.docx",  # subject_id = 2
    "KichBanKTLX.docx",    # subject_id = 3
    "kb_PLGTDB.docx"
]

OUTPUT_JSON = "lesson/lessons_index.json"
MIN_SPLIT_LEN = 50  # đoạn ngắn hơn thế sẽ hạn chế tách theo ';'

# Các bài (lesson number) KHÔNG CHUNK → gom thành 1 chunk duy nhất
NO_CHUNK_LESSONS = set()

# ============== CẤU HÌNH CHUNK SIZE + OVERLAP ==============
# Cách chia chunk:
# - "sentence": cắt theo . / ; (không overlap, giống bản gốc)
# - "mixed": cắt theo câu trước, sau đó "đóng gói" <= CHUNK_SIZE, có OVERLAP
# - "size": cắt thẳng theo độ dài với overlap (không quan tâm ranh giới câu)
CHUNK_MODE = "mixed"        # "sentence" | "mixed" | "size"
CHUNK_SIZE = 40          # số ký tự / tokens tùy COUNT_UNIT
CHUNK_OVERLAP = 10         # số ký tự / tokens chồng lấn
COUNT_UNIT = "char"         # "char" | "token" (token cần tiktoken, có fallback)

# =========================
# TIỆN ÍCH
# =========================
def _clean(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()

def _uuid() -> str:
    return str(uuid.uuid4())

def _smart_split(text: str) -> List[str]:
    """Chỉ cắt theo dấu chấm phẩy (';').

    Quy tắc:
        - Tách theo dấu ';' và giữ lại dấu ';' ở cuối mỗi segment
      (trừ segment cuối nếu không có dấu ';' sau cùng).
        - KHÔNG gộp các segment ngắn vào đoạn trước; mọi segment sau khi tách
      đều được giữ nguyên theo ranh giới dấu ';'.
    """
    text = _clean(text)

    # Split on semicolons; keep them by splitting on ';' and re-adding
    raw_parts = [p.strip() for p in text.split(';')]
    out: List[str] = []
    for i, p in enumerate(raw_parts):
        if not p:
            continue
        seg = p.strip()
        # Gắn lại dấu ';' cho mọi segment trừ segment cuối cùng
        if i != len(raw_parts) - 1:
            seg = seg + ';'
        out.append(seg)
    return out

# --------- Bộ đo độ dài + đóng gói có overlap ----------
def _maybe_token_len(text: str) -> int:
    """Đo chiều dài theo token (nếu có tiktoken) hoặc fallback ~ số từ; mặc định theo ký tự."""
    if COUNT_UNIT == "char":
        return len(text)
    try:
        import tiktoken  # optional
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # fallback: xấp xỉ 1 token ~ 1.3 từ
        return int(len(text.split()) * 1.3)

def _seq_len(seq: List[str]) -> int:
    return _maybe_token_len(" ".join(seq).strip())

def _pack_with_overlap_sentences(sentences: List[str], max_len: int, overlap: int) -> List[str]:
    """
    Đóng gói danh sách câu thành các chunk có tổng độ dài <= max_len.
    Giữa 2 chunk có overlap tính theo COUNT_UNIT (char/token) bằng cách
    tái sử dụng phần đuôi (một số câu cuối) của chunk trước.
    """
    # With the requested change we no longer pack by character/token length.
    # Instead, each "sentence" (here a semicolon-delimited segment returned by
    # _smart_split) is treated as its own chunk. This function preserves the
    # original signature but simply returns the sentences as-is.
    return [s.strip() for s in sentences if s.strip()]

def _pack_with_overlap_chars(text: str, max_len: int, overlap: int) -> List[str]:
    """Chia thẳng theo ký tự với overlap (không quan tâm ranh giới câu)."""
    # After the new chunking policy, we don't cut by fixed char windows anymore.
    # Fall back to returning the whole text as a single chunk (or split by
    # semicolon upstream via _smart_split).
    text = text.strip()
    return [text] if text else []

# regex nhận diện "Bài <số>" với nhiều loại dấu
_BAI_ANCHOR_RE = re.compile(r"(?i)\bBài\s*[0-9ivxlcdm]+(?:\s*[:.\-–—]?)\s*")

def _extract_title_from_paragraph(paragraph) -> Optional[str]:
    """
    Lấy tiêu đề in đậm ngay sau 'Bài <số>'
    """
    runs = paragraph.runs
    raw = "".join(r.text or "" for r in runs)
    m = _BAI_ANCHOR_RE.search(raw)
    if not m:
        return None

    start = m.end()
    cur = 0
    start_run_idx, start_off = None, None
    for i, r in enumerate(runs):
        t = r.text or ""
        if cur + len(t) > start:
            start_run_idx = i
            start_off = start - cur
            break
        cur += len(t)
    if start_run_idx is None:
        return None

    # Thu thập runs bold liên tiếp
    title_parts = []
    r0 = runs[start_run_idx]
    if bool(getattr(r0, "bold", False)):
        title_parts.append(r0.text[start_off:])
        j = start_run_idx + 1
        while j < len(runs) and bool(getattr(runs[j], "bold", False)):
            title_parts.append(runs[j].text or "")
            j += 1
        title = _clean("".join(title_parts))
        return title if title else None
    return None

def _tail_after_title_in_same_paragraph(paragraph, title_text: str) -> str:
    """Phần còn lại trong paragraph sau tiêu đề"""
    raw = "".join(r.text or "" for r in paragraph.runs)
    m_anchor = _BAI_ANCHOR_RE.search(raw)
    if not m_anchor:
        return ""
    after_anchor = raw[m_anchor.end():]
    i = after_anchor.lower().find(title_text.lower())
    if i < 0:
        return ""
    after_title = after_anchor[i + len(title_text):]
    return _clean(after_title)

def parse_docx_into_lessons(docx_path: Path) -> List[Dict]:
    """
    Đọc 1 file docx → list lesson {lesson, tittle, chunks[]}
    lesson đánh số bắt đầu 1 và tăng mỗi khi gặp tiêu đề 'Bài ...'
    Với các bài trong NO_CHUNK_LESSONS, toàn bộ nội dung sẽ gom thành 1 chunk duy nhất.
    Với bài thường:
      - CHUNK_MODE="sentence": mỗi câu = 1 chunk (giống bản gốc)
      - CHUNK_MODE="mixed": gom câu rồi pack theo CHUNK_SIZE + CHUNK_OVERLAP
      - CHUNK_MODE="size": nối toàn bộ rồi cắt theo độ dài + overlap
    """
    try:
        doc = Document(str(docx_path))
    except Exception:
        return []

    lessons: List[Dict] = []
    cur: Optional[Dict] = None
    cur_is_nochunk = False
    nochunk_accumulator: List[str] = []   # gom đoạn cho bài “không chunk”
    sentence_accumulator: List[str] = []  # gom toàn bộ câu cho bài thường để pack
    lesson_idx = 0  # bắt đầu từ 0, gặp tiêu đề mới thì +1

    def _flush_current_lesson():
        """Đẩy lesson hiện tại vào danh sách, xử lý gom chunk nếu cần."""
        nonlocal cur, cur_is_nochunk, nochunk_accumulator, sentence_accumulator, lessons
        if not cur:
            return

        if cur_is_nochunk:
            full_text = _clean(" ".join(nochunk_accumulator))
            cur["chunks"] = [{"_id": _uuid(), "text": full_text}] if full_text else []
        else:
            # Đóng gói theo mode
            sentences = [s for s in (sentence_accumulator or []) if s.strip()]
            if CHUNK_MODE == "sentence":
                cur["chunks"] = [{"_id": _uuid(), "text": s} for s in sentences]
            elif CHUNK_MODE == "mixed":
                if sentences:
                    packed = _pack_with_overlap_sentences(sentences, CHUNK_SIZE, CHUNK_OVERLAP)
                else:
                    packed = []
                cur["chunks"] = [{"_id": _uuid(), "text": s} for s in packed]
            elif CHUNK_MODE == "size":
                joined = " ".join(sentences).strip()
                packed = _pack_with_overlap_chars(joined, CHUNK_SIZE, CHUNK_OVERLAP) if joined else []
                cur["chunks"] = [{"_id": _uuid(), "text": s} for s in packed]
            else:
                # fallback an toàn
                cur["chunks"] = [{"_id": _uuid(), "text": s} for s in sentences]

        lessons.append(cur)
        # reset
        cur = None
        cur_is_nochunk = False
        nochunk_accumulator = []
        sentence_accumulator = []

    for p in doc.paragraphs:
        title = _extract_title_from_paragraph(p)
        if title:
            # Khi gặp tiêu đề mới, flush bài trước (nếu có)
            _flush_current_lesson()

            # mở lesson mới
            lesson_idx += 1
            cur_is_nochunk = (lesson_idx in NO_CHUNK_LESSONS)
            cur = {
                "lesson": lesson_idx,
                "title": title,   # giữ nguyên key 'tittle' theo dữ liệu của bạn
                "chunks": []
            }

            # phần đuôi sau tiêu đề cùng dòng (nếu có)
            tail = _tail_after_title_in_same_paragraph(p, title)
            if tail:
                if cur_is_nochunk:
                    nochunk_accumulator.append(tail)
                else:
                    sentence_accumulator.extend(_smart_split(tail))
            continue

        text = _clean(p.text or "")
        if not text:
            continue
        if cur is None:
            # bỏ qua các đoạn trước khi gặp "Bài ..."
            continue

        if cur_is_nochunk:
            nochunk_accumulator.append(text)
        else:
            sentence_accumulator.extend(_smart_split(text))

    # Kết thúc file: flush bài cuối
    _flush_current_lesson()
    return lessons

# =========================
# MAIN
# =========================
def main():
    lesson_records: List[Dict] = []
    for idx, file_name in enumerate(SUBJECT_FILES, start=1):
        docx_path = Path(file_name)
        if not docx_path.exists():
            print(f"⚠️ Không tìm thấy {docx_path}, bỏ qua.")
            continue

        lessons = parse_docx_into_lessons(docx_path)
        print(f"{file_name}: {len(lessons)} lessons")

        # Gắn subject_id + giữ nguyên lesson đã đánh số trong parse()
        for les in lessons:
            record = {
                "_id": _uuid(),
                "subject_id": idx,
                "lesson": les["lesson"],   # số bài
                "title": les["title"],
                "chunks": les["chunks"],
            }
            lesson_records.append(record)

    out_path = Path(OUTPUT_JSON)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(lesson_records, f, ensure_ascii=False, indent=2)
    print(f"OK! {len(lesson_records)} lessons → {out_path.resolve()}")

if __name__ == "__main__":
    main()

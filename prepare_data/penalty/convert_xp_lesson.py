#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from docx import Document  # pip install python-docx

# =========================
# CẤU HÌNH
# =========================
DOC_PATH = r"C:\Users\pc\Desktop\code_llm\rag\AI.LLM\168_2024_ND-CP_619502.docx"
OUTPUT_JSON = "penalty/lessons_xp.json"

# Regex nhận diện "Chương" (để xác định subject_id)
_CHUONG_RE = re.compile(r"(?i)Chương\s+([IVivxlcdm]+)", re.IGNORECASE)

# Regex nhận diện "Điều" in đậm (để xác định lesson mới)
_DIEU_RE = re.compile(r"(?i)Điều\s+\d+", re.IGNORECASE)

# Regex nhận diện số thứ tự ở đầu dòng (1., 2., etc.) cho Khoản
_KHOANG_RE = re.compile(r"^\s*(\d+)\s*[.\-:]\s*")

# =========================
# TIỆN ÍCH
# =========================
def _clean(s: str) -> str:
    """Loại bỏ khoảng trắng thừa."""
    if not s:
        return ""
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()

def _uuid() -> str:
    """Tạo UUID."""
    return str(uuid.uuid4())

def _roman_to_int(s: str) -> int:
    """Chuyển số La Mã thành số Ả Rập."""
    s = s.upper()
    roman_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0
    prev_val = 0
    for char in reversed(s):
        val = roman_values.get(char, 0)
        if val < prev_val:
            total -= val
        else:
            total += val
        prev_val = val
    return total

def _extract_chuong_number(text: str) -> Optional[int]:
    """Trích xuất số chương từ text 'Chương I', 'Chương II', etc."""
    m = _CHUONG_RE.search(text)
    if m:
        roman = m.group(1)
        return _roman_to_int(roman)
    return None

def _extract_title_from_paragraph(paragraph) -> Optional[str]:
    """
    Lấy tiêu đề in đậm từ đoạn (bắt đầu từ 'Điều')
    """
    runs = paragraph.runs
    raw = "".join(r.text or "" for r in runs)
    
    m = _DIEU_RE.search(raw)
    if not m:
        return None

    start = m.start()
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

    # Thu thập runs bold liên tiếp từ vị trí "Điều"
    title_parts = []
    for j in range(start_run_idx, len(runs)):
        r = runs[j]
        if j == start_run_idx:
            # Lấy phần từ start_off đến cuối run
            text_part = r.text[start_off:] if r.text else ""
        else:
            text_part = r.text or ""
        
        # Kiểm tra xem run này in đậm không
        if bool(getattr(r, "bold", False)):
            title_parts.append(text_part)
        else:
            # Nếu không in đậm nữa, dừng lại
            break
    
    title = _clean("".join(title_parts))
    return title if title else None

def _tail_after_title_in_same_paragraph(paragraph, title_text: str) -> str:
    """Phần còn lại trong paragraph sau tiêu đề"""
    raw = "".join(r.text or "" for r in paragraph.runs)
    m = _DIEU_RE.search(raw)
    if not m:
        return ""
    
    after_anchor = raw[m.start():]
    i = after_anchor.lower().find(title_text.lower())
    if i < 0:
        return ""
    after_title = after_anchor[i + len(title_text):]
    return _clean(after_title)

def _split_into_khoang(text: str) -> List[Tuple[Optional[int], str]]:
    """
    Chia text thành các khoản dựa trên regex số ở đầu dòng.
    Trả về list (khoang_number, text_content)
    """
    lines = text.split('\n')
    khoang_list: List[Tuple[Optional[int], str]] = []
    
    current_khoang = None
    current_text = []
    
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            if current_text:
                current_text.append("")  # preserve empty lines
            continue
        
        m = _KHOANG_RE.match(line_clean)
        if m:
            # Tìm thấy khoản mới
            if current_khoang is not None or current_text:
                # Lưu khoản trước đó
                text_content = _clean(" ".join(current_text))
                if text_content:
                    khoang_list.append((current_khoang, text_content))
            
            # Bắt đầu khoản mới
            khoang_num = int(m.group(1))
            remaining_text = line_clean[m.end():]
            current_khoang = khoang_num
            current_text = [remaining_text] if remaining_text else []
        else:
            # Tiếp tục text của khoản hiện tại
            current_text.append(line_clean)
    
    # Xử lý khoản cuối cùng
    if current_khoang is not None or current_text:
        text_content = _clean(" ".join(current_text))
        if text_content:
            khoang_list.append((current_khoang, text_content))
    
    return khoang_list

def parse_doc_into_lessons(doc_path: Path) -> List[Dict]:
    """
    Đọc file .doc → list lesson
    - Bắt "Chương" để xác định subject_id
    - Bắt "Điều" in đậm để xác định bài (lesson) mới
    - Chia nội dung thành các khoản (chunks)
    """
    try:
        doc = Document(str(doc_path))
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return []

    lessons: List[Dict] = []
    current_lesson: Optional[Dict] = None
    current_subject_id = 5  # Mặc định chương đầu tiên là 5
    lesson_idx = 0
    body_accumulator: List[str] = []  # Gom nội dung cho lesson hiện tại

    def _flush_current_lesson():
        """Đẩy lesson hiện tại vào danh sách."""
        nonlocal current_lesson, lessons, body_accumulator
        
        if not current_lesson:
            return

        # Chia nội dung thành các khoản
        body_text = "\n".join(body_accumulator)
        khoang_list = _split_into_khoang(body_text)
        
        chunks = []
        for khoang_num, khoang_text in khoang_list:
            if khoang_num is not None:
                # Thêm "Khoản" trước số
                chunk_text = f"Khoản {khoang_num}. {khoang_text}"
            else:
                chunk_text = khoang_text
            
            if chunk_text.strip():
                chunks.append({
                    "_id": _uuid(),
                    "text": chunk_text
                })
        
        current_lesson["chunks"] = chunks
        lessons.append(current_lesson)
        
        # Reset
        current_lesson = None
        body_accumulator = []

    for p in doc.paragraphs:
        text = _clean(p.text or "")
        if not text:
            continue
        
        # Kiểm tra xem có "Chương" không
        chuong_num = _extract_chuong_number(text)
        if chuong_num is not None:
            current_subject_id = 4 + chuong_num  # Chương I = 5, II = 6, etc.
            continue
        
        # Kiểm tra xem có "Điều" in đậm không
        title = _extract_title_from_paragraph(p)
        if title:
            # Flush lesson trước đó
            _flush_current_lesson()
            
            # Tạo lesson mới
            lesson_idx += 1
            current_lesson = {
                "_id": _uuid(),
                "subject_id": current_subject_id,
                "lesson": lesson_idx,
                "title": title,
                "chunks": []
            }
            
            # Lấy phần đuôi sau tiêu đề nếu có
            tail = _tail_after_title_in_same_paragraph(p, title)
            if tail:
                body_accumulator.append(tail)
            continue
        
        # Nếu có lesson đang mở, gom nội dung
        if current_lesson is not None:
            body_accumulator.append(text)

    # Flush lesson cuối cùng
    _flush_current_lesson()
    return lessons

# =========================
# MAIN
# =========================
def main():
    doc_path = Path(DOC_PATH)
    
    if not doc_path.exists():
        print(f"❌ Không tìm thấy file {doc_path}")
        return
    
    print(f"📖 Đang đọc {doc_path.name}...")
    lessons = parse_doc_into_lessons(doc_path)
    
    print(f"✅ Trích xuất {len(lessons)} bài")
    
    out_path = Path(OUTPUT_JSON)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(lessons, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Lưu vào {out_path.resolve()}")

if __name__ == "__main__":
    main()

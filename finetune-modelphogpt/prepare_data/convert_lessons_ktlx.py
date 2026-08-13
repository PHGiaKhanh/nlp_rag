#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert KTLX (Kỹ thuật lái xe) lesson JSON
into JSONL fine-tune samples: {"prompt": <str>, "response": <str>}.

Heuristics tailored for the KTLX domain to create meaningful Vietnamese Q&A pairs.
Each chunk becomes one example. Questions are phrased to match procedural/feature descriptions.

Usage:
    python convert_lessons_ktlx.py --input data/lesson/KTLX.json --output dataset/lesson/KTLX_converted.jsonl
"""
import argparse
import json
import os
import re
import sys
from typing import Dict, Iterable, Iterator, List, Union

# =============== Regex patterns (KTLX domain) ===============
# Mục dạng nhãn kèm dấu hai chấm: "Vô lăng:" "Phanh tay:" "Khóa điện:" ...
ITEM_COLON_RE = re.compile(r"^\s*([A-ZÀ-Ỹa-zà-ỹ0-9][^:\n]{1,120}):\s+", flags=re.UNICODE)

# Tư thế, điều chỉnh
DIEU_CHINH_GHE_RE = re.compile(r"^\s*Điều\s+chỉnh\s+ghế\s+lái", flags=re.IGNORECASE | re.UNICODE)
DIEU_CHINH_GUONG_RE = re.compile(r"^\s*Điều\s+chỉnh\s+gương\s+chiếu\s+hậu", flags=re.IGNORECASE | re.UNICODE)
CAI_DAY_AN_TOAN_RE = re.compile(r"^\s*Cài\s+dây\s+an\s+toàn", flags=re.IGNORECASE | re.UNICODE)
TU_THE_LAI_RE = re.compile(r"tư\s*thế\s*ngồi", flags=re.IGNORECASE | re.UNICODE)

# Vô lăng, bàn đạp
THAO_TAC_VO_LANG_RE = re.compile(r"^\s*Thao\s+tác\s+cầm\s+vô\s+lăng", flags=re.IGNORECASE | re.UNICODE)
LY_HOP_RE = re.compile(r"\bbàn\s+đạp\s+ly\s+hợp\b", flags=re.IGNORECASE | re.UNICODE)
PHANH_CHAN_RE = re.compile(r"\bbàn\s+đạp\s+phanh\b|phanh\s+gấp", flags=re.IGNORECASE | re.UNICODE)
BAN_DAP_GA_RE = re.compile(r"\bbàn\s+đạp\s+ga\b", flags=re.IGNORECASE | re.UNICODE)

# Cần số, số tự động
CAN_SO_RE = re.compile(r"^\s*Thao\s+tác\s+với\s+cần\s+số|^\s*Với\s+xe\s+số\s+cơ\s+khí|^\s*Đối\s+với\s+xe\s+số\s+tự\s+động", flags=re.IGNORECASE | re.UNICODE)
SO_TU_DONG_RE = re.compile(r"xe\s+số\s+tự\s+động|\bP\b|\bR\b|\bN\b|\bD\b", flags=re.IGNORECASE | re.UNICODE)

# Khởi động / tắt máy
KHOI_DONG_RE = re.compile(r"^\s*Trước\s+khi\s+khởi\s+động|^\s*Với\s+xe\s+số\s+sàn", flags=re.IGNORECASE | re.UNICODE)
TAT_DONG_CO_RE = re.compile(r"\bkhi\s+tắt\s+máy\b|xoay\s+chìa\s+về\s+LOCK|tắt\s+động\s+cơ", flags=re.IGNORECASE | re.UNICODE)

# Khởi hành, dừng, tăng giảm tốc
KHOI_HANH_RE = re.compile(r"^\s*Số\s+cơ\s+khí:|^\s*Số\s+tự\s+động:", flags=re.IGNORECASE | re.UNICODE)
DUNG_XE_RE = re.compile(r"\bKhi\s+dừng\s+xe\b", flags=re.IGNORECASE | re.UNICODE)

# Lùi xe, ghép dọc/ngang/chéo
LUI_XE_RE = re.compile(r"^\s*Trước\s+khi\s+lùi\s+xe|^\s*Khi\s+lùi\b|lùi\s+ghép\s+dọc|lùi\s+ghép\s+ngang|ghép\s+chéo", flags=re.IGNORECASE | re.UNICODE)

# Chuyển làn, rẽ, vượt
CHUYEN_LAN_RE = re.compile(r"^\s*Khi\s+chuyển\s+làn", flags=re.IGNORECASE | re.UNICODE)
RE_TRAI_RE = re.compile(r"\brẽ\s+trái\b", flags=re.IGNORECASE | re.UNICODE)
RE_PHAI_RE = re.compile(r"\brẽ\s+phải\b", flags=re.IGNORECASE | re.UNICODE)
VUOT_XE_RE = re.compile(r"\bVượt\s+xe\b", flags=re.IGNORECASE | re.UNICODE)

# Quay đầu
QUAY_DAU_RE = re.compile(r"^\s*Khi\s+quay\s+đầu\s+xe|\bquay\s+đầu\b", flags=re.IGNORECASE | re.UNICODE)

# Quan sát, điểm mù, tín hiệu
GUONG_CHIEU_HAU_RE = re.compile(r"gương\s+chiếu\s+hậu", flags=re.IGNORECASE | re.UNICODE)
DIEM_MU_RE = re.compile(r"điểm\s+mù", flags=re.IGNORECASE | re.UNICODE)
TIN_HIEU_DEN_RE = re.compile(r"đèn\s+(báo|tín\s+hiệu|khẩn\s+cấp|chiếu\s+sáng)", flags=re.IGNORECASE | re.UNICODE)

# Điều hòa, giải trí, gạt nước
DIEU_HOA_RE = re.compile(r"Hệ\s+thống\s+điều\s+hòa", flags=re.IGNORECASE | re.UNICODE)
GAT_NUOC_RE = re.compile(r"gạt\s+nước|rửa\s+kính", flags=re.IGNORECASE | re.UNICODE)
GIAI_TRI_RE = re.compile(r"Hệ\s+thống\s+giải\s+trí", flags=re.IGNORECASE | re.UNICODE)
KINH_CUA_RE = re.compile(r"^\s*Điều\s+chỉnh\s+kính\s+cửa", flags=re.IGNORECASE | re.UNICODE)

# Chuẩn bị, kiểm tra
CHUAN_BI_KIEM_TRA_RE = re.compile(r"^\s*Trước\s+khi\s+lái\s+xe|^\s*Trước\s+khi\s+đưa\s+xe\s+ra|^\s*Khi\s+lên\s+xe|^\s*Khi\s+xuống\s+xe", flags=re.IGNORECASE | re.UNICODE)

# =============== Helpers ===============

def read_json(path: str) -> Union[Dict, List]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_files(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        files = []
        for root, _, names in os.walk(input_path):
            for n in names:
                if n.lower().endswith((".json", ".jsonl")):
                    files.append(os.path.join(root, n))
        return sorted(files)
    return [input_path]


def normalize_text(s: str) -> str:
    s = s.replace("\u00A0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    # Normalize explicit newline markers from source to real newlines
    s = s.replace("/n", "\n")
    return s.strip()


def format_response(text: str) -> str:
    """Clean and optimize the response text so it's precise and readable.
    - Remove leading item label before ':' (e.g., "Vô lăng:")
    - Strip trailing semicolons and normalize punctuation
    - Preserve newlines already normalized by normalize_text
    """
    t = text.strip()
    # Remove leading label before colon
    m = re.match(r"^\s*([A-ZÀ-Ỹa-zà-ỹ0-9][^:\n]{1,120}):\s+(.*)$", t)
    if m:
        t = m.group(2).strip()
    # Trim trailing semicolons per source style
    t = re.sub(r";\s*$", ".", t)
    # Normalize bullet lines introduced by source "/n-" forms
    t = re.sub(r"\n\s*-\s*", "\n- ", t)
    # Ensure sentence ends with period if not present
    if not re.search(r"[\.!?]$", t):
        t = t + "."
    # Normalize spaces around newlines (already converted in normalize_text)
    t = re.sub(r"\s*\n\s*", "\n", t)
    return t


def _clean_term(term: str, max_words: int = 10) -> str:
    term = term.strip(' "“”\'’:;,.()[]')
    # If label contains a parenthetical qualifier, keep the main part before '('
    if '(' in term:
        term = term.split('(', 1)[0].strip()
    words = term.split()
    if len(words) == 0 or len(words) > max_words:
        return ""
    if term and term[0].islower():
        term = term[0].upper() + term[1:]
    return term

# =============== Core heuristic ===============

def guess_question_from_chunk(chunk_text: str, title: str, index: int) -> str:
    # 1. Direct item label: "Tên mục: ..."
    m = ITEM_COLON_RE.match(chunk_text)
    if m:
        item = _clean_term(m.group(1), max_words=8)
        if item:
            # More natural pairing with colon-labeled function descriptions
            return f"{item} có chức năng gì và cách sử dụng an toàn như thế nào?"

    # Chuẩn bị và kiểm tra
    if CHUAN_BI_KIEM_TRA_RE.match(chunk_text):
        return "Cần chuẩn bị và kiểm tra những gì trước/sau khi lên xuống xe?"

    # Ghế, gương, dây an toàn
    if DIEU_CHINH_GHE_RE.match(chunk_text) or TU_THE_LAI_RE.search(chunk_text):
        return "Cách điều chỉnh ghế lái và tư thế ngồi đúng là gì?"
    # Ưu tiên lùi xe trước gương để tránh gán câu hỏi lùi xe thành gương
    if LUI_XE_RE.search(chunk_text):
        return "Các bước lùi xe và ghép vào chỗ đỗ (dọc/ngang/chéo) an toàn là gì?"
    if DIEU_CHINH_GUONG_RE.match(chunk_text) or DIEM_MU_RE.search(chunk_text) or GUONG_CHIEU_HAU_RE.search(chunk_text):
        return "Điều chỉnh gương chiếu hậu thế nào để giảm điểm mù và quan sát tốt?"
    if CAI_DAY_AN_TOAN_RE.match(chunk_text):
        return "Cách cài dây an toàn đúng cho người lớn, phụ nữ mang thai và trẻ em là gì?"

    # Vô lăng, bàn đạp
    if THAO_TAC_VO_LANG_RE.match(chunk_text):
        return "Kỹ thuật cầm và điều khiển vô lăng ở các tốc độ như thế nào?"
    if LY_HOP_RE.search(chunk_text):
        return "Cách đạp/nhả bàn đạp ly hợp đúng kỹ thuật là gì?"
    if BAN_DAP_GA_RE.search(chunk_text):
        return "Đặt chân và tăng ga đúng cách như thế nào?"
    if PHANH_CHAN_RE.search(chunk_text):
        return "Sử dụng bàn đạp phanh an toàn, nhất là khi phanh gấp, như thế nào?"

    # Cần số, số tự động
    if CAN_SO_RE.search(chunk_text):
        return "Thao tác chuyển số đúng kỹ thuật với xe số sàn và số tự động là gì?"
    if SO_TU_DONG_RE.search(chunk_text):
        return "Ý nghĩa các ký hiệu P, R, N, D và lưu ý khi sử dụng số tự động là gì?"

    # Khởi động/tắt máy
    if KHOI_DONG_RE.search(chunk_text):
        return "Quy trình kiểm tra và khởi động động cơ cho các loại xe là gì?"
    if TAT_DONG_CO_RE.search(chunk_text):
        return "Cách tắt động cơ an toàn đối với máy xăng và Diesel là gì?"

    # Khởi hành, dừng, tăng/giảm tốc
    if KHOI_HANH_RE.search(chunk_text):
        return "Quy trình khởi hành, tăng giảm số và dừng xe với số sàn/số tự động là gì?"
    if DUNG_XE_RE.search(chunk_text):
        return "Các bước dừng và đỗ xe an toàn gồm những gì?"

    # Dừng/đỗ trên dốc
    if re.search(r"đỗ\s+trên\s+đường\s+dốc|trên\s+dốc", chunk_text, flags=re.IGNORECASE):
        return "Lưu ý khi dừng/đỗ xe trên dốc để tránh trôi xe là gì?"

    # Chuyển làn, rẽ, vượt
    if CHUYEN_LAN_RE.search(chunk_text):
        if re.search(r"từ\s+tốc\s+độ\s+cao\s+sang\s+thấp", chunk_text, flags=re.IGNORECASE):
            return "Khi chuyển làn từ tốc độ cao sang thấp cần làm gì để an toàn?"
        if re.search(r"từ\s+làn\s+tốc\s+độ\s+thấp\s+sang\s+cao", chunk_text, flags=re.IGNORECASE):
            return "Khi chuyển làn từ tốc độ thấp sang cao cần làm gì để an toàn?"
        return "Quy trình chuyển làn an toàn gồm những bước nào?"
    if RE_TRAI_RE.search(chunk_text):
        return "Quy trình rẽ trái an toàn tại ngã ba/ngã tư là gì?"
    if RE_PHAI_RE.search(chunk_text):
        return "Quy trình rẽ phải an toàn là gì?"
    if VUOT_XE_RE.search(chunk_text):
        return "Các bước vượt xe trên đường hai chiều một cách an toàn là gì?"

    # Đèn, điều hòa, gạt nước, giải trí
    if TIN_HIEU_DEN_RE.search(chunk_text):
        return "Sử dụng đèn khẩn cấp, đèn chiếu sáng và tín hiệu đúng lúc như thế nào?"
    if DIEU_HOA_RE.search(chunk_text):
        return "Vai trò và cách sử dụng hệ thống điều hòa trên xe là gì?"
    if KINH_CUA_RE.match(chunk_text):
        return "Chức năng và cách sử dụng điều chỉnh kính cửa như thế nào?"
    if GAT_NUOC_RE.search(chunk_text):
        return "Cách sử dụng gạt nước và rửa kính để đảm bảo tầm nhìn là gì?"
    if GIAI_TRI_RE.search(chunk_text):
        return "Hệ thống giải trí trên xe giúp ích như thế nào khi lái xe?"

    # Fallback: nếu có nhãn thiết bị, hỏi tác dụng; nếu không, tóm tắt
    # Thử bắt nhãn thiết bị dạng đầu câu trước dấu chấm phẩy
    m2 = re.match(r"^\s*([A-ZÀ-Ỹa-zà-ỹ0-9][^;]{1,80});", chunk_text)
    if m2:
        item2 = _clean_term(m2.group(1), max_words=8)
        if item2:
            return f"{item2} có chức năng gì và cách vận hành an toàn?"

    return "Tóm tắt ý chính của đoạn sau."  # generic safe fallback

# =============== Emission ===============

def records_from_any(path: str) -> Iterator[Dict]:
    if path.lower().endswith(".jsonl"):
        yield from read_jsonl(path)
    else:
        obj = read_json(path)
        if isinstance(obj, list):
            yield from obj
        else:
            yield obj


def iterate_all_objects(input_paths: List[str]) -> Iterator[Dict]:
    for p in input_paths:
        yield from records_from_any(p)


def _specialize_question_for_duplicate(q: str, text: str, idx: int) -> str:
    # Try to append a concise qualifier derived from the chunk's leading phrase
    lead = re.split(r"[\.;\n]", text.strip())[0]
    # Remove common prefaces
    lead = re.sub(r"^\s*(Khi|Trước\s+khi|Sau\s+khi|Đối\s+với|Với|Nếu|Tại|Trong)\s+", "", lead, flags=re.IGNORECASE)
    # Keep a short span
    # Word-safe truncation ~40 chars
    max_len = 40
    if len(lead) > max_len:
        cut = lead[:max_len]
        # avoid cutting the last word in half
        last_space = cut.rfind(" ")
        if last_space > 10:
            cut = cut[:last_space]
        lead = cut
    qualifier = lead.strip()
    qualifier = qualifier.strip(' "“”\'’:;,.()[]')
    if qualifier:
        return f"{q} — {qualifier}"
    # Fallback: numbered part
    return f"{q} (phần {idx + 1})"


def emit_examples(obj: Dict) -> Iterator[Dict]:
    title = obj.get("title", "")
    chunks = obj.get("chunks", []) or []
    seen_questions: Dict[str, int] = {}
    for i, ch in enumerate(chunks):
        text = normalize_text(ch.get("text", ""))
        if not text:
            continue
        question = guess_question_from_chunk(text, title, i)
        # De-duplicate questions within the same lesson by specializing on content
        if question in seen_questions:
            count = seen_questions[question] + 1
            specialized = _specialize_question_for_duplicate(question, text, count - 1)
            # record both base and specialized to avoid immediate re-collision
            seen_questions[question] = count
            seen_questions[specialized] = 0
            question = specialized
        else:
            seen_questions[question] = 0
        prompt = f"### CÂU HỎI: {question}\n\n### TRẢ LỜI:\n"
        yield {"prompt": prompt, "response": format_response(text)}


def write_jsonl(examples: Iterable[Dict], out_path: str) -> int:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n += 1
    return n

# =============== CLI ===============

def main():
    ap = argparse.ArgumentParser(description="Convert KTLX lessons JSON -> prompt/response JSONL.")
    ap.add_argument("--input", required=True, help="Path to input KTLX.json or folder")
    ap.add_argument("--output", required=True, help="Output .jsonl path")
    args = ap.parse_args()

    files = iter_files(args.input)
    if not files:
        print("No input files found.", file=sys.stderr)
        sys.exit(2)

    def gen():
        for obj in iterate_all_objects(files):
            yield from emit_examples(obj)

    total = write_jsonl(gen(), args.output)
    print(f"✅ Done. Wrote {total} examples to {args.output}")

if __name__ == "__main__":
    main()

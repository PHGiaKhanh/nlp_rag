#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert PLGTDB (Pháp luật giao thông đường bộ) lessons JSON
into JSONL fine-tune samples: {"prompt": <str>, "response": <str>}.

Heuristics tailored for traffic-law domain to create clear Vietnamese Q&A pairs.
Each chunk becomes one example. Questions are phrased to match legal definitions,
quy tắc, nguyên tắc, biển báo, tốc độ, quyền ưu tiên, v.v.

Usage:
    python prepare_data/convert_lesson_plgtdb.py \
        --input data/lesson/PLGTDB.json \
        --output dataset/lesson/PLGTDB_converted.jsonl
"""
import argparse
import json
import os
import re
import sys
from typing import Dict, Iterable, Iterator, List, Union

# =============== Regex patterns (PLGTDB domain) ===============
# Generic structure
TERM_DEF_RE = re.compile(r"^\s*([A-ZÀ-Ỹa-zà-ỹ0-9][^:\n]{1,120})\s+là\s+", flags=re.UNICODE)
ITEM_COLON_RE = re.compile(r"^\s*([A-ZÀ-Ỹa-zà-ỹ0-9][^:\n]{1,120}):\s+", flags=re.UNICODE)
BAO_GOM_RE = re.compile(r"\b(bao\s*gồm|gồm):?\b", flags=re.IGNORECASE | re.UNICODE)
CAM_RE = re.compile(r"\b(bị\s+nghiêm\s+cấm|cấm)\b", flags=re.IGNORECASE | re.UNICODE)

# Thematic rules
TOC_DO_RE = re.compile(r"tốc\s+độ|cự\s+ly\s+tối\s+thiểu|khu\s+đông\s+dân\s+cư", flags=re.IGNORECASE | re.UNICODE)
VUOT_RE = re.compile(r"\bvượt\s+xe\b|cấm\s+vượt", flags=re.IGNORECASE | re.UNICODE)
NHUONG_DUONG_RE = re.compile(r"nhường\s+đường|vòng\s+xuyến|giao\s+lộ|giao\s+nhau", flags=re.IGNORECASE | re.UNICODE)
DUONG_SAT_RE = re.compile(r"đường\s+sắt|đường\s+ngang|cầu\s+chung\s+đường\s+sắt", flags=re.IGNORECASE | re.UNICODE)
PHA_RE = re.compile(r"\bphà\b|cầu\s+phao", flags=re.IGNORECASE | re.UNICODE)
CAO_TOC_RE = re.compile(r"cao\s+tốc", flags=re.IGNORECASE | re.UNICODE)
HAM_RE = re.compile(r"hầm\s+đường\s+bộ|đường\s+hầm", flags=re.IGNORECASE | re.UNICODE)
DUNG_DO_RE = re.compile(r"dừng\s+và\s+đỗ|cấm\s+dừng|cấm\s+đỗ|khi\s+dừng\s+xe|khi\s+đỗ\s+xe", flags=re.IGNORECASE | re.UNICODE)
CUA_DEN_COI_RE = re.compile(r"mở\s+cửa\s+xe|sử\s+dụng\s+đèn|tín\s+hiệu\s+còi|còi\s+điện|đèn\s+chiếu\s+sáng", flags=re.IGNORECASE | re.UNICODE)
XE_UU_TIEN_RE = re.compile(r"xe\s+ưu\s+tiên|tín\s+hiệu\s+ưu\s+tiên", flags=re.IGNORECASE | re.UNICODE)
KEO_ROMOOC_RE = re.compile(r"rơ\s*moóc|sơ\s*mi\s*rơ\s*moóc|kéo\s+xe", flags=re.IGNORECASE | re.UNICODE)
GPLX_RE = re.compile(r"giấy\s+phép\s+lái\s+xe|hạng\s+[ABCD]\w*", flags=re.IGNORECASE | re.UNICODE)
GPLX_DIEM_RE = re.compile(r"điểm\s+của\s+giấy\s+phép\s+lái\s+xe|trừ\s+điểm|12\s+điểm", flags=re.IGNORECASE | re.UNICODE)
DO_TUOI_SUC_KHOE_RE = re.compile(r"đủ\s+(tuổi|sức\s+khỏe)|độ\s+tuổi", flags=re.IGNORECASE | re.UNICODE)
SAT_HACH_RE = re.compile(r"(đào\s+tạo|sát\s+hạch|thi|mô\s+phỏng|thực\s+hành)\b", flags=re.IGNORECASE | re.UNICODE)
LAN_DUONG_RE = re.compile(r"làn\s+đường\s+dành\s+riêng|làn\s+đường\s+dành|làn\s+đường", flags=re.IGNORECASE | re.UNICODE)
VACH_KE_DUONG_RE = re.compile(r"vạch\s+kẻ\s+đường", flags=re.IGNORECASE | re.UNICODE)

# Sign-specific
BIEN_SO_RE = re.compile(r"Biển\s+số\s+([A-ZĐ]\.?(:?[A-Z])?,?\.?\d+[a-z]?(?:,?\s*[A-ZĐ]?\.?[A-Z]?,?\.?\d+[a-z]?)*)(?=\s|\W)", flags=re.UNICODE)
# Tighten to avoid matching generic 'cảnh báo' texts not about signs
NHOM_BIEN_RE = re.compile(r"Biển\s+(báo\s+cấm|nguy\s+hiểm|hiệu\s+lệnh|chỉ\s+dẫn)", flags=re.IGNORECASE | re.UNICODE)

# Traffic light / light usage specifics
DEN_VANG_NN_RE = re.compile(r"đèn\s+vàng\s+nhấp\s+nháy", flags=re.IGNORECASE | re.UNICODE)
DEN_DO_NN_RE = re.compile(r"đèn\s+đỏ\s+nhấp\s+nháy", flags=re.IGNORECASE | re.UNICODE)
DEN_HAI_MAU_RE = re.compile(r"đèn\s+hai\s+màu|đèn\s+2\s+màu", flags=re.IGNORECASE | re.UNICODE)
DEN_MOT_MAU_RE = re.compile(r"đèn\s+một\s+màu|đèn\s+1\s+màu", flags=re.IGNORECASE | re.UNICODE)
DEN_PHU_RE = re.compile(r"đèn\s+phụ\b|mũi\s+tên\s+chỉ\s+hướng", flags=re.IGNORECASE | re.UNICODE)

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
    # Map custom newline marker to real newline
    s = s.replace("/n", "\n")
    s = re.sub(r"\s+", " ", s)
    # Recover intended newlines around list-like markers
    s = re.sub(r"\s*\n\s*", "\n", s)
    return s.strip()


def _clean_term(term: str, max_words: int = 12) -> str:
    term = term.strip(' "“”\'’:;,.()[]')
    if '(' in term:
        term = term.split('(', 1)[0].strip()
    words = term.split()
    if len(words) == 0 or len(words) > max_words:
        return ""
    # Capitalize first letter if needed
    if term and term[0].islower():
        term = term[0].upper() + term[1:]
    return term


def format_response(text: str) -> str:
    """Clean and optimize the response text so it's precise and readable.
    - Remove leading item label before ':' if present
    - Strip trailing semicolons and normalize punctuation
    - Normalize bullet-like dashes
    - Preserve newlines already normalized by normalize_text
    """
    t = text.strip()
    # Drop common heading noise like "Bài 27", "Những quy định chung"
    t = re.sub(r"^(Bài\s*\d+\s*[:\-]?\s*)", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^(Những\s+quy\s+định\s+chung\s*[:\-]?\s*)", "", t, flags=re.IGNORECASE)
    # Remove leading label before colon
    m = ITEM_COLON_RE.match(t)
    if m:
        t = t[m.end():].strip()
    # Trim trailing semicolon
    t = re.sub(r";\s*$", ".", t)
    # Fix accidental duplicated words e.g. "theo theo"
    t = re.sub(r"\b(\w+)\s+\1\b", r"\1", t, flags=re.IGNORECASE)
    # Normalize bullets
    t = re.sub(r"\n\s*-\s*", "\n- ", t)
    # Ensure sentence-ish ending
    if not re.search(r"[\.!?]$", t):
        t = t + "."
    return t

# =============== Core heuristic ===============

def _extract_sign_codes(chunk_text: str) -> str:
    m = BIEN_SO_RE.search(chunk_text)
    if not m:
        return ""
    codes = m.group(1)
    # Clean extra spaces and commas
    codes = re.sub(r"\s+", " ", codes)
    codes = re.sub(r"\s*,\s*", ", ", codes)
    return codes.strip()


def _subject_before_baogom(text: str) -> str:
    """Extract a subject immediately preceding 'bao gồm/gồm' to form specific questions.
    Take the last clause before the phrase and clean it.
    """
    m = re.search(r"(.{3,200}?)(?:\s+(?:bao\s*gồm|gồm))\b", text, flags=re.IGNORECASE | re.UNICODE)
    if not m:
        return ""
    left = m.group(1).strip()
    # Keep only the trailing clause before separators
    left = re.split(r"[\.;\n\-–—]", left)[-1].strip()
    # Remove generic openers
    left = re.sub(r"^(Các|Những|Quy\s*tắc|Quy\s*định|Nội\s*dung)\s+", "", left, flags=re.IGNORECASE)
    return _clean_term(left, max_words=16)


def guess_question_from_chunk(chunk_text: str, title: str, index: int) -> str:
    """Guess a concrete, non-generic question that matches the chunk content.

    Priority:
    1) Định nghĩa: "X là ..." → "X là gì?"
    2) Tình huống cụ thể (đèn, cao tốc, đường sắt, hầm, phà,...)
    3) Quy tắc/chủ đề lớn (tốc độ, vượt, nhường đường, dừng đỗ,...)
    4) Nhóm "bao gồm" (liệt kê) / hành vi bị cấm
    5) Nhóm biển báo, fallback tóm tắt.
    """
    text = chunk_text.strip()

    # 0. Specific signage by code
    codes = _extract_sign_codes(text)
    if codes:
        if "," in codes or " và " in codes:
            return f"Ý nghĩa và phạm vi hiệu lực của các biển {codes} là gì?"
        return f"Ý nghĩa và phạm vi hiệu lực của biển {codes} là gì?"

    # 1. Definitions: "X là ..." (ưu tiên cao nhất)
    m = TERM_DEF_RE.match(text)
    if m:
        term = _clean_term(m.group(1), max_words=10)
        if term:
            return f"{term} là gì?"

    # 1.1 Traffic lights specific questions (check before nhường đường)
    if DEN_VANG_NN_RE.search(text):
        return "Đèn vàng nhấp nháy có ý nghĩa gì và xử lý như thế nào?"
    if DEN_DO_NN_RE.search(text):
        return "Khi đèn đỏ nhấp nháy, người tham gia giao thông phải làm gì?"
    if DEN_PHU_RE.search(text):
        return "Đèn phụ hình mũi tên có ý nghĩa gì?"
    if DEN_HAI_MAU_RE.search(text):
        return "Đèn tín hiệu hai màu dùng trong trường hợp nào và ý nghĩa ra sao?"
    if DEN_MOT_MAU_RE.search(text):
        return "Đèn tín hiệu một màu dùng khi nào và cần lưu ý gì?"

    # 2. Các quy tắc/chủ đề lớn – nhưng cố gắng match sát nội dung hơn
    # 2.1 Dừng/đỗ xe
    if DUNG_DO_RE.search(text):
        return "Quy định dừng và đỗ xe như thế nào và các vị trí cấm dừng/đỗ là gì?"

    # 2.2 Vượt xe
    if VUOT_RE.search(text):
        # Phân biệt điều kiện, cách vượt, hoặc cấm vượt
        if "cấm vượt" in text.lower():
            return "Những trường hợp nào bị cấm vượt xe?"
        if "điều kiện" in text.lower():
            return "Điều kiện xin vượt xe an toàn là gì?"
        if "vượt xe ở đường" in text.lower() or "vượt bên phải" in text.lower():
            return "Cách vượt xe an toàn trong các loại đường khác nhau là gì?"
        return "Điều kiện xin vượt, cách vượt an toàn và các trường hợp cấm vượt là gì?"

    # 2.3 Nhường đường / đi chậm, thận trọng
    if NHUONG_DUONG_RE.search(text) and not DUONG_SAT_RE.search(text):
        # Nếu có nhiều cụm "nơi dành cho người đi bộ", "tầm nhìn bị hạn chế",... → đó là
        # các trường hợp cần giảm tốc/đi thận trọng chứ không thuần nhường đường.
        if any(kw in text.lower() for kw in ["nơi dành cho người đi bộ", "tầm nhìn bị hạn chế", "đường cong", "đèo, dốc", "khu vực đông dân cư", "tránh xe đi ngược chiều"]):
            return "Những trường hợp nào người lái xe phải giảm tốc độ và đi thận trọng?"
        if "vòng xuyến" in text.lower():
            return "Quy tắc nhường đường tại vòng xuyến được quy định như thế nào?"
        if "đường giao nhau" in text.lower() or "giao lộ" in text.lower():
            return "Quy tắc nhường đường tại nơi đường giao nhau là gì?"
        return "Quy tắc nhường đường tại nơi giao nhau/vòng xuyến là gì?"

    # 2.4 Đường sắt
    if DUONG_SAT_RE.search(text):
        if "đèn đỏ nhấp nháy" in text.lower() or "chuông kêu" in text.lower():
            return "Khi qua đường sắt có tín hiệu đèn đỏ nhấp nháy hoặc chuông kêu, người lái xe phải làm gì?"
        return "Quy định qua đường sắt có/không rào chắn là gì?"

    # 2.5 Phà / cầu phao
    if PHA_RE.search(text):
        if "thứ tự ưu tiên" in text.lower():
            return "Thứ tự ưu tiên qua phà và cầu phao được quy định như thế nào?"
        if "xuống phà" in text.lower() or "lên bến" in text.lower():
            return "Quy định khi xuống phà, ở trên phà và lên bến như thế nào?"
        return "Quy định, thứ tự ưu tiên khi qua phà/cầu phao là gì?"

    # 2.6 Cao tốc
    if CAO_TOC_RE.search(text):
        if "nhập làn" in text.lower():
            return "Quy tắc nhập làn vào đường cao tốc được quy định như thế nào?"
        if "ra khỏi đường cao tốc" in text.lower():
            return "Khi rời khỏi đường cao tốc, người lái xe phải thực hiện những bước nào?"
        if "dừng, đỗ" in text.lower() or "làn dừng khẩn cấp" in text.lower():
            return "Khi dừng, đỗ xe khẩn cấp trên đường cao tốc cần tuân thủ những quy định gì?"
        if "không được đi vào cao tốc" in text.lower() or "không được đi vào" in text.lower():
            return "Những loại phương tiện và đối tượng nào không được phép đi vào đường cao tốc?"
        return "Quy tắc nhập làn, di chuyển, dừng khẩn cấp và rời khỏi đường cao tốc là gì?"

    # 2.7 Hầm đường bộ
    if HAM_RE.search(text):
        return "Quy định khi đi qua hầm đường bộ và xử lý sự cố là gì?"

    # 2.8 Cửa xe, đèn, còi
    if CUA_DEN_COI_RE.search(text):
        if "mở cửa" in text.lower():
            return "Khi mở cửa xe cần tuân thủ những quy định nào để bảo đảm an toàn?"
        if "bật đèn chiếu sáng" in text.lower():
            return "Quy định sử dụng đèn chiếu sáng trên xe như thế nào?"
        if "còi" in text.lower():
            return "Quy định sử dụng còi xe trong giao thông đường bộ ra sao?"
        return "Quy định mở cửa xe, sử dụng đèn và còi như thế nào?"

    # 2.9 Xe ưu tiên
    if XE_UU_TIEN_RE.search(text):
        if "thứ tự ưu tiên" in text.lower():
            return "Thứ tự ưu tiên của các loại xe ưu tiên được quy định như thế nào?"
        if "tín hiệu ưu tiên" in text.lower() or "đèn nhấp nháy" in text.lower():
            return "Tín hiệu ưu tiên và màu sắc của từng loại xe ưu tiên là gì?"
        if "quyền đi lại" in text.lower() or "không bị hạn chế tốc độ" in text.lower():
            return "Quyền đi lại của xe ưu tiên được quy định như thế nào?"
        if "nhường đường" in text.lower():
            return "Người tham gia giao thông có trách nhiệm nhường đường cho xe ưu tiên như thế nào?"
        return "Thứ tự ưu tiên, tín hiệu và quyền/ trách nhiệm liên quan đến xe ưu tiên là gì?"

    # 2.10 Tốc độ, cự ly tối thiểu, khu đông dân cư
    if TOC_DO_RE.search(text):
        # Nếu nội dung nói rõ về "tuân thủ tốc độ, khoảng cách" → hỏi đúng ý đó
        if "khoảng cách an toàn" in text.lower():
            return "Người lái xe phải tuân thủ tốc độ và khoảng cách an toàn như thế nào?"
        if "tốc độ lái xe phải phù hợp" in text.lower():
            return "Tốc độ lái xe phải phù hợp với những điều kiện nào trên đường?"
        if "xe có tốc độ thấp hơn" in text.lower():
            return "Khi có sự chênh lệch tốc độ, xe có tốc độ thấp hơn phải đi như thế nào?"
        return "Quy định về tốc độ, cự ly tối thiểu và khu đông dân cư là gì?"

    # 2.11 Kéo xe, rơ moóc, sơ mi rơ moóc
    if KEO_ROMOOC_RE.search(text):
        if "không được chở người trên xe được kéo" in text.lower():
            return "Xe được kéo có được chở người hay không và ngoại lệ nào được phép?"
        if "không được kéo thêm" in text.lower():
            return "Xe kéo rơ moóc, sơ mi rơ moóc bị cấm kéo thêm những gì?"
        if "rơ moóc là phương tiện" in text.lower():
            return "Rơ moóc là gì?"
        if "sơ mi rơ moóc là phương tiện" in text.lower():
            return "Sơ mi rơ moóc là gì?"
        return "Quy định kéo xe, kéo rơ moóc và sơ mi rơ moóc như thế nào?"

    # 2.12 Giấy phép lái xe & điểm
    if GPLX_DIEM_RE.search(text):
        return "Cơ chế điểm của giấy phép lái xe (12 điểm), cách trừ và phục hồi điểm hoạt động như thế nào?"
    if GPLX_RE.search(text):
        if "thời hạn" in text.lower() or "hiệu lực" in text.lower():
            return "Các hạng giấy phép lái xe, thời hạn và hiệu lực tương ứng là gì?"
        return "Các hạng giấy phép lái xe được phân loại như thế nào?"

    # 2.13 Độ tuổi, sức khỏe
    if DO_TUOI_SUC_KHOE_RE.search(text):
        if "đủ 16 tuổi" in text.lower() or "đủ 18 tuổi" in text.lower():
            return "Quy định về độ tuổi tối thiểu, tối đa đối với người lái xe là gì?"
        return "Quy định về độ tuổi và sức khỏe đối với người lái xe là gì?"

    # 2.14 Đào tạo, sát hạch
    if SAT_HACH_RE.search(text):
        if "sát hạch lái xe mô phỏng" in text.lower():
            return "Sát hạch lái xe mô phỏng là gì và được thực hiện như thế nào?"
        if "sát hạch thực hành" in text.lower() and "trong hình" in text.lower():
            return "Sát hạch thực hành lái xe ô tô trong hình được tổ chức ra sao?"
        if "sát hạch lý thuyết" in text.lower():
            return "Nội dung sát hạch lý thuyết lái xe ô tô bao gồm những gì?"
        return "Điều kiện và nội dung sát hạch/đào tạo lái xe ô tô gồm những gì?"

    # 2.15 Làn đường nói chung
    if LAN_DUONG_RE.search(text):
        if "xe thô sơ" in text.lower():
            return "Xe thô sơ phải đi ở làn đường nào và có được đi vào làn dành riêng không?"
        return "Quy định sử dụng làn đường và làn dành riêng cho từng loại phương tiện là gì?"

    # 2.16 Vạch kẻ đường
    if VACH_KE_DUONG_RE.search(text):
        if "sử dụng độc lập" in text.lower():
            return "Khi vạch kẻ đường được sử dụng độc lập, người tham gia giao thông phải tuân thủ như thế nào?"
        if "kết hợp với đèn tín hiệu" in text.lower() or "biển báo" in text.lower():
            return "Thứ tự ưu tiên hiệu lực khi vạch kẻ đường kết hợp với đèn tín hiệu và biển báo được quy định ra sao?"
        if "quy chuẩn quốc gia" in text.lower() and "G1" in text:
            return "Theo Quy chuẩn quốc gia về báo hiệu đường bộ, các nhóm vạch kẻ đường G1, G2, G3, G4 được phân loại như thế nào?"
        return "Vạch kẻ đường có ý nghĩa và hiệu lực như thế nào trong tổ chức giao thông?"

    # 3. Groups with 'bao gồm/gồm' (sau khi xử lý các chủ đề cụ thể)
    if BAO_GOM_RE.search(text):
        if CAM_RE.search(text):
            return "Những hành vi bị nghiêm cấm trong trường hợp này bao gồm những gì?"
        subj = _subject_before_baogom(text)
        if subj:
            return f"{subj} bao gồm những nội dung gì?"
        if title:
            return f"{title} bao gồm những nội dung gì?"
        return "Danh mục các nội dung/đối tượng bao gồm những gì?"

    # 4. Signage category introductions
    if NHOM_BIEN_RE.search(text):
        return "Đặc điểm nhận biết, phạm vi hiệu lực và mục đích sử dụng của nhóm biển báo này là gì?"

    # 5. Fallback: nếu chunk mở đầu bằng nhãn có dấu ':' → hỏi cụ thể theo nhãn đó
    m2 = ITEM_COLON_RE.match(text)
    if m2:
        item = _clean_term(m2.group(1), max_words=10)
        if item:
            return f"{item} được quy định như thế nào?"

    return "Tóm tắt các quy định chính của đoạn sau."

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
    lead = re.sub(r"^\s*(Khi|Trước\s+khi|Sau\s+khi|Đối\s+với|Với|Nếu|Tại|Trong)\s+", "", lead, flags=re.IGNORECASE)
    # Stop at first comma to keep qualifier short
    lead = lead.split(",")[0]
    max_len = 48
    if len(lead) > max_len:
        cut = lead[:max_len]
        last_space = cut.rfind(" ")
        if last_space > 12:
            cut = cut[:last_space]
        lead = cut
    qualifier = lead.strip(' "“”\'’:;,.()[]')
    if qualifier:
        return f"{q} — {qualifier}"
    return f"{q} (phần {idx + 1})"


def emit_examples(obj: Dict) -> Iterator[Dict]:
    title = obj.get("title", "") or ""
    chunks = obj.get("chunks", []) or []
    seen_questions: Dict[str, int] = {}
    for i, ch in enumerate(chunks):
        text = normalize_text(ch.get("text", ""))
        if not text:
            continue
        question = guess_question_from_chunk(text, title, i)
        if question in seen_questions:
            count = seen_questions[question] + 1
            specialized = _specialize_question_for_duplicate(question, text, count - 1)
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
    ap = argparse.ArgumentParser(description="Convert PLGTDB lessons JSON -> prompt/response JSONL.")
    ap.add_argument("--input", required=True, help="Path to input PLGTDB.json or folder")
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

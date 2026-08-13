#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert DD (Đạo đức / Văn hoá giao thông / An toàn / Sơ cứu / Cháy nổ / Rượu bia) lesson JSON
into JSONL fine-tune samples: {"prompt": <str>, "response": <str>}.

Heuristics tailored for the DD domain to create meaningful Vietnamese Q&A pairs.
Each chunk becomes one example.

Usage:
    python convert_lessons_dd.py --input data/lesson/DD.json --output dataset/lesson/DD_converted.jsonl
"""
import argparse
import json
import os
import re
import sys
from typing import Dict, Iterable, Iterator, List, Union

# =============== Regex patterns (Vietnamese domain) ===============
# Định nghĩa chung: "Khái niệm X là ..." or "Đạo đức nghề nghiệp là ..."
TERM_DEF_RE = re.compile(r"^\s*([A-ZÀ-Ỹa-zà-ỹ0-9][^.:;]{1,100}?)\s+là\s+", flags=re.UNICODE)
# Mục dạng nhãn kèm dấu hai chấm: "Vô lăng:" "Phanh tay:" "Khóa điện:" ...
ITEM_COLON_RE = re.compile(r"^\s*([A-ZÀ-Ỹa-zà-ỹ0-9][^:\n]{1,120}):\s+", flags=re.UNICODE)
# Vai trò của <X>
VAI_TRO_RE = re.compile(r"^\s*Vai\s*trò\s*của\s+([^,.;:]{1,80})", flags=re.IGNORECASE | re.UNICODE)
# Các yếu tố cấu thành <X> bao gồm
YEU_TO_CAU_THANH_RE = re.compile(r"^\s*Các\s+yếu\s+tố\s+cấu\s+thành\s+([^,.;:]{1,80})\s+bao\s+gồm", flags=re.IGNORECASE | re.UNICODE)
# Quan niệm về <X>
QUAN_NIEM_RE = re.compile(r"^\s*Quan\s+niệm\s+về\s+([^,.;:]{1,80})", flags=re.IGNORECASE | re.UNICODE)
# Tác hại của <A> đối với <B>
TAC_HAI_RE = re.compile(r"^\s*Tác\s+hại\s+của\s+([^,.;:]{1,80})\s+đối\s+với\s+([^,.;:]{1,80})", flags=re.IGNORECASE | re.UNICODE)
# Tác hại của rượu, bia khi lái xe
TAC_HAI_KHI_LAI_XE_RE = re.compile(r"^\s*Tác\s+hại\s+của\s+rượu,?\s*bia\s+khi\s+lái\s+xe", flags=re.IGNORECASE | re.UNICODE)
# Các mức nồng độ cồn ...
NONG_DO_CON_LEVELS_RE = re.compile(r"^\s*Các\s+mức\s+nồng\s+độ\s+cồn\s+và\s+ảnh\s+hưởng", flags=re.IGNORECASE | re.UNICODE)
# nồng độ cồn 0,08%
NONG_DO_CON_080_RE = re.compile(r"nồng\s+độ\s+cồn\s*0,?0?8%", flags=re.IGNORECASE | re.UNICODE)
# 8 tiêu chí đạo đức nghề nghiệp ...
TIEU_CHI_8_RE = re.compile(r"^\s*8\s+tiêu\s+chí\s+đạo\s+đức\s+nghề\s+nghiệp\s+của\s+([^,.;:]{1,80})\s+bao\s+gồm", flags=re.IGNORECASE | re.UNICODE)
# Xử lý tai nạn giao thông đường bộ bao gồm
XU_LY_TAI_NAN_RE = re.compile(r"^\s*Xử\s+lý\s+tai\s+nạn\s+giao\s+thông\s+đường\s+bộ\s+bao\s+gồm", flags=re.IGNORECASE | re.UNICODE)
# Sơ cứu / Sơ cấp cứu người bị ...
SO_CUU_RE = re.compile(r"^\s*Sơ\s*(?:cấp\s*cứu|cứu)\s+người\s+bị\s+([^,.;:]{1,80})", flags=re.IGNORECASE | re.UNICODE)
# Cầm máu trực tiếp / Cầm máu không trực tiếp / Cầm máu bằng garo
CAM_MAU_RE = re.compile(r"^\s*Cầm\s+máu\s+(trực\s+tiếp|không\s+trực\s+tiếp|bằng\s+garo)", flags=re.IGNORECASE | re.UNICODE)
# Kỹ thuật an toàn chung / Kỹ thuật chữa cháy
KY_THUAT_RE = re.compile(r"^\s*Kỹ\s+thuật\s+(?:an\s+toàn\s+chung|chữa\s+cháy)", flags=re.IGNORECASE | re.UNICODE)
# Nguyên nhân của cháy, nổ đối với xe ô tô sử dụng nhiên liệu hóa thạch
NGUYEN_NHAN_FOSSIL_RE = re.compile(r"^\s*Nguyên\s+nhân\s+của\s+cháy,?\s*nổ\s+đối\s+với\s+xe\s+ô\s+tô\s+sử\s+dụng\s+nhiên\s+liệu\s+hóa\s+thạch", flags=re.IGNORECASE | re.UNICODE)
# Nguyên nhân của cháy, nổ đối với xe ô tô điện
NGUYEN_NHAN_ELECTRIC_RE = re.compile(r"^\s*Nguyên\s+nhân\s+của\s+cháy,?\s*nổ\s+đối\s+với\s+xe\s+ô\s+tô\s+điện", flags=re.IGNORECASE | re.UNICODE)
# Biện pháp phòng cháy, nổ dành cho hành khách
PHONG_CHAY_HANH_KHACH_RE = re.compile(r"^\s*Biện\s+pháp\s+phòng\s+cháy,?\s*nổ\s+dành\s+cho\s+hành\s+khách", flags=re.IGNORECASE | re.UNICODE)
# Khi cháy xe đối với hành khách
KHI_CHAY_XE_HANH_KHACH_RE = re.compile(r"^\s*Khi\s+cháy\s+xe\s+đối\s+với\s+hành\s+khách", flags=re.IGNORECASE | re.UNICODE)
# Các lưu ý quan trọng
LUU_Y_QUAN_TRONG_RE = re.compile(r"^\s*Các\s+lưu\s+ý\s+quan\s+trọng", flags=re.IGNORECASE | re.UNICODE)
# Những điểm cơ bản giúp bạn lái xe an toàn
DIEM_CO_BAN_RE = re.compile(r"^\s*Những\s+điểm\s+cơ\s+bản\s+giúp\s+bạn\s+lái\s+xe\s+an\s+toàn", flags=re.IGNORECASE | re.UNICODE)
# Thực hiện quy tắc chung khi tham gia giao thông đường bộ
QUY_TAC_CHUNG_RE = re.compile(r"^\s*Thực\s+hiện\s+quy\s+tắc\s+chung\s+khi\s+tham\s+gia\s+giao\s+thông\s+đường\s+bộ", flags=re.IGNORECASE | re.UNICODE)
# Trách nhiệm cao nhất với bản thân và cộng đồng
TRACH_NHIEM_CAO_NHAT_RE = re.compile(r"^\s*Trách\s+nhiệm\s+cao\s+nhất\s+với\s+bản\s+thân\s+và\s+cộng\s+đồng", flags=re.IGNORECASE | re.UNICODE)
# Xử phạt vi phạm ở (nồng độ cồn)
XU_PHAT_RUOU_BIA_RE = re.compile(r"^\s*Xử\s+phạt\s+vi\s+phạm\s+ở", flags=re.IGNORECASE | re.UNICODE)
# Quy định pháp luật dựa vào
QUY_DINH_PHAP_LUAT_RE = re.compile(r"^\s*Quy\s+định\s+pháp\s+luật\s+dựa\s+vào", flags=re.IGNORECASE | re.UNICODE)
# Tăng cường tuần tra, kiểm soát
TANG_CUONG_TUAN_TRA_RE = re.compile(r"^\s*Tăng\s+cường\s+tuần\s+tra,?\s+kiểm\s+soát", flags=re.IGNORECASE | re.UNICODE)
# Hoạt động tuyên truyền, giáo dục
TUYEN_TRUYEN_GD_RE = re.compile(r"^\s*Hoạt\s+động\s+tuyên\s+truyền,?\s+giáo\s+dục", flags=re.IGNORECASE | re.UNICODE)
# Tình huống tổng hợp: lái xe ẩu / cắt ngang / đèn đỏ / sự cố / tai nạn / giúp đỡ
TONG_HOP_TINH_HUONG_RE = re.compile(r"^\s*Khi\s+gặp\s+người\s+lái\s+xe\s+ẩu", flags=re.IGNORECASE | re.UNICODE)
# Chương trình tuyên truyền 'Đã uống rượu bia – Không lái xe'
PROMO_PROGRAMS_RE = re.compile(r"^\s*các\s+chương\s+trình\s+như\s+“?Đã\s+uống\s+rượu\s+bia", flags=re.IGNORECASE | re.UNICODE)
ANH_HUONG_VHGT_RE = re.compile(r"^\s*Văn\s+hóa\s+giao\s+thông\s+ảnh\s+hưởng", flags=re.IGNORECASE | re.UNICODE)
TAC_HAI_RUOU_BIA_DOI_VOI_RE = re.compile(r"^\s*Tác\s+hại\s+của\s+rượu,?\s*bia\s+đối\s+với\s+([^,.;:]{1,80})", flags=re.IGNORECASE | re.UNICODE)

# ===== New fine-grained patterns (first-aid & safety refinements) =====
NHAN_BIET_KHONG_HO_HAP_RE = re.compile(r"^\s*nhận\s+biết\s+tình\s+trạng\s+không\s+hô\s+hấp", flags=re.IGNORECASE | re.UNICODE)
KHONG_CO_DAU_HIEU_THO_RE = re.compile(r"^\s*nếu\s+không\s+có\s+dấu\s+hiệu\s+thở", flags=re.IGNORECASE | re.UNICODE)
HO_HAP_NHAN_TAO_RE = re.compile(r"^\s*Thực\s+hiện\s+hô\s+hấp\s+nhân\s+tạo", flags=re.IGNORECASE | re.UNICODE)
HOI_SUC_TIM_PHOI_RE = re.compile(r"^\s*Hồi\s+sức\s+tim\s+phổi", flags=re.IGNORECASE | re.UNICODE)
DAT_NAN_NHAN_TU_THE_RE = re.compile(r"^\s*Cách\s+đặt\s+nạn\s+nhân\s+đúng\s+tư\s+thế", flags=re.IGNORECASE | re.UNICODE)
GAY_XUONG_DAU_HIEU_RE = re.compile(r"^\s*Gãy\s+xương\s+có\s+dấu\s+hiệu", flags=re.IGNORECASE | re.UNICODE)
CHAN_THUONG_SO_NAO_RE = re.compile(r"^\s*Chấn\s+thương\s+sọ\s+não", flags=re.IGNORECASE | re.UNICODE)
CO_GIAT_RE = re.compile(r"^\s*Co\s+giật", flags=re.IGNORECASE | re.UNICODE)
BONG_GAN_TRAT_KHOP_RE = re.compile(r"^\s*Bong\s+gân,?\s+trật\s+khớp", flags=re.IGNORECASE | re.UNICODE)
XU_LY_KHI_TAI_NAN_GT_RE = re.compile(r"^\s*Xử\s+lý\s+khi\s+gặp\s+tai\s+nạn\s+giao\s+thông", flags=re.IGNORECASE | re.UNICODE)
CHUAN_BI_TUI_CAP_CUU_RE = re.compile(r"^\s*Chuẩn\s+bị\s+túi\s+cấp\s+cứu", flags=re.IGNORECASE | re.UNICODE)
KY_THUAT_CHUA_CHAY_FOSSIL_RE = re.compile(r"^\s*Kỹ\s+thuật\s+chữa\s+cháy\s+Đối\s+với\s+người\s+lái\s+xe\s+dùng\s+nhiên\s+liệu\s+hóa\s+thạch", flags=re.IGNORECASE | re.UNICODE)
KY_THUAT_CHUA_CHAY_ELECTRIC_RE = re.compile(r"^\s*Kỹ\s+thuật\s+chữa\s+cháy\s+đối\s+với\s+người\s+lái\s+xe\s+điện", flags=re.IGNORECASE | re.UNICODE)
SO_CUU_BI_BONG_RE = re.compile(r"^\s*Sơ\s*cứu\s+người\s+bị\s+bỏng", flags=re.IGNORECASE | re.UNICODE)
SO_CUU_HIT_KHOI_RE = re.compile(r"^\s*Sơ\s*cứu\s+người\s+hít\s+phải\s+khói", flags=re.IGNORECASE | re.UNICODE)

# Bổ sung: Văn hóa giao thông là
VAN_HOA_GT_LA_RE = re.compile(r"^\s*Văn\s+hóa\s+giao\s+thông\s+là", flags=re.IGNORECASE | re.UNICODE)
# Bổ sung: Đạo đức là phương thức điều chỉnh hành vi con người
DAO_DUC_PHUONG_THUC_RE = re.compile(r"^\s*Đạo\s+đức\s+là\s+phương\s+thức\s+điều\s+chỉnh\s+hành\s+vi\s+con\s+người", flags=re.IGNORECASE | re.UNICODE)
# Bổ sung: Đạo đức là một hệ thống các giá trị
DAO_DUC_HE_GIA_TRI_RE = re.compile(r"^\s*Đạo\s+đức\s+là\s+một\s+hệ\s+thống\s+các\s+giá\s+trị", flags=re.IGNORECASE | re.UNICODE)
# Bổ sung: Nền đạo đức mới ở nước ta hiện nay
NEN_DAO_DUC_MOI_RE = re.compile(r"^\s*Nền\s+đạo\s+đức\s+mới\s+ở\s+nước\s+ta\s+hiện\s+nay", flags=re.IGNORECASE | re.UNICODE)
# Bổ sung: Lái xe ô tô là ... (đặc điểm chung)
LAI_XE_LA_RE = re.compile(r"^\s*Lái\s+xe\s+ô\s+tô\s+là", flags=re.IGNORECASE | re.UNICODE)

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

def _clean_term(term: str, max_words: int = 10) -> str:
    term = term.strip(' "“”\'’:;,.()[]')
    words = term.split()
    if len(words) == 0 or len(words) > max_words:
        return ""
    BAD_TOKENS = {"gồm", "bao", "bao_gồm", "được", "trong", "với", "khi", "ở", "do", "đến", "của"}
    low = term.lower()
    if any(tok in low for tok in BAD_TOKENS) and not re.match(r"^(đạo\s+đức|văn\s+hóa|tai\s+nạn|cháy|nổ|sơ\s+cấp\s+cứu|rượu|bia|nồng\s+độ\s+cồn)", low):
        return ""
    if term and term[0].islower():
        term = term[0].upper() + term[1:]
    return term

# =============== Core heuristic ===============

def guess_question_from_chunk(chunk_text: str, title: str, index: int) -> str:
    # 1. Direct patterns (ordered by specificity)
    # Nhãn thiết bị/chức năng dạng "Tên mục: ..."
    m = ITEM_COLON_RE.match(chunk_text)
    if m:
        term = _clean_term(m.group(1))
        if term:
            # Nếu là cụm bắt đầu bằng "Điều chỉnh" hoặc động từ, đặt câu hỏi phù hợp hành động.
            low = term.lower()
            if low.startswith("điều chỉnh"):
                return f"Khi {term.lower()} cần lưu ý điều gì?"
            if any(low.startswith(x) for x in ["cài", "thao tác", "lên xe", "xuống xe"]):
                return f"{term} thực hiện như thế nào?"
            # Mặc định: hỏi về chức năng/công dụng để khớp với câu trả lời thường bắt đầu bằng "Dùng để..."
            return f"“{term}” có chức năng gì?"

    # Văn hóa giao thông định nghĩa rõ ràng
    if VAN_HOA_GT_LA_RE.match(chunk_text):
        return "Văn hóa giao thông là gì?"

    # New first-aid & safety specialized patterns
    if DAT_NAN_NHAN_TU_THE_RE.match(chunk_text):
        return "Đặt nạn nhân đúng tư thế sơ cứu như thế nào?"
    if NHAN_BIET_KHONG_HO_HAP_RE.match(chunk_text):
        return "Làm thế nào nhận biết nạn nhân không còn hô hấp?"
    if KHONG_CO_DAU_HIEU_THO_RE.match(chunk_text):
        return "Cần làm gì khi nạn nhân không còn dấu hiệu thở?"
    if HO_HAP_NHAN_TAO_RE.match(chunk_text):
        return "Các bước thực hiện hô hấp nhân tạo là gì?"
    if HOI_SUC_TIM_PHOI_RE.match(chunk_text):
        return "Các bước hồi sức tim phổi (CPR) cơ bản là gì?"
    if GAY_XUONG_DAU_HIEU_RE.match(chunk_text):
        return "Dấu hiệu gãy xương và cách xử lý ban đầu là gì?"
    if CHAN_THUONG_SO_NAO_RE.match(chunk_text):
        return "Dấu hiệu chấn thương sọ não và xử lý ban đầu như thế nào?"
    if CO_GIAT_RE.match(chunk_text):
        return "Cách xử lý an toàn khi nạn nhân bị co giật là gì?"
    if BONG_GAN_TRAT_KHOP_RE.match(chunk_text):
        return "Dấu hiệu bong gân, trật khớp và cách xử lý an toàn là gì?"
    if XU_LY_KHI_TAI_NAN_GT_RE.match(chunk_text):
        return "Cách xử lý ngay khi vừa gặp tai nạn giao thông là gì?"
    if CHUAN_BI_TUI_CAP_CUU_RE.match(chunk_text):
        return "Túi cấp cứu cơ bản cần có những gì?"
    if KY_THUAT_CHUA_CHAY_FOSSIL_RE.match(chunk_text):
        return "Kỹ thuật chữa cháy dành cho xe dùng nhiên liệu hóa thạch là gì?"
    if KY_THUAT_CHUA_CHAY_ELECTRIC_RE.match(chunk_text):
        return "Kỹ thuật chữa cháy an toàn cho xe điện là gì?"
    if SO_CUU_BI_BONG_RE.match(chunk_text):
        return "Các bước sơ cứu người bị bỏng là gì?"
    if SO_CUU_HIT_KHOI_RE.match(chunk_text):
        return "Các bước sơ cứu người hít phải khói là gì?"

    m = VAI_TRO_RE.match(chunk_text)
    if m:
        obj_raw = m.group(1).strip()
        # Loại bỏ phần đuôi 'bao gồm' nếu bị bắt vào
        obj_raw = re.sub(r"\bbao\s+gồm.*$", "", obj_raw, flags=re.IGNORECASE)
        # Loại bỏ phần đuôi 'là ...' để chỉ giữ đối tượng
        obj_raw = re.sub(r"\s+là\s+.*$", "", obj_raw, flags=re.IGNORECASE)
        obj = _clean_term(obj_raw)
        if obj:
            return f"Vai trò của “{obj}” là gì?"

    m = YEU_TO_CAU_THANH_RE.match(chunk_text)
    if m:
        obj = _clean_term(m.group(1))
        if obj:
            return f"Các yếu tố cấu thành “{obj}” gồm những gì?"

    m = QUAN_NIEM_RE.match(chunk_text)
    if m:
        obj_raw = m.group(1).strip().strip(' "“”\'’:;,.()[]')
        # Nếu có ' là ' bên trong phần bắt, cắt trước ' là '
        obj_clean = re.split(r"\s+là\s+", obj_raw, 1)[0].strip()
        if obj_clean:
            return f"Quan niệm về “{obj_clean}” hiện nay là gì?"

    # Đạo đức là phương thức điều chỉnh hành vi
    if DAO_DUC_PHUONG_THUC_RE.match(chunk_text):
        return "Đạo đức điều chỉnh hành vi con người như thế nào?"

    # Đạo đức là hệ thống các giá trị
    if DAO_DUC_HE_GIA_TRI_RE.match(chunk_text):
        return "Đạo đức phản ánh những giá trị và thái độ nào trong xã hội?"

    # Nền đạo đức mới ở Việt Nam
    if NEN_DAO_DUC_MOI_RE.match(chunk_text):
        return "Nền đạo đức mới ở Việt Nam hiện nay có đặc điểm gì?"

    # Lái xe ô tô là hoạt động ... (đặc điểm)
    if LAI_XE_LA_RE.match(chunk_text):
        return "Đặc điểm của hoạt động lái xe ô tô là gì?"

    m = TAC_HAI_RE.match(chunk_text)
    if m:
        a = _clean_term(m.group(1), max_words=12)
        b = _clean_term(m.group(2), max_words=8)
        if a and b:
            return f"Tác hại của “{a}” đối với “{b}” là gì?"

    if TAC_HAI_KHI_LAI_XE_RE.match(chunk_text):
        return "Tác hại của rượu bia đối với khả năng lái xe là gì?"

    if NONG_DO_CON_LEVELS_RE.match(chunk_text):
        return "Các mức nồng độ cồn ảnh hưởng tới khả năng lái xe như thế nào?"

    if NONG_DO_CON_080_RE.search(chunk_text):
        return "Nồng độ cồn 0,08% ảnh hưởng gì tới khả năng kiểm soát xe?"

    m = TIEU_CHI_8_RE.match(chunk_text)
    if m:
        subj = _clean_term(m.group(1))
        if subj:
            return "Các tiêu chí đạo đức nghề nghiệp của người lái xe ô tô là gì?"

    if XU_LY_TAI_NAN_RE.match(chunk_text):
        return "Các bước xử lý tai nạn giao thông đường bộ là gì?"

    m = SO_CUU_RE.match(chunk_text)
    if m:
        cond_raw = m.group(1)
        # Trim trailing instruction words
        cond_trim = re.split(r"\b(bằng|nên|cần|để|bao\s+gồm)\b", cond_raw, 1)[0].strip()
        cond_trim = cond_trim.strip(' "“”\'’:;,.()[]')
        cond_low = cond_trim.lower()
        if cond_low in {"bỏng", "hít phải khói"}:
            # Already handled by specialized patterns above; skip to avoid duplicate
            pass
        else:
            cond = _clean_term(cond_trim, max_words=8)
            if cond:
                return f"Các bước sơ cứu người bị {cond.lower()} là gì?"

    m = CAM_MAU_RE.match(chunk_text)
    if m:
        kind = m.group(1).lower()
        return f"Cách cầm máu {kind} thực hiện như thế nào?"

    if KY_THUAT_RE.match(chunk_text):
        if "chữa cháy" in chunk_text.lower():
            return "Kỹ thuật chữa cháy cơ bản dành cho lái xe là gì?"
        return "Các kỹ thuật an toàn chung cần thực hiện là gì?"

    if NGUYEN_NHAN_FOSSIL_RE.match(chunk_text):
        return "Những nguyên nhân gây cháy nổ ở xe ô tô dùng nhiên liệu hóa thạch là gì?"

    if NGUYEN_NHAN_ELECTRIC_RE.match(chunk_text):
        return "Những nguyên nhân gây cháy nổ ở xe ô tô điện là gì?"

    if PHONG_CHAY_HANH_KHACH_RE.match(chunk_text):
        return "Biện pháp phòng cháy nổ dành cho hành khách là gì?"

    if KHI_CHAY_XE_HANH_KHACH_RE.match(chunk_text):
        return "Hành khách cần làm gì khi xe bị cháy?"

    if LUU_Y_QUAN_TRONG_RE.match(chunk_text):
        return "Các lưu ý quan trọng khi xử lý cháy xe là gì?"

    if DIEM_CO_BAN_RE.match(chunk_text):
        return "Những điểm cơ bản giúp lái xe an toàn là gì?"

    if QUY_TAC_CHUNG_RE.match(chunk_text):
        return "Các quy tắc chung khi tham gia giao thông đường bộ là gì?"

    if TRACH_NHIEM_CAO_NHAT_RE.match(chunk_text):
        return "Trách nhiệm cao nhất với bản thân và cộng đồng khi lái xe là gì?"

    if XU_PHAT_RUOU_BIA_RE.match(chunk_text):
        return "Các hình thức xử phạt vi phạm nồng độ cồn hiện nay là gì?"

    if QUY_DINH_PHAP_LUAT_RE.match(chunk_text):
        return "Những quy định pháp luật về xử phạt nồng độ cồn là gì?"

    if TANG_CUONG_TUAN_TRA_RE.match(chunk_text):
        return "Việc tăng cường tuần tra kiểm soát giúp giảm tai nạn giao thông như thế nào?"

    if TUYEN_TRUYEN_GD_RE.match(chunk_text):
        return "Hoạt động tuyên truyền giáo dục nâng cao ý thức về rượu bia gồm những gì?"

    if TONG_HOP_TINH_HUONG_RE.match(chunk_text):
        return "Cách ứng xử an toàn trong các tình huống giao thông phức tạp (lái xe ẩu, bị cắt ngang, tắc đường, sự cố, tai nạn, giúp đỡ) là gì?"

    if PROMO_PROGRAMS_RE.match(chunk_text):
        return "Mục tiêu của các chương trình 'Đã uống rượu bia – Không lái xe' và 'Uống có trách nhiệm' là gì?"
    if ANH_HUONG_VHGT_RE.match(chunk_text):
        return "Văn hóa giao thông ảnh hưởng đến đời sống xã hội như thế nào?"

    m = TAC_HAI_RUOU_BIA_DOI_VOI_RE.match(chunk_text)
    if m:
        target_raw = m.group(1).strip()
        # Cắt trước các từ nối mô tả ('thì', 'là', 'sẽ', 'gây') để giữ đối tượng gọn.
        target_clean = re.split(r"\b(thì|là|sẽ|gây)\b", target_raw, 1)[0].strip()
        # Loại bỏ dấu câu dư thừa
        target_clean = target_clean.strip(' "“”\'’:;,.()[]')
        if target_clean:
            return f"Tác hại của rượu bia đối với {target_clean.lower()} là gì?"

    # Định nghĩa chung (sau đặc thù để tránh mâu thuẫn)
    m = TERM_DEF_RE.match(chunk_text)
    if m:
        term = _clean_term(m.group(1))
        if term:
            # Nếu cụm bắt đầu bằng "Khái niệm" thì chuẩn hóa cách đặt câu hỏi.
            if term.lower().startswith("khái niệm"):
                rest = term[len("Khái niệm"):].strip()
                rest = rest if rest else term
                return f"Khái niệm{(' ' + rest) if rest else ''} là gì?".replace("  ", " ")
            return f"“{term}” là gì?"

    # =============== Fallback subject extraction ===============
    STOP_SUFFIXES = {"còn", "thì", "ra", "vì", "do", "bằng", "việc", "gây", "đối", "đến", "với"}
    DOMAIN_NOUNS = [
        "đạo đức", "văn hóa giao thông", "lái xe an toàn", "tai nạn giao thông", "rượu bia",
        "nồng độ cồn", "sơ cứu", "cháy nổ", "xe ô tô điện", "xe ô tô nhiên liệu hóa thạch",
        "cấp cứu", "cầm máu", "hô hấp nhân tạo", "CPR", "bỏng", "chảy máu", "ngất", "gãy xương"
    ]

    def _extract_subject(text: str) -> str:
        clause = re.split(r"[\n\.;:]\s*", text.strip())[0]
        clause = re.sub(r"^\s*(Các|Những|Một|Khi|Nếu|Trong|Đối\s+với)\s+", "", clause, flags=re.IGNORECASE)
        words = clause.split()
        candidate_words: List[str] = []
        for w in words:
            lw = w.strip(',"“”').lower()
            if lw in STOP_SUFFIXES:
                break
            candidate_words.append(w.strip(',"“”'))
            if len(candidate_words) >= 8:
                break
        candidate = " ".join(candidate_words)
        if not candidate:
            return ""
        last = candidate_words[-1].lower()
        if last in STOP_SUFFIXES:
            candidate = ""
        cleaned = _clean_term(candidate)
        if cleaned:
            return cleaned
        # domain noun fallback
        low = text.lower()
        best = ""
        for noun in DOMAIN_NOUNS:
            if noun in low and len(noun) > len(best):
                best = noun
        return _clean_term(best)

    subj = _extract_subject(chunk_text)
    if subj:
        return f"Giải thích “{subj}” là gì?"

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

def emit_examples(obj: Dict) -> Iterator[Dict]:
    title = obj.get("title", "")
    chunks = obj.get("chunks", []) or []
    for i, ch in enumerate(chunks):
        text = normalize_text(ch.get("text", ""))
        if not text:
            continue
        question = guess_question_from_chunk(text, title, i)
        prompt = f"### CÂU HỎI: {question}\n\n### TRẢ LỜI:\n"
        yield {"prompt": prompt, "response": text}

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
    ap = argparse.ArgumentParser(description="Convert DD lessons JSON -> prompt/response JSONL.")
    ap.add_argument("--input", required=True, help="Path to input DD.json or folder")
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

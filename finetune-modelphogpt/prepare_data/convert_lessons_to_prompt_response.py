#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert driving-theory lesson docs (subject/lesson/title/chunks[]) -> JSONL fine-tune
with schema: {"prompt": <str>, "response": <str>}.

- Accepts: a single JSON object, a JSON array of such objects, or a JSONL of objects.
- For each chunk in `chunks`, emits one training sample.
- Heuristics:
    * Nếu đoạn bắt đầu bằng "<Thuật_ngữ> là ..." -> sinh câu hỏi dạng "Định nghĩa "<Thuật_ngữ>" là gì?"
    * Ngược lại -> sinh câu hỏi tổng quát theo tiêu đề bài học.

Usage:

    python convert_lessons_to_prompt_response.py 
        --input /home/phgiakhanh/Desktop/code_new/data_final/data/lesson/CTSCTT.json 
        --output dataset/lesson/CTSCTT.jsonl

"""

import argparse
import json
import os
import re
import sys
from typing import Dict, Iterable, Iterator, List, Union

# Regex / Heuristics nhận dạng mẫu câu phổ biến trong bài học (tiếng Việt)
# 1) "<Thuật ngữ> là ..." (định nghĩa)
TERM_DEF_RE = re.compile(r"^\s*([A-ZÀ-Ỹa-zà-ỹ0-9][^.:;]{1,80}?)\s+là\s+", flags=re.UNICODE)
# 2) "Cấu tạo chung của <Đối tượng> gồm ..."
COMPOSITION_RE = re.compile(r"^\s*Cấu\s*tạo\s*chung\s*của\s+([^,.;:]{1,60})\s+gồm\b", flags=re.IGNORECASE | re.UNICODE)
# 3) "(Hệ thống|Thiết bị|Bộ phận) X có (công dụng|chức năng) ..."
USAGE_RE = re.compile(r"^\s*((?:Hệ\s*thống|Thiết\s*bị|Bộ\s*phận)\s+[^,.;:]{1,60}?)\s+có\s+(?:công\s*dụng|chức\s*năng)\b", flags=re.IGNORECASE | re.UNICODE)
# 4) "<Đối tượng> được phân loại ..."
CLASSIFY_RE = re.compile(r"^\s*([A-ZÀ-Ỹa-zà-ỹ0-9][^.:;]{1,80}?)\s+được\s+phân\s+loại\b", flags=re.IGNORECASE | re.UNICODE)
# 5) "<Đối tượng> hoạt động theo ..."
OPERATE_SEQ_RE = re.compile(r"^\s*([A-ZÀ-Ỹa-zà-ỹ0-9][^.:;]{1,80}?)\s+hoạt\s+động\s+theo\b", flags=re.IGNORECASE | re.UNICODE)
# 6) Câu mở đầu mang tính "Nguyên lý hoạt động ..."
PRINCIPLE_RE = re.compile(r"^\s*nguyên\s*lý\s*hoạt\s*động\b", flags=re.IGNORECASE | re.UNICODE)

# Bổ sung: một số mẫu câu giúp đa dạng hóa câu hỏi, giảm trùng lặp
# 7) "Ý nghĩa của <X> ..."
MEANING_RE = re.compile(r"^\s*Ý\s*nghĩa\s*của\s+([^,.;:]{1,80})", flags=re.IGNORECASE | re.UNICODE)
# 8) "Cấu tạo|Cấu trúc (của) <X> gồm ..." (tổng quát, không chỉ "Cấu tạo chung")
# avoid capturing the word "chung" as part of the captured object
COMPOSITION2_RE = re.compile(
    r"^\s*(?:Cấu\s*tạo|Cấu\s*trúc)\s*(?:chung\s*)?(?:của\s+)?([^,.;:]{1,60})\s+gồm\b",
    flags=re.IGNORECASE | re.UNICODE,
)
# 9) "Ở hành trình ..." hoặc "Ở cuối kỳ/kì ..."
STAGE_RE = re.compile(r"^\s*Ở\s+((?:hành\s*trình\s*[^,.;:]{1,40})|(?:cuối\s+k[ỳy]\s*[^,.;:]{0,20}))", flags=re.IGNORECASE | re.UNICODE)
# 10) Nhận biết cụm "vòi phun điện tử"
INJECT_EFI_RE = re.compile(r"vòi\s*phun\s+điện\s*tử", flags=re.IGNORECASE | re.UNICODE)
# 11) Nhận biết "động cơ diesel"
DIESEL_RE = re.compile(r"động\s*cơ\s*diesel", flags=re.IGNORECASE | re.UNICODE)
# 12) Thông số trên hông lốp
SIDEWALL_TIRE_RE = re.compile(r"bên\s+hông\s+lốp", flags=re.IGNORECASE | re.UNICODE)
# 13) Vai trò/chức năng khi câu mở đầu bằng danh từ + (biến|giúp|đóng vai trò)
ROLE_RE = re.compile(r"^\s*([A-ZÀ-Ỹa-zà-ỹ0-9][^,.;:]{1,40}?)\s+(?:biến|giúp|đóng\s+vai\s+trò)\b", flags=re.IGNORECASE | re.UNICODE)
# 14) "<Chủ thể> có hai loại chính"
TWO_MAIN_TYPES_RE = re.compile(r"^\s*([A-ZÀ-Ỹa-zà-ỹ0-9][^,.;:]{1,80}?)\s+có\s+hai\s+loại\s+chính\b", flags=re.IGNORECASE | re.UNICODE)
# 15) Các thực thể điện phổ biến (Ắc quy/Máy phát điện/Bộ điều chỉnh điện/Hệ thống đánh lửa/Máy khởi động)
ELECTRIC_SUBJECT_RE = re.compile(r"^\s*((?:Ắc\s*quy|Máy\s*phát\s*điện|Bộ\s*điều\s*chỉnh\s*điện|Hệ\s*thống\s*đánh\s*lửa|Máy\s*khởi\s*động))\b", flags=re.IGNORECASE | re.UNICODE)
# 16) Mô tả sự cố: "<Hiện tượng> có thể/ thường ..."
ISSUE_RE = re.compile(r"^\s*([^,.;:]{3,50}?)\s+(?:có\s*thể|thường)\b", flags=re.IGNORECASE | re.UNICODE)
# 17) Các dạng bố trí hệ thống truyền lực
DRIVETRAIN_LAYOUTS_RE = re.compile(r"^\s*Các\s+dạng\s+bố\s+trí\s+hệ\s+thống\s+truyền\s+lực\b", flags=re.IGNORECASE | re.UNICODE)
# 18) Cụm cầu chủ động bao gồm ...
ACTIVE_AXLE_RE = re.compile(r"^\s*Cụm\s+cầu\s+chủ\s+động\s+bao\s+gồm\b", flags=re.IGNORECASE | re.UNICODE)
# 19) Vỏ cầu chủ động ...
AXLE_HOUSING_ROLE_RE = re.compile(r"^\s*Vỏ\s+cầu\s+chủ\s+động\b", flags=re.IGNORECASE | re.UNICODE)
# 20) Truyền lực chính ...
FINAL_DRIVE_ROLE_RE = re.compile(r"^\s*Truyền\s+lực\s+chính\b", flags=re.IGNORECASE | re.UNICODE)
# 21) Vi sai dùng để ...
DIFFERENTIAL_ROLE_RE = re.compile(r"^\s*Vi\s+sai\s+dùng\s+để\b", flags=re.IGNORECASE | re.UNICODE)
# 22) Hai bán trục ...
HALF_SHAFT_ROLE_RE = re.compile(r"^\s*Hai\s+bán\s+trục\b", flags=re.IGNORECASE | re.UNICODE)
# 23) Ly hợp chức năng ...
CLUTCH_FUNCTION_RE = re.compile(r"^\s*Ly\s+hợp\s+chức\s+năng\b", flags=re.IGNORECASE | re.UNICODE)
# 24) Hộp số thay đổi mô men ...
GEARBOX_FUNCTION_RE = re.compile(r"^\s*Hộp\s+số\s+thay\s+đổi\s+mô\s+men\b", flags=re.IGNORECASE | re.UNICODE)
# 25) Hộp số cơ khí 5 cấp tiến 1 cấp lùi thường gồm ...
GEARBOX_5_RE = re.compile(r"^\s*Hộp\s+số\s+cơ\s*khí\s+5\s+cấp\s+tiến\s+1\s+cấp\s+lùi\s+thường\s+gồm\b", flags=re.IGNORECASE | re.UNICODE)
# 26) Hộp số cơ khí Việc gài số ...
GEARBOX_SHIFT_RE = re.compile(r"^\s*Hộp\s+số\s+cơ\s*khí\s+Việc\s+gài\s+số\b", flags=re.IGNORECASE | re.UNICODE)
# 27) Trên ôtô hiện nay thường dùng hộp số ...
GEARBOX_TYPES_RE = re.compile(r"^\s*Trên\s+ôtô\s+hiện\s+nay\s+thường\s+dùng\s+hộp\s+số\b", flags=re.IGNORECASE | re.UNICODE)
# 28) Trong khi xe chạy, các tín hiệu ...
AUTO_GEARBOX_SIGNALS_RE = re.compile(r"^\s*Trong\s+khi\s+xe\s+chạy,\s*các\s+tín\s+hiệu\b", flags=re.IGNORECASE | re.UNICODE)
# 29) Truyền động các đăng dùng để ...
DRIVESHAFT_USAGE_RE = re.compile(r"^\s*Truyền\s+động\s+các\s+đăng\s+dùng\s+để\b", flags=re.IGNORECASE | re.UNICODE)
# 30) Trong xe ôtô, truyền động các đăng ...
DRIVESHAFT_IN_CAR_RE = re.compile(r"^\s*Trong\s+xe\s+ôtô,\s*truyền\s+động\s+các\s+đăng\b", flags=re.IGNORECASE | re.UNICODE)
# 31) Khi người lái (đạp/nhả) bàn đạp ly hợp ...
CLUTCH_EVENT_RE = re.compile(r"^\s*Khi\s+người\s+lái\s+(?:xe\s+)?(đạp|nhả)\s+bàn\s+đạp\s+ly\s*hợp\s*(?:((?:chưa\s+hết|hết)\s+hành\s+trình))?", flags=re.IGNORECASE | re.UNICODE)

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
    return s.strip()

def _clean_term(term: str, max_words: int = 8) -> str:
    term = term.strip(' "“”\'’:;,.()[]')
    # Giới hạn độ dài hợp lý để tránh bắt cả mệnh đề dài
    words = term.split()
    if len(words) == 0 or len(words) > max_words:
        return ""
    # Loại các mấu từ thường khiến câu trở nên dài/không phải là danh ngữ thuần
    BAD_TOKENS = {"gồm", "bao", "bao_gồm", "được", "trong", "với", "khi", "ở", "do", "đến"}
    low = term.lower()
    if any(tok in low for tok in BAD_TOKENS):
        # Cho phép "cấu tạo|hệ thống|động cơ|máy phát|ắc quy" là đầu mục hợp lệ
        if not re.match(r"^(cấu tạo|hệ thống|động cơ|máy phát|ắc quy)\b", low):
            return ""
    # Viết hoa chữ cái đầu nếu cần
    if term and term[0].islower():
        term = term[0].upper() + term[1:]
    return term

def guess_question_from_chunk(chunk_text: str, title: str, index: int) -> str:
    # Ưu tiên các mẫu đặc thù để đa dạng câu hỏi
    # A) Ý nghĩa của <X>
    m = MEANING_RE.match(chunk_text)
    if m:
        obj = _clean_term(m.group(1))
        if obj:
            return f"Ý nghĩa của “{obj}” là gì?"

    # B) Cấu tạo/Cấu trúc (của) <X> gồm ...
    m = COMPOSITION2_RE.match(chunk_text)
    if m:
        obj = _clean_term(m.group(1))
        if obj:
            return f"Cấu tạo của “{obj}” gồm những phần nào?"

    # C) Ở hành trình/Ở cuối kỳ ...
    m = STAGE_RE.match(chunk_text)
    if m:
        stage = _clean_term(m.group(1), max_words=6)
        if stage:
            # Avoid using the lesson title in generated questions.
            return f"Mô tả “{stage}” là gì?"

    # D) Vòi phun điện tử / EFI
    if INJECT_EFI_RE.search(chunk_text):
        return "Hệ thống phun xăng điện tử hoạt động như thế nào?"

    # E) Động cơ diesel
    if DIESEL_RE.search(chunk_text):
        # Title not used: keep question generic and title-free.
        return "Hệ thống cung cấp nhiên liệu cho động cơ diesel gồm những loại nào?"

    # F) Thông số trên hông lốp
    if SIDEWALL_TIRE_RE.search(chunk_text):
        return "Các thông số in trên hông lốp cho biết những gì?"

    # G) Vai trò/chức năng khi câu mở đầu là danh từ
    m = ROLE_RE.match(chunk_text)
    if m:
        subj = _clean_term(m.group(1))
        if subj:
            return f"Vai trò của “{subj}” là gì?"

    # H) <Chủ thể> có hai loại chính
    m = TWO_MAIN_TYPES_RE.match(chunk_text)
    if m:
        subj = _clean_term(m.group(1))
        if subj:
            return f"Các loại chính của “{subj}” là gì?"

    # I) Các thực thể điện phổ biến
    m = ELECTRIC_SUBJECT_RE.match(chunk_text)
    if m:
        subj = _clean_term(m.group(1))
        if subj:
            return f"Chức năng của “{subj}” là gì?"

    # J) Sự cố/hư hỏng: trích hiện tượng đầu câu
    m = ISSUE_RE.match(chunk_text)
    if m and title and ("Hư hỏng" in title or "sự cố" in chunk_text.lower()):
        phen = _clean_term(m.group(1), max_words=10)
        if phen:
            return f"Nguyên nhân và cách khắc phục “{phen}” là gì?"

    # 1) Cấu tạo chung của <X> gồm ...
    m = COMPOSITION_RE.match(chunk_text)
    if m:
        obj = _clean_term(m.group(1))
        if obj:
            return f"Cấu tạo chung của “{obj}” gồm những phần nào?"

    # 2) <Đối tượng> có công dụng/chức năng ...
    m = USAGE_RE.match(chunk_text)
    if m:
        obj = _clean_term(m.group(1))
        if obj:
            return f"Công dụng của “{obj}” là gì?"

    # 3) <Đối tượng> được phân loại ...
    m = CLASSIFY_RE.match(chunk_text)
    if m:
        obj = _clean_term(m.group(1))
        if obj:
            return f"“{obj}” được phân loại như thế nào?"

    # 4) <Đối tượng> hoạt động theo ...
    m = OPERATE_SEQ_RE.match(chunk_text)
    if m:
        obj = _clean_term(m.group(1))
        if obj:
            return f"Trình tự hoạt động của “{obj}” là gì?"

    # 5) Câu bắt đầu bằng "Nguyên lý hoạt động ..."
    if PRINCIPLE_RE.match(chunk_text):
        # Keep principle questions generic and avoid embedding the lesson title.
        return "Nguyên lý hoạt động gồm những bước nào?"

    # 6) Định nghĩa: <Thuật ngữ> là ...
    m = TERM_DEF_RE.match(chunk_text)
    if m:
        term = _clean_term(m.group(1))
        if term:
            return f"“{term}” là gì?"

    # K) Các dạng bố trí hệ thống truyền lực
    if DRIVETRAIN_LAYOUTS_RE.match(chunk_text):
        return "Các dạng bố trí hệ thống truyền lực trên ô tô là gì?"

    # L) Cụm cầu chủ động
    if ACTIVE_AXLE_RE.match(chunk_text):
        return "Cấu tạo của cụm cầu chủ động gồm những phần nào?"

    # M) Vỏ cầu chủ động
    if AXLE_HOUSING_ROLE_RE.match(chunk_text):
        return "Vai trò của vỏ cầu chủ động là gì?"

    # N) Truyền lực chính
    if FINAL_DRIVE_ROLE_RE.match(chunk_text):
        return "Chức năng của truyền lực chính là gì?"

    # O) Vi sai
    if DIFFERENTIAL_ROLE_RE.match(chunk_text):
        return "Chức năng của vi sai là gì?"

    # P) Hai bán trục
    if HALF_SHAFT_ROLE_RE.match(chunk_text):
        return "Chức năng của hai bán trục là gì?"

    # Q) Ly hợp chức năng
    if CLUTCH_FUNCTION_RE.match(chunk_text):
        return "Chức năng của ly hợp là gì?"

    # R) Hộp số thay đổi mô men
    if GEARBOX_FUNCTION_RE.match(chunk_text):
        return "Chức năng của hộp số là gì?"

    # S) Hộp số cơ khí 5 cấp
    if GEARBOX_5_RE.match(chunk_text):
        return "Cấu tạo của hộp số cơ khí 5 cấp tiến 1 lùi gồm những gì?"

    # T) Việc gài số trên hộp số cơ khí
    if GEARBOX_SHIFT_RE.match(chunk_text):
        return "Cách gài các số trên hộp số cơ khí như thế nào?"

    # U) Loại hộp số thường dùng hiện nay
    if GEARBOX_TYPES_RE.match(chunk_text):
        return "Những loại hộp số thường dùng trên ô tô hiện nay là gì?"

    # V) Tín hiệu hộp số tự động thu thập
    if AUTO_GEARBOX_SIGNALS_RE.match(chunk_text):
        return "Những tín hiệu nào được hộp số tự động thu thập khi xe chạy?"

    # W) Truyền động các đăng dùng để
    if DRIVESHAFT_USAGE_RE.match(chunk_text):
        return "Công dụng của truyền động các đăng là gì?"

    # X) Truyền động các đăng trong ô tô
    if DRIVESHAFT_IN_CAR_RE.match(chunk_text):
        return "Vai trò của truyền động các đăng trong ô tô là gì?"

    # Y) Sự kiện hành động với ly hợp
    m = CLUTCH_EVENT_RE.match(chunk_text)
    if m:
        action = m.group(1).lower()
        state_raw = m.group(2) or ""
        if "chưa" in state_raw:
            if action == "đạp":
                return "Điều gì xảy ra khi người lái chỉ đạp một phần bàn đạp ly hợp?"
            else:
                return "Điều gì xảy ra khi người lái chỉ nhả một phần bàn đạp ly hợp?"
        elif "hết" in state_raw:
            if action == "đạp":
                return "Điều gì xảy ra khi người lái đạp hết bàn đạp ly hợp?"
            else:
                return "Điều gì xảy ra khi người lái nhả hết bàn đạp ly hợp?"
        else:
            if action == "đạp":
                return "Điều gì xảy ra khi người lái đạp bàn đạp ly hợp?"
            else:
                return "Điều gì xảy ra khi người lái nhả bàn đạp ly hợp?"

    # 7) Mặc định (fallback nâng cao): cố gắng suy ra một danh ngữ phù hợp, tránh mảnh vỡ.
    STOP_SUFFIXES = {"còn", "thì", "ra", "vì", "do", "bằng", "nhả", "ly", "việc", "gài", "dẫn", "chủ", "thứ", "được"}
    DOMAIN_NOUNS = [
        "ô tô", "động cơ", "hộp số", "ly hợp", "truyền động các đăng", "cụm cầu chủ động", "vỏ cầu chủ động",
        "truyền lực chính", "vi sai", "bán trục", "bánh xe", "lốp", "hệ thống lái", "hệ thống phanh",
        "hệ thống phanh chân", "hệ thống phanh đỗ", "hệ thống điện", "ắc quy", "máy phát điện",
        "bộ điều chỉnh điện", "hệ thống đánh lửa", "máy khởi động", "dây đai an toàn", "túi khí", "ECU"
    ]

    def _extract_subject(text: str) -> str:
        clause = re.split(r"[\n\.;:]\s*", text.strip())[0]
        clause = re.sub(r"^\s*(?:Các|Những|Một|Một\s+số|Về|Với|Trên|Khi)\s+", "", clause, flags=re.IGNORECASE)
        words = clause.split()
        # try expanding to up to 8 words but stop at stop-suffix
        candidate_words: List[str] = []
        for w in words:
            clean_w = w.strip(',"“”').lower()
            if clean_w in STOP_SUFFIXES:
                break
            candidate_words.append(w.strip(',"“”'))
            if len(candidate_words) >= 8:
                break
        candidate = " ".join(candidate_words)
        # If too short or ends with stop suffix -> reject
        if len(candidate_words) < 1:
            return ""
        last = candidate_words[-1].lower()
        if last in STOP_SUFFIXES or len(candidate_words) == 1 and len(last) < 3:
            candidate = ""
        candidate_clean = _clean_term(candidate)
        if candidate_clean:
            return candidate_clean
        # Domain noun scan fallback
        low = text.lower()
        best = ""
        for noun in DOMAIN_NOUNS:
            if noun in low:
                if len(noun) > len(best):
                    best = noun
        return _clean_term(best)

    subj = _extract_subject(chunk_text)
    if subj:
        return f"Giải thích “{subj}” là gì?"

    return "Giải thích nội dung đoạn sau bằng một câu ngắn." 

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

def main():
    ap = argparse.ArgumentParser(description="Convert lesson JSON -> prompt/response JSONL.")
    ap.add_argument("--input", required=True, help="Path to input .json/.jsonl or folder")
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

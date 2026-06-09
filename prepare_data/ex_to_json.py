# qa_xlsx_to_flat_json.py
import json
import uuid
from pathlib import Path
from typing import Optional, List, Dict

from openpyxl import load_workbook
from openpyxl.styles.colors import Color
from docx import Document  # pip install python-docx
import re

# =========================
# CẤU HÌNH
# =========================
INPUT_XLSX = "DS CÂU HỎI AI CÁC MÔN LT_all.xlsx"

# None = xử lý tất cả sheet theo thứ tự xuất hiện trong file
SUBJECT_SHEETS: Optional[List[str]] = None

# (Không còn dùng subject/topic trong output)
SUBJECT_NAMES: Optional[List[str]] = [
    "Đạo đức, văn hóa giao thông và phòng chống tác hại của rượu, bia khi tham gia giao thông",
    "Cấu tạo và sửa chữa thông thường",
    "Kỹ thuật lái xe",
    "Pháp luật giao thông đường bộ",
]

# (tuỳ chọn) nếu vẫn cần dò tiêu đề từ Word
SUBJECT_FILES: List[str] = [
    "kb_DDNLX.docx",       # cho SUBJECT_NAMES[0]
    "KichBanCTSCTT.docx",  # cho SUBJECT_NAMES[1]
    "KichBanKTLX.docx",    # cho SUBJECT_NAMES[2]
    "kb_PLGTDB.docx",      # cho SUBJECT_NAMES[3]
]

ROW_START = 2
QUESTION_COL = "A"
ANSWER_COL = "B"
COLOR_REF_COL = "A"

# File JSON đích: MẢNG các object {_id, lesson_id, question, answer, subject_id, lesson_number}
OUTPUT_JSON = "qa_flat.json"

# lessons_index.json: danh sách lessons đã parse từ .docx
LESSONS_INDEX_JSON = "lesson/lessons_index.json"

SKIP_EMPTY_ROWS = True
TREAT_NONE_FILL_AS_COLOR = True  # coi NOFILL là một "màu" hợp lệ để tách bài

# =========================
# HÀM TIỆN ÍCH EXCEL
# =========================
def color_key_from_openpyxl(color: Optional[Color]) -> Optional[str]:
    if color is None:
        return None
    val = getattr(color, "rgb", None)
    if val:
        return f"RGB:{str(val).upper()}"
    theme = getattr(color, "theme", None)
    tint = getattr(color, "tint", None)
    if theme is not None:
        return f"THEME:{theme}:{tint}"
    indexed = getattr(color, "indexed", None)
    if indexed is not None:
        return f"INDEXED:{indexed}"
    return str(color)

def get_cell_fill_key(ws, row: int, col_letter: str) -> Optional[str]:
    cell = ws[f"{col_letter}{row}"]
    fill = getattr(cell, "fill", None)
    # ô không tô màu
    if fill is None or getattr(fill, "patternType", None) in (None, "none"):
        return "NOFILL" if TREAT_NONE_FILL_AS_COLOR else None
    start_color = getattr(fill, "start_color", None)
    key = color_key_from_openpyxl(start_color)
    return key if key is not None else ("NOFILL" if TREAT_NONE_FILL_AS_COLOR else None)

def get_cell_str(ws, row: int, col_letter: str) -> str:
    val = ws[f"{col_letter}{row}"].value
    return "" if val is None else str(val).strip()

def row_is_empty(q: str, a: str) -> bool:
    return (q.strip() == "" and a.strip() == "")

# =========================
# (Tuỳ chọn) HỖ TRỢ WORD – nếu cần đọc tiêu đề "Bài ..."
# =========================
_BAI_RE = re.compile(r"\bbài\b", flags=re.IGNORECASE)

def _bold_chunks_in_paragraph(par) -> List[str]:
    chunks, cur = [], []
    def flush():
        nonlocal chunks, cur
        if cur:
            text = "".join(cur)
            if text.strip():
                chunks.append(text)
        cur = []
    for run in par.runs:
        if bool(getattr(run, "bold", False)):
            cur.append(run.text or "")
        else:
            flush()
    flush()
    return chunks

def _extract_bai_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    m = _BAI_RE.search(text)
    if not m:
        return None
    out = text[m.start():].strip()
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"[\.:\-–—]\s*$", "", out)
    return out

def load_bai_titles_from_docx(docx_path: Path) -> List[str]:
    try:
        doc = Document(str(docx_path))
    except Exception:
        return []
    titles: List[str] = []
    for p in doc.paragraphs:
        for chunk in _bold_chunks_in_paragraph(p):
            t = _extract_bai_from_text(chunk)
            if t:
                titles.append(t)
    # Khử trùng lặp liên tiếp
    dedup: List[str] = []
    for t in titles:
        if not dedup or dedup[-1] != t:
            dedup.append(t)
    return dedup

# =========================
# LOAD lessons_index.json → map subject_id → [lesson_uuid1, lesson_uuid2, ...]
# =========================
def load_lesson_uuid_map(lessons_index_path: Path) -> Dict[int, List[str]]:
    """
    Đọc lessons_index.json (mỗi record như:
      {"_id": "...", "subject_id": 1, "tittle": "...", "chunks":[...], ...}
    ) và gom theo subject_id, giữ nguyên thứ tự xuất hiện trong file.
    Nếu file có field "lesson" thì vẫn giữ thứ tự hiện có (không sort),
    vì thường file đã được ghi đúng thứ tự.
    """
    if not lessons_index_path.exists():
        raise FileNotFoundError(f"Không thấy {lessons_index_path}")

    with lessons_index_path.open("r", encoding="utf-8") as f:
        lessons = json.load(f)

    out: Dict[int, List[str]] = {}
    for rec in lessons:
        sid = int(rec.get("subject_id", 0))
        lid = str(rec.get("_id"))
        if not sid or not lid:
            continue
        out.setdefault(sid, []).append(lid)
    return out

# =========================
# XỬ LÝ TỪNG SHEET → TRẢ VỀ DANH SÁCH {_id, lesson_id, question, answer, subject_id, lesson_number}
# =========================
def sheet_to_flat_items(ws, subject_id: int, lesson_uuid_list: List[str]) -> List[dict]:
    """
    - Phân bài theo đổi màu (COLOR_REF_COL)
    - Map bài thứ n (1-based) của subject_id này → lesson_uuid_list[n-1] nếu có
    """
    max_row = ws.max_row
    items: List[dict] = []

    prev_color_key: Optional[str] = None
    lesson_number = 0  # sẽ tăng khi đổi màu

    for r in range(ROW_START, max_row + 1):
        q = get_cell_str(ws, r, QUESTION_COL)
        a = get_cell_str(ws, r, ANSWER_COL)

        if SKIP_EMPTY_ROWS and row_is_empty(q, a):
            continue

        color_key = get_cell_fill_key(ws, r, COLOR_REF_COL)

        # khởi tạo lesson đầu tiên
        if prev_color_key is None:
            prev_color_key = color_key
            lesson_number = 1
        else:
            # khi màu đổi -> sang bài mới
            if color_key != prev_color_key:
                lesson_number += 1
                prev_color_key = color_key

        # Lấy lesson_id (UUID) từ lessons_index.json theo thứ tự
        lesson_id = lesson_uuid_list[lesson_number - 1] if (0 <= lesson_number - 1 < len(lesson_uuid_list)) else None
        if lesson_id is None:
            # Cảnh báo nhẹ nếu vượt quá số lesson đã có
            # (không raise để vẫn xuất kết quả)
            print(f"⚠️ subject_id={subject_id} có bài #{lesson_number} (theo màu) nhưng lessons_index chỉ có {len(lesson_uuid_list)} bài.")

        items.append({
            "_id": str(uuid.uuid4()),
            "lesson_id": lesson_id,          # UUID lấy từ lessons_index.json
            "question": q,
            "answer": a,
            "lesson": lesson_number  # số thứ tự theo màu trong sheet
        })
    return items

# =========================
# MAIN
# =========================
def main():
    # 1) Load map lesson UUID theo subject_id
    lesson_uuid_map = load_lesson_uuid_map(Path(LESSONS_INDEX_JSON))

    # 2) Mở Excel
    xlsx_path = Path(INPUT_XLSX)
    wb = load_workbook(filename=str(xlsx_path), data_only=True)

    target_sheet_names = SUBJECT_SHEETS if SUBJECT_SHEETS is not None else wb.sheetnames

    # (Tuỳ chọn) đọc tiêu đề từ docx, hiện không dùng trong output
    _ = {i: load_bai_titles_from_docx(Path(fn)) for i, fn in enumerate(SUBJECT_FILES)}

    flat: List[dict] = []
    for idx, name in enumerate(target_sheet_names, start=1):
        if name not in wb.sheetnames:
            print(f"⚠️ Bỏ qua sheet '{name}' vì không tồn tại trong file.")
            continue

        ws = wb[name]
        subject_id = idx  # sheet thứ idx → subject_id = idx

        lesson_uuid_list = lesson_uuid_map.get(subject_id, [])
        if not lesson_uuid_list:
            print(f"⚠️ Không tìm thấy lesson cho subject_id={subject_id} trong {LESSONS_INDEX_JSON}.")

        flat.extend(sheet_to_flat_items(ws, subject_id, lesson_uuid_list))

    out_path = Path(OUTPUT_JSON)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(flat, f, ensure_ascii=False, indent=2)

    print(f"OK! Tổng {len(flat)} dòng → {out_path.resolve()}")

if __name__ == "__main__":
    main()
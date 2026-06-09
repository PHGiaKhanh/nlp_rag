import json
from pathlib import Path

# file gốc
INPUT_FILE = "lesson/lessons_index.json"

# thư mục output
OUTPUT_DIR = Path("data/lesson")

# ánh xạ subject_id → tên file
SUBJECT_MAP = {
    1: "DD.json",        # Đạo đức
    2: "CTSCTT.json",    # Cấu tạo và sửa chữa thông thường
    3: "KTLX.json",      # Kỹ thuật lái xe
    4: "PLGTDB.json"     # Pháp luật giao thông đường bộ
}

def split_json(input_path: str, subject_map: dict):
    # Đọc file gốc
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Tạo dict để chứa dữ liệu phân loại
    grouped = {sid: [] for sid in subject_map.keys()}

    # Phân loại theo subject_id
    for item in data:
        sid = item.get("subject_id")
        if sid in grouped:
            grouped[sid].append(item)

    # Đảm bảo thư mục output tồn tại
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Ghi ra từng file trong data/lesson
    for sid, filename in subject_map.items():
        out_path = OUTPUT_DIR / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(grouped[sid], f, ensure_ascii=False, indent=2)
        print(f"Đã tạo file {out_path} ({len(grouped[sid])} items)")

if __name__ == "__main__":
    split_json(INPUT_FILE, SUBJECT_MAP)

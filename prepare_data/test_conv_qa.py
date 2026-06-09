import json
from pathlib import Path

INPUT_JSON = "DS CÂU HỎI AI CÁC MÔN LT_new.json"

TARGET_SUBJECT = " Kỹ thuật lái xe"
TARGET_LESSON = "7"   # Nếu cần lọc theo số bài, bạn phải có trường 'lesson' trong dữ liệu

def load_data(path: str):
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def main():
    data = load_data(INPUT_JSON)
    
    total_subjects = len(data)
    print(f"✅ Tổng số subject: {total_subjects}")

    print("\n=== Thống kê từng subject ===")
    for subj in data:
        subject_name = subj.get("subject", "")
        questions = subj.get("questions", [])
        num_questions = len(questions)
        print(f"- {subject_name}: {num_questions} câu hỏi")

    # In câu hỏi & trả lời đầu tiên của subject lựa chọn
    subj_block = next((s for s in data if s.get("subject") == TARGET_SUBJECT), None)
    if subj_block is None:
        print(f"⚠️ Không tìm thấy subject '{TARGET_SUBJECT}' trong dữ liệu.")
        return

    questions = subj_block.get("questions", [])
    if not questions:
        print(f"⚠️ Subject '{TARGET_SUBJECT}' không có câu hỏi.")
        return

    first_item = questions[0]
    q = first_item.get("question", "").strip() if "question" in first_item else ""
    a = first_item.get("answer_text", "").strip()
    print(f"===Subject: {TARGET_SUBJECT}===")
    print(f"Câu hỏi đầu tiên: {q if q else '[Không có trường question]'}")
    print(f"Trả lời: {a}")

if __name__ == "__main__":
    main()

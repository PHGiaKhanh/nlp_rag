import json
import uuid
import pandas as pd
import re

def extract_dieu_number(text):
    """
    Trích xuất số điều từ text (ví dụ: "Điều 6" -> 6)
    """
    if pd.isna(text):
        return None
    
    # Tìm pattern "Điều X" hoặc "điều X"
    match = re.search(r'[Đđ]iều\s+(\d+)', str(text))
    if match:
        return int(match.group(1))
    
    # Nếu chỉ là số thuần túy
    try:
        return int(text)
    except:
        return None

def load_lessons_data(json_file):
    """
    Đọc file lessons_xp_with_dieu.json và tạo mapping từ số điều sang _id
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        lessons = json.load(f)
    
    # Tạo dictionary mapping: lesson number -> _id
    lesson_map = {}
    for lesson in lessons:
        lesson_num = lesson.get('lesson')
        if lesson_num:
            lesson_map[lesson_num] = lesson['_id']
    
    return lesson_map

def process_penalty_data(excel_file, lesson_map):
    """
    Đọc file penalty.xlsx và tạo dữ liệu Q&A
    """
    # Đọc file Excel
    df = pd.read_excel(excel_file)
    
    print(f"Đã đọc {len(df)} dòng từ file Excel")
    print(f"Các cột trong file: {df.columns.tolist()}")
    
    # Tạo danh sách kết quả
    penalty_qa_list = []
    
    for idx, row in df.iterrows():
        # Kiểm tra các cột có thể có trong file Excel
        # Điều chỉnh tên cột theo file thực tế của bạn
        lesson_col = None
        question_col = None
        answer_col = None
        
        # Tìm cột chứa thông tin "Điều"
        for col in df.columns:
            col_lower = str(col).lower()
            if 'điều' in col_lower or 'dieu' in col_lower or 'lesson' in col_lower:
                lesson_col = col
            elif 'lỗi' in col_lower or 'vi phạm' in col_lower or 'question' in col_lower or 'hành vi' in col_lower:
                question_col = col
            elif 'phạt' in col_lower or 'mức' in col_lower or 'answer' in col_lower or 'xử phạt' in col_lower:
                answer_col = col
        
        # Nếu không tìm thấy, sử dụng cột đầu tiên làm mặc định
        if lesson_col is None:
            lesson_col = df.columns[0] if len(df.columns) > 0 else None
        if question_col is None:
            question_col = df.columns[1] if len(df.columns) > 1 else None
        if answer_col is None:
            answer_col = df.columns[2] if len(df.columns) > 2 else None
        
        # Lấy dữ liệu từ row
        lesson_text = row.get(lesson_col) if lesson_col else None
        question = row.get(question_col) if question_col else None
        answer = row.get(answer_col) if answer_col else None
        
        # Bỏ qua nếu thiếu dữ liệu
        if pd.isna(question) or pd.isna(answer):
            continue
        
        # Trích xuất số điều
        lesson_number = extract_dieu_number(lesson_text)
        
        if lesson_number is None:
            print(f"Cảnh báo: Không tìm thấy số điều tại dòng {idx + 2}: {lesson_text}")
            continue
        
        # Tìm lesson_id từ mapping
        lesson_id = lesson_map.get(lesson_number)
        
        if lesson_id is None:
            print(f"Cảnh báo: Không tìm thấy lesson_id cho Điều {lesson_number} tại dòng {idx + 2}")
            continue
        
        # Tạo đối tượng Q&A
        qa_item = {
            "_id": str(uuid.uuid4()),
            "lesson_id": lesson_id,
            "question": str(question).strip(),
            "answer": str(answer).strip(),
            "lesson": lesson_number
        }
        
        penalty_qa_list.append(qa_item)
    
    return penalty_qa_list

def main():
    # Đường dẫn file
    lessons_file = 'lessons_xp_with_dieu.json'
    penalty_file = '../penalty.xlsx'
    output_file = 'penalty.json'
    
    try:
        # Đọc dữ liệu lessons
        print("Đang đọc file lessons...")
        lesson_map = load_lessons_data(lessons_file)
        print(f"Đã tải {len(lesson_map)} lessons")
        
        # Xử lý dữ liệu penalty
        print("\nĐang xử lý file penalty.xlsx...")
        penalty_qa_list = process_penalty_data(penalty_file, lesson_map)
        
        # Lưu kết quả
        print(f"\nĐang lưu {len(penalty_qa_list)} câu hỏi vào {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(penalty_qa_list, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Hoàn thành! Đã tạo {len(penalty_qa_list)} câu hỏi trong file {output_file}")
        
        # Hiển thị một vài mẫu
        if penalty_qa_list:
            print("\nMẫu dữ liệu đầu tiên:")
            print(json.dumps(penalty_qa_list[0], ensure_ascii=False, indent=2))
        
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy file - {e}")
    except Exception as e:
        print(f"Lỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

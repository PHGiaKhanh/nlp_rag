#chạy code kiểu python embed/em_lesson.py data/lesson/CTSCTT.json

import json
import sys
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ====== Load model Vi-bge ======
MODEL_ID = "AITeamVN/Vietnamese_Embedding"
model = SentenceTransformer(MODEL_ID)

def embed_texts(texts):
    """Sinh embedding từ danh sách text"""
    return model.encode(
        texts,
        normalize_embeddings=True,  # cosine-friendly
        batch_size=32,
        show_progress_bar=False
    ).astype(np.float32)

def main():
    # ====== Nhận đường dẫn file qua terminal ======
    if len(sys.argv) < 2:
        print("❌ Bạn chưa nhập đường dẫn file JSON.")
        print("👉 Cách dùng: python test_em_lesson.py <path_toi_file_json>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"❌ Không tìm thấy file: {input_path}")
        sys.exit(1)

    # ====== Load dữ liệu ======
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Lỗi đọc JSON ở {input_path}: {e}")
        sys.exit(1)

    # ====== Gom tất cả text của chunks để embed theo batch ======
    chunk_refs = []   # (doc_idx, chunk_idx)
    texts = []
    for di, doc in enumerate(data):
        chunks = doc.get("chunks", [])
        for ci, chunk in enumerate(chunks):
            text = (chunk.get("text") or "").strip()
            if text:
                chunk_refs.append((di, ci))
                texts.append(text)

    # Không có text hợp lệ
    if not texts:
        print("⚠️ Không tìm thấy 'text' hợp lệ trong bất kỳ chunk nào.")
    else:
        # ====== Tạo embeddings theo batch ======
        embs = embed_texts(texts)

        # ====== Gán embed về lại từng chunk ======
        for (di, ci), emb in zip(chunk_refs, embs):
            data[di]["chunks"][ci]["embed"] = emb.tolist()

        # Với chunk không có text → set None
        for di, doc in enumerate(data):
            for ci, chunk in enumerate(doc.get("chunks", [])):
                if "embed" not in chunk:
                    chunk["embed"] = None

    # ====== Ghi ra JSON (giữ nguyên cấu trúc) ======
    out_dir = Path("embed_data/lesson/")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{input_path.stem}_with_emb.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Done! Đã tạo file {out_path}")

if __name__ == "__main__":
    main()

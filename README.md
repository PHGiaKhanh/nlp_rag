# NLP RAG cho Hệ thống Câu Hỏi Lý Thuyết Lái Xe

Dự án này xây dựng một hệ thống truy xuất và trả lời câu hỏi theo hướng RAG (Retrieval-Augmented Generation) dành cho kiến thức lý thuyết lái xe ô tô và tài liệu đào tạo liên quan.

Nội dung chính của hệ thống gồm:
- chuẩn bị dữ liệu và tách theo môn học / bài học
- sinh embedding cho văn bản và QA
- tìm kiếm theo BM25 và semantic search
- chọn ngữ cảnh phù hợp
- gọi mô hình LLM qua Ollama để trả lời

---

## 1. Tổng quan dự án

Project gồm 3 lớp chức năng chính:

1. Data preparation
   - chuẩn hóa dữ liệu từ các file JSON
   - tách theo môn học, bài học, QA
   - tạo dữ liệu huấn luyện cho mô hình fine-tune

2. Retrieval layer
   - BM25 cho tra cứu lexical
   - semantic search dựa trên embedding
   - ưu tiên ngữ cảnh từ lesson / dialogue / qa600

3. Generation layer
   - gọi mô hình LLM qua Ollama
   - tạo câu trả lời dựa trên các đoạn ngữ cảnh đã truy xuất

Mục tiêu của dự án là trả lời các câu hỏi như:
- biển báo nào có ý nghĩa gì?
- luật giao thông liên quan đến trường hợp nào?
- định nghĩa / khái niệm trong lý thuyết lái xe

## 2. Luồng xử lý dữ liệu

### 2.1 Chuẩn bị dữ liệu

Các script trong `prepare_data/` thực hiện các bước:
- chuyển đổi dữ liệu thô sang định dạng JSON chuẩn
- chia theo môn học và bài học
- tách QA theo chủ đề
- chuẩn hóa nội dung và làm sạch text

Một số điểm chính:

- `prepare_data/split_qa_by_subject.py`
  - chia `qa_flat.json` thành nhiều file theo `subject_id`
  - ví dụ: `DD.json`, `CTSCTT.json`, `KTLX.json`, `PLGTDB.json`

- `prepare_data/lesson/json_lesson.py`
  - xử lý dữ liệu lesson
  - tạo cấu trúc dữ liệu theo bài học

- `prepare_data/embed/qa_embed.py`
  - tạo embedding cho QA bằng `SentenceTransformer`

- `prepare_data/embed/em_lesson.py`
  - tạo embedding cho lesson/chunk text

### 2.2 Tạo embedding

Các file embedding lưu ở `embed_data/dialogue/` và `embed_data/lesson/`.

Các file có dạng:
- `*_with_emb.json`
- mỗi record/chunk có trường `embed` hoặc `embedding`

Điều này cho phép hệ thống thực hiện:
- semantic search trên vector corpus
- tương đồng cosine similarity

---

## 3. Kiến trúc Retrieval + LLM

### 3.1 BM25
- `retriever/bm25.py`

BM25 dùng để kiếm các câu hỏi / đoạn văn bản phù hợp theo từ khóa. Đây là lớp tìm kiếm lexical, hiệu quả với truy vấn có từ khóa rõ ràng.

### 3.2 Semantic Search
- `retriever/model_semantic.py`
- `retriever/load_data.py`

Semantic search dùng embedding tiếng Việt của mô hình `AITeamVN/Vietnamese_Embedding` hoặc các vector đã được tạo sẵn. Hệ thống nhúng câu hỏi và so sánh với corpus đã nhúng để tìm đoạn tương đồng nhất.

### 3.3 Tích hợp LLM
- `retriever/model_llm.py`
- `retriever/main.py`

LLM được gọi qua Ollama, không dùng mô hình HF cục bộ mặc định nữa.

Các biến môi trường quan trọng:

```bash
LLM_MODEL_ID=phogpt4b-ft-v3
LLM_OLLAMA_HOST=http://localhost:11434
LLM_TEMP=0.3
LLM_MAX_NEW=1536
```

## 4. Ví dụ nhanh

### Ví dụ 1: chạy query một lần

```bash
python retriever/main.py --once "Nồng độ cồn tối đa cho người lái xe?"
```

### Ví dụ 2: chạy với model khác

```bash
python retriever/main.py --once "Cách xử lý khi gặp đèn đỏ ở giao lộ?" --llm-model-id phogpt4b-ft-v3 --llm-temp 0.2
```

---

## 5. Mô hình dữ liệu

Dữ liệu chính nằm trong `data/` và `embed_data/`:

- `data/dialogue/*`: hội thoại / câu hỏi và câu trả lời
- `data/lesson/*`: bài học / tài liệu lý thuyết
- `data/qa600.json`: bộ QA lớn dùng cho BM25
- `sub.json`: thông tin các môn học / chủ đề

Các file `embed_data/..._with_emb.json` chứa embeddings đã tạo sẵn, thuận tiện cho thực nghiệm nhanh mà không cần tính toán lại embedding mỗi lần chạy.

---

## 6. Một số lưu ý quan trọng

- Dự án có nhiều phiên bản và script tiền xử lý khác nhau. Một số file trong `finetune-modelphogpt/prepare_data/` là cho việc fine-tune mô hình, không phải pipeline truy xuất chính.
- Một số notebook / script hỗ trợ demo hoặc kiểm tra, không phải entrypoint chính.
- Nếu máy chạy không có Ollama hoặc mô hình tương ứng, hệ thống có thể trả về câu trả lời rỗng hoặc cảnh báo lỗi.
- Phần `retriever/main.py` là điểm khởi đầu đáng tin cậy nhất để chạy demo tích hợp.

---

## 7. Các file chính nên dùng khi phát triển

- `retriever/main.py` — pipeline chính để test và chạy demo
- `services/get_results_bm25_and_semantic_service.py` — service tổng hợp kết quả BM25 + semantic
- `retriever/model_llm.py` — giao diện LLM qua Ollama
- `prepare_data/embed/qa_embed.py` — tạo embedding cho QA
- `prepare_data/embed/em_lesson.py` — tạo embedding cho lesson
- `prepare_data/split_qa_by_subject.py` — chia dữ liệu QA theo môn học

---

## 8. Kết luận

Dự án này là một nền tảng RAG đầy đủ cho hệ thống hỏi đáp về lý thuyết lái xe, kết hợp:
- tìm kiếm từ khóa mạnh (BM25)
- tìm kiếm ngữ nghĩa mạnh (semantic embedding)
- sinh câu trả lời bằng LLM qua Ollama

Nếu cần phát triển tiếp, bạn có thể mở rộng theo các hướng sau:
- tăng chất lượng dữ liệu embedding
- thêm bộ lọc theo môn học / bài học
- tối ưu hội thoại / memory
- fine-tune mô hình cho tiếng Việt chuyên ngành giao thông

---

## 9. Gợi ý phát triển tiếp theo
- thêm API REST hoặc FastAPI để expose cho client
- lưu lịch sử hội thoại trong DB hoặc JSON
- xây dựng benchmark đánh giá độ chính xác theo bộ câu hỏi mẫu
- tích hợp hệ thống ranking với trọng số ưu tiên lesson > dialogue > qa600
- tối ưu prompt để trả lời ngắn gọn, rõ ràng và đúng chuẩn pháp luật

---

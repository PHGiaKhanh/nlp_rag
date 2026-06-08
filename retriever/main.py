import sys, types, os

def _patch_multihead_attention():
    try:
        import torch.nn as nn, inspect
        if 'dtype' in inspect.signature(nn.MultiheadAttention.__init__).parameters:
            print("[INFO] MultiheadAttention already supports dtype parameter")
            return
        if not hasattr(nn.MultiheadAttention, "_original_init"):
            nn.MultiheadAttention._original_init = nn.MultiheadAttention.__init__
        def patched_init(self, *args, **kwargs):
            # Loại các tham số không hỗ trợ ở bản torch cũ
            for k in ["dtype", "bias", "add_bias_kv", "add_zero_attn"]:
                if k in kwargs:
                    kwargs.pop(k, None)
            return nn.MultiheadAttention._original_init(self, *args, **kwargs)
        nn.MultiheadAttention.__init__ = patched_init
        print("[INFO] Successfully patched MultiheadAttention")
    except Exception as e:
        print(f"[WARN] Could not patch MultiheadAttention: {e}")

def _patch_flash_attn():
    try:
        import flash_attn_triton  # noqa: F401
    except Exception:
        print("[INFO] Creating flash_attn_triton mock...")
        m = types.ModuleType("flash_attn_triton")
        m.HAS_FLASH_ATTN = False
        m.available = lambda: False
        sys.modules["flash_attn_triton"] = m

_patch_multihead_attention()
_patch_flash_attn()

# Force offline mode by default (có thể override bằng env nếu muốn)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ================= Imports =================
import argparse, re, unicodedata, json
from typing import List, Dict, Any, Optional, Tuple

from load_data import load_corpus
from model_semantic import SemanticSearcher
from bm25 import BM25
from model_llm import build_llm_from_args, generate_answer
# from chat_memory import resolve_after_preprocess, update_focus_after_sem
from preprocess import clean_user_input

# ===== Small-talk / identity detection and answer enrichment helpers =====

def _build_answer_preface(orig_question: str) -> str:
    q = (orig_question or "").strip()
    if not q:
        return "Vâng, tôi có thể giúp gì cho bạn."
    topic = q.rstrip("?.! ")
    if len(topic) > 140:
        topic = topic[:140].rstrip() + "…"
    return f"Vâng, bạn muốn hỏi về {topic}."

def _maybe_with_preface(text: str, orig_question: str, enabled: bool) -> str:
    if not enabled:
        return (text or "").strip()
    body = (text or "").strip()
    pre = _build_answer_preface(orig_question)
    return (pre + "\n" + body).strip()

def _enrich_short_answer(answer_text: str, context_text: str, max_points: int = 3) -> str:
    # Đơn giản hóa: không còn enrich, trả về nguyên bản
    return (answer_text or "").strip()

# =========================
# Vietnamese Traffic Law Prompting System (Simple & Basic)
def build_simple_vietnamese_prompt(question: str, context: str, sample_answer: str = "") -> str:
    """Xây dựng prompt đơn giản cho lý thuyết lái xe Việt Nam"""
    
    # System prompt cơ bản
    system_prompt = """Bạn là trợ lý AI chuyên về LÝ THUYẾT LÁI XE Ô TÔ tại Việt Nam.

Nhiệm vụ:
- Nhận vào 2 tham số:
  1) CÂU HỎI của người dùng.
  2) NỘI DUNG LÝ THUYẾT THAM KHẢO (là đoạn text đã được xử lý, trích xuất từ tài liệu chuẩn).

Quy tắc quan trọng:
- Dựa vào NỘI DUNG LÝ THUYẾT THAM KHẢO để trả lời.
- Khi chắc chắn: trả lời đúng trọng tâm, rõ ràng, đầy đủ các nội dung tham khảo; dùng gạch đầu dòng để làm nổi bật ý chính.
- Luôn luôn sinh câu trả lời, không được từ chối trả lời.
- Không bịa đặt, không lấy dữ liệu có năm 2008.
- Văn phong: tiếng Việt, thân thiện, trung lập, dễ hiểu.
- Không hiển thị lại câu hỏi , các câu có dấu "?" cần chuyển đồi để kết hợp cho phù hợp câu trả lời.
- Có thể gợi ý thêm các nội dung tương tự 
- Sinh câu trả lời dài, chi tiết, đầy đủ.
- Đầu ra: chỉ câu trả lời (không thêm JSON, metadata hay cấu trúc kỹ thuật)."""

    # Làm sạch input
    clean_question = (question or "").strip()
    clean_context = (context or "").strip()  
    
    # Xây dựng prompt
    prompt_parts = [system_prompt]
    
    # Thêm nội dung tham khảo
    if clean_context:
        prompt_parts.append(f"\nNỘI DUNG LÝ THUYẾT THAM KHẢO:\n{clean_context}")
    else:
        prompt_parts.append(f"\nNỘI DUNG LÝ THUYẾT THAM KHẢO:\n(Không có nội dung tham khảo)")
    
    # Thêm câu hỏi
    prompt_parts.append(f"\nCÂU HỎI:\n{clean_question}")
    return "\n".join(prompt_parts)

def simple_generate_answer(llm, question: str, context: str, max_new_tokens: int = 512, temperature: float = 0.3, max_prompt_chars: int = 12000) -> str:
    """Sinh câu trả lời với prompt đơn giản - FIXED"""
    try:
        if not hasattr(llm, 'generate'):
            return "LLM không khả dụng"
        
        print(f"[DEBUG] Sử dụng simple prompting system")
        
        # Xây dựng prompt đơn giản
        simple_prompt = build_simple_vietnamese_prompt(question, context)
        
        print(f"[DEBUG] Prompt length: {len(simple_prompt)} chars")
        
        # Truncate if too long (raise limit to allow longer context/answers)
        if len(simple_prompt) > max_prompt_chars:
            print(f"[WARN] Prompt quá dài (> {max_prompt_chars} chars), cắt ngắn...")
            # giữ nhiều context hơn để sinh câu trả lời dài hơn
            keep_ctx = max(1000, min(len(context), max_prompt_chars - 1500))
            context = (context[:keep_ctx] + "...") if len(context) > keep_ctx else context
            simple_prompt = build_simple_vietnamese_prompt(question, context)
        
        # Thiết lập parameters
        if hasattr(llm, 'set_params'):
            llm.set_params(
                temperature=temperature,
                max_new_tokens=max_new_tokens
            )
        
        # Sinh câu trả lời
        stop_tokens = ["<|endoftext|>", "</s>"]  # Simplified stop tokens
        
        print(f"[DEBUG] Calling LLM generate...")
        answer = llm.generate(simple_prompt, stop=stop_tokens)
        print(f"[DEBUG] LLM returned: '{answer[:100] if answer else 'None'}...'")
        
        # Làm sạch câu trả lời
        if answer:
            # Loại bỏ các từ thừa đầu câu
            clean_patterns = [
                r'^(trả lời|câu trả lời|đáp án|trợ lý trả lời)[:.]?\s*',
                r'^(dựa vào|theo|căn cứ)\s+.*?[:]\s*',
                r'^(tôi|em)\s+',
            ]
            
            for pattern in clean_patterns:
                answer = re.sub(pattern, '', answer, flags=re.IGNORECASE)
            
            answer = answer.strip()
        
        # ✅ CRITICAL FIX: Always return
        return answer or "Không thể tạo câu trả lời từ thông tin đã cho"
        
    except Exception as e:
        print(f"[ERROR] Simple generation failed: {e}")
        import traceback
        traceback.print_exc()
        # ✅ CRITICAL FIX: Return error message
        return f"Lỗi sinh câu trả lời: {str(e)}"
# ================= Pretty printing =================
def _term_width(default: int = 100) -> int:
    try:
        import shutil as _sh
        return max(60, _sh.get_terminal_size((default, 20)).columns)
    except Exception:
        return default

def _box(title: str, body: str, width: int = None) -> str:
    width = width or _term_width()
    line = "═" * (width - 2)
    top, bot = f"╔{line}╗", f"╚{line}╝"
    head, sep = f"║ {title.strip()}".ljust(width - 1) + "║", "╟" + ("─" * (width - 2)) + "╢"
    rows = []
    for paragraph in (body or "").split("\n"):
        if not paragraph.strip():
            rows.append("║".ljust(width - 1) + "║"); continue
        import textwrap as _tw
        for w in (_tw.wrap(paragraph, width=width - 4) or [""]):
            rows.append(("║ " + w).ljust(width - 1) + "║")
    return "\n".join([top, head, sep] + rows + [bot])

def _clean_answer(s: str) -> str:
    s = (s or "").strip()
    while "\n\n\n" in s: s = s.replace("\n\n\n", "\n\n")
    return s

def _preview_line(record: Dict[str, Any]) -> str:
    qa = record.get("qa") or {}; ch = record.get("channel")
    if ch in ("dialogue","qa600"):
        parts = []
        if qa.get("question"): parts.append("Q: " + qa["question"])
        if qa.get("answer"):   parts.append("A: " + qa["answer"])
        txt = " | ".join(parts)
    else:
        txt = (record.get("text") or "").strip().splitlines()
        txt = txt[0] if txt else ""
    return (txt[:180] + "…") if len(txt) > 180 else txt

def _print_hits(title: str, records: List[Dict[str, Any]], hits: List[Dict[str, Any]]):
    lines = []
    for i, h in enumerate(hits, 1):
        r = records[h["rid"]]
        src = h.get("src", ""); src_tag = f" [{src}]" if src else ""
        score_show = h.get("score", h.get("score_raw", 0.0))
        lines.append(f"#{i:02d}  score={score_show:.2f}{src_tag}  [{r['channel']}/{r['file']}]")
        lines.append("     " + _preview_line(r))
    print(_box(title, "\n".join(lines) if lines else "(không có)"))

# ================= Cleans & helpers =================
_word = re.compile(r"\w+", flags=re.UNICODE)
def _tok(s: str) -> List[str]:
    if not s: return []
    return [w.lower() for w in _word.findall(s)]

def clean_text(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFC", s).lower()
    s = re.sub(r"[^a-z0-9\u00c0-\u024f\u1e00-\u1eff\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _extract_q_and_a(r: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    qa = r.get("qa") or {}
    q = qa.get("question") or r.get("question")
    a = qa.get("answer")   or r.get("answer")
    q = q.strip() if isinstance(q, str) and q.strip() else None
    a = a.strip() if isinstance(a, str) and a.strip() else None
    return q, a

def _extract_text(r: Dict[str, Any]) -> str:
    for key in ["text","content","search_text","body","chunk_text","paragraph"]:
        v = r.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    q, a = _extract_q_and_a(r)
    if q or a: return ("Q: " + (q or "") + (f"\nA: {a}" if a else ""))
    return ""

# ================= context helpers =================
def dedup_by_source(records, hits, max_per_source: int = 1):
    seen, kept = {}, []
    for h in hits:
        r = records[h["rid"]]; key = f"{r.get('channel','?')}::{r.get('file','?')}"
        if seen.get(key, 0) < max_per_source:
            kept.append(h); seen[key] = seen.get(key, 0) + 1
    return kept

_DEF_PAT = re.compile(r"\b(là gì|định nghĩa|ý nghĩa|ký hiệu|đáp án)\b", re.IGNORECASE)
def sort_by_definition_first(records, hits):
    def score(h): return 1 if _DEF_PAT.search((records[h["rid"]].get("text") or "").lower()) else 0
    return sorted(hits, key=score, reverse=True)

def join_adjacent_chunks(records, hits, window: int = 1, max_chars: int = 1200) -> List[str]:
    out, taken = [], set()
    for h in hits:
        rid = h["rid"]
        if rid in taken: continue
        r = records[rid]; file_id = r.get("file")
        pieces = [r.get("text") or ""]
        for off in range(1, window+1):
            for rrid in [rid-off, rid+off]:
                if 0 <= rrid < len(records) and rrid not in taken:
                    rr = records[rrid]
                    if rr.get("file") == file_id:
                        pieces.append(rr.get("text") or ""); taken.add(rrid)
        merged = " ".join([p.strip() for p in pieces if p and p.strip()])
        out.append(merged[:max_chars] + ("…" if len(merged) > max_chars else ""))
        taken.add(rid)
    return out

def _build_context_block_from_hits(ctx_hits: List[str], topn_ctx: int, max_chars_per_item: int, max_total_chars: int) -> str:
    blocks, total = [], 0
    for idx, text in enumerate(ctx_hits[:topn_ctx], 1):
        head = f"[{idx}]"
        body = (text or "").replace("\r"," ").replace("\t"," ").strip()
        if len(body) > max_chars_per_item: body = body[:max_chars_per_item] + "…"
        block = head + "\n" + body
        if total + len(block) > max_total_chars: break
        blocks.append(block); total += len(block)
    return "\n\n".join(blocks)

def _build_bm25_hints_block(records: List[Dict[str,Any]], bm25_hits: List[Dict[str,Any]], max_chars_per_item: int = 160) -> str:
    lines = []
    for i, h in enumerate(bm25_hits, 1):
        r = records[h["rid"]]
        q, a = _extract_q_and_a(r)
        txt = (q or a or _extract_text(r)).replace("\n", " ")
        if len(txt) > max_chars_per_item: txt = txt[:max_chars_per_item] + "…"
        lines.append(f"[B{i}] {txt}")
    return "\n".join(lines)

def _build_bm25_only_block(records: List[Dict[str, Any]], bm25_hits: List[Dict[str, Any]], topn: int = 3, max_chars: int = 700) -> str:
    blocks = []
    for i, h in enumerate(bm25_hits[:topn], 1):
        r = records[h["rid"]]
        q, a = _extract_q_and_a(r)
        txt = ("Q: " + (q or "") + ("\nA: " + a if a else "")).strip() if (q or a) else (_extract_text(r) or "").strip()
        if len(txt) > max_chars: txt = txt[:max_chars] + "…"
        blocks.append(f"[B{i}]\n{txt}")
    return "\n\n".join(blocks)

# ================= Main =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-root", type=str, default="embed_data")
    ap.add_argument("--channels", nargs="+", default=["dialogue","lesson"])

    # qa600.json (BM25-only)
    ap.add_argument("--qa600-path", type=str, default="data/qa600.json")
    ap.add_argument("--qa600-topn", type=int, default=3)
    ap.add_argument("--qa600-thr", type=float, default=0.9)

    # Semantic (embed)
    ap.add_argument("--model-id", type=str, default=os.getenv("EMB_MODEL","AITeamVN/Vietnamese_Embedding"))
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu"])

    # Retrieval (ALL = embed corpus)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--bm25-topn", type=int, default=3)
    ap.add_argument("--bm25-k1", type=float, default=1.5)
    ap.add_argument("--bm25-b", type=float, default=0.75)
    ap.add_argument("--sem-limit", type=int, default=4)

    # LLM
    ap.add_argument("--llm-off", action="store_true")
    # Ollama model id (or HF local dir if using local-hf)
    # Mặc định dùng model Ollama đã tạo: phogpt4b-ft (PhoGPT base + LoRA adapter GGUF)
    ap.add_argument("--llm-model-id", type=str, default="phogpt4b-ft-v3")
    ap.add_argument("--llm-temp", type=float, default=float(os.getenv("LLM_TEMP","0.3")))  # Tăng từ 0.2
    ap.add_argument("--llm-ollama-host", type=str, default=os.getenv("LLM_OLLAMA_HOST","http://localhost:11434"))
    # Tăng mặc định để sinh câu trả lời dài hơn (có thể override bằng env hoặc CLI)
    ap.add_argument("--llm-max-new", type=int, default=int(os.getenv("LLM_MAX_NEW","1536")))

    # LLM Provider (Ollama-only)
    ap.add_argument("--llm-provider", type=str, default="ollama",
                    choices=["ollama"])

    ap.add_argument("--session-id", type=str, default="cli",
                help="ID phiên chat để lưu/đọc memory (file JSONL)")
    ap.add_argument("--memory-max-turns", type=int, default=4,
                    help="Số lượt hội thoại gần nhất đưa vào prompt")
    ap.add_argument("--memory-max-chars", type=int, default=1200,
                    help="Giới hạn ký tự history nhúng vào prompt")
    
    # Simple Prompting Options
    ap.add_argument("--enable-simple-prompting", action="store_true", default=True,
                    help="Bật simple prompting system (default: True)")
    ap.add_argument("--prompt-temperature", type=float, default=0.3,
                    help="Temperature cho prompting system")
    ap.add_argument("--enable-answer-preface", action="store_true", default=True,
                    help="Thêm câu mở đầu lịch sự trước câu trả lời")

    # Context
    ap.add_argument("--ctx-topn", type=int, default=3)
    ap.add_argument("--ctx-max-chars", type=int, default=16000)
    ap.add_argument("--ctx-join-window", type=int, default=0)
    ap.add_argument("--ctx-item-max-chars", type=int, default=2000,
                    help="Giới hạn ký tự mỗi đoạn context")

    # Thresholds
    ap.add_argument("--thr-warn", type=float, default=0.5)

    # Query gate
    ap.add_argument("--min-query-chars", type=int, default=4)
    ap.add_argument("--min-keyword-len", type=int, default=2)

    # Debug
    ap.add_argument("--no-print-prompt", action="store_true")
    ap.add_argument("--max-preview-len", type=int, default=180)

    # One shot
    ap.add_argument("--once", type=str, default=None)

    args = ap.parse_args()

    print(f"[INFO] LLM Provider: {args.llm_provider}")
    print(f"[INFO] LLM Model ID: {args.llm_model_id}")
    print(f"[INFO] Simple Prompting: {'✅ Enabled' if args.enable_simple_prompting else '❌ Disabled'}")

    # 1) Load embedded corpus (semantic)
    index, records = load_corpus(root=args.embed_root, channels=tuple(args.channels))
    sem = SemanticSearcher(index, model_id=args.model_id, device=args.device)

    # 2) BM25 over ALL embedded records
    bm25_docs_all = []
    for r in records:
        q, a = _extract_q_and_a(r)
        bm25_docs_all.append("\n".join([x for x in [q, a] if x]) if (q or a) else _extract_text(r))
    bm25_all = BM25(bm25_docs_all, k1=args.bm25_k1, b=args.bm25_b)

    # 3) Load qa600.json (BM25-only)
    qa_records, qa_docs = [], []
    if args.qa600_path and os.path.isfile(args.qa600_path):
        try:
            with open(args.qa600_path, "r", encoding="utf-8") as f:
                items = json.load(f)
            if isinstance(items, dict): items = [items]
            for it in items:
                q = (it.get("question") or "").strip()
                a = (it.get("answer")   or it.get("answer_text") or "").strip()
                if not (q or a): continue
                qa_records.append({
                    "text": (q + ("\n" + a if a else "")).strip(),
                    "qa": {"question": q or None, "answer": a or None},
                    "meta": {"order": it.get("order"), "importance": it.get("importance", False)},
                    "file": os.path.basename(args.qa600_path),
                    "channel": "qa600",
                })
                qa_docs.append("\n".join([x for x in [q, a] if x]))
            print(f"[OK] Loaded {len(qa_records)} QA from {args.qa600_path} (BM25-first).")
        except Exception as e:
            print(f"[WARN] Cannot read qa600 at {args.qa600_path}: {e}")
    else:
        print(f"[WARN] qa600 not found at: {args.qa600_path}")
    bm25_qa = BM25(qa_docs, k1=args.bm25_k1, b=args.bm25_b) if qa_docs else None

    # 4) LLM
    llm = None

    if not args.llm_off:
        print(f"[INFO] Initializing LLM with provider: {args.llm_provider}")
        llm = build_llm_from_args(args)

    def run_query(q: str):
        # Preprocess user input (normalize spacing, units, remove special chars)
        orig_q = (q or "").strip()
        q = clean_user_input(orig_q)
        if len(q) < args.min_query_chars:
            print(_box("CẢNH BÁO", f"Câu hỏi quá ngắn (< {args.min_query_chars} ký tự)."))
            return

        # Resolve follow-up using last focus (if any) BEFORE searching
        # try:
        #     q_resolved = resolve_after_preprocess(q, args.session_id)
        # except Exception:
        #     q_resolved = q
        q_resolved = q  # Tắt memory tracking
        q_proc = clean_text(q_resolved)
        if not q_proc or len(q_proc) < 2:
            print(_box("CẢNH BÁO", "Câu hỏi không hợp lệ sau khi xử lý."))
            return

        # Show both original and post-processed input for transparency
        display = f"Nguyên gốc: {orig_q}\n\nSau tiền xử lý: {q}"
        print(_box("CÂU HỎI", display))

        # ---------- 1) BM25 trên QA600 trước ----------
        if bm25_qa is not None:
            qa_hits = bm25_qa.search(q_proc, topk=max(1, args.qa600_topn), min_keyword_len=args.min_keyword_len)
            _print_hits("KẾT QUẢ BM25 (qa600)", qa_records, qa_hits)
            if qa_hits and qa_hits[0]["score"] >= args.qa600_thr:
                top = qa_hits[0]; rec = qa_records[top["rid"]]
                ans = (rec.get("qa") or {}).get("answer") or _extract_text(rec) or ""

                print(_box("TRẢ LỜI (CHẮC CHẮN)", _clean_answer(ans))); print(); return

        # ---------- 2) Semantic + BM25(all) ----------
        # Dùng câu hỏi đã resolve (giữ dấu câu, ngữ cảnh) cho SEMANTIC để tăng độ giống
        se_hits = sem.search(q_resolved, topk=args.topk)
        # Prioritize semantic hits coming from `lesson` channel.
        try:
            lesson_hits = [h for h in se_hits if (records[h["rid"]].get("channel") == "lesson")]
            other_hits = [h for h in se_hits if (records[h["rid"]].get("channel") != "lesson")]
            # Preserve original ordering within each group (already by score desc from search)
            se_hits = lesson_hits + other_hits
        except Exception:
            # If anything goes wrong, keep original se_hits
            pass

        # Small lexical boost: if exact phrase (cleaned) appears inside record text (cleaned), bump slightly
        try:
            q_clean_for_boost = clean_text(orig_q)
            if q_clean_for_boost and len(q_clean_for_boost) >= 8:
                boosted = []
                for h in se_hits:
                    rtxt = clean_text(records[h["rid"]].get("text") or _extract_text(records[h["rid"]]) or "")
                    score = h.get("score", 0.0)
                    if rtxt and q_clean_for_boost in rtxt:
                        score = min(1.0, float(score) + 0.06)
                    boosted.append({**h, "score": float(score)})
                se_hits = boosted
        except Exception:
            pass
        bm_hits = bm25_all.search(q_proc, topk=args.bm25_topn, min_keyword_len=args.min_keyword_len)
        mx = max([h["score"] for h in se_hits], default=0.0)

        # Update memory focus if semantic confidence is high enough
        # try:
        #     update_focus_after_sem(q_resolved, args.session_id, sem_max=mx, threshold=args.thr_warn)
        # except Exception:
        #     pass

        print(_box("TÍN HIỆU TIN CẬY", f"SEM_max={mx:.2f}"))
        _print_hits("KẾT QUẢ SEMANTIC", records, se_hits)
        _print_hits("KẾT QUẢ BM25 ", records, bm_hits)

        if args.llm_off:
            print(_box("LLM", "LLM đang tắt (--llm-off).")); print(); return

        # --- semantic >= 0.5 => chắc chắn ---
        if mx >= 0.5 and llm:
            # Chỉ dùng SEMANTIC hits, không RRF fusion
            se_sorted = sorted(se_hits, key=lambda x: -x["score"])
            se_dedup = dedup_by_source(records, se_sorted, max_per_source=2)
            se_def_first = sort_by_definition_first(records, se_dedup)
            ctx_ids = se_def_first[:args.sem_limit]
            
            # Tạo context chính từ semantic hits
            if args.ctx_join_window > 0:
                semantic_blocks = join_adjacent_chunks(records, ctx_ids, window=args.ctx_join_window, max_chars=args.ctx_item_max_chars)
            else:
                semantic_blocks = [(records[h["rid"]].get("text") or "")[:args.ctx_item_max_chars] for h in ctx_ids]

            main_context = _build_context_block_from_hits(semantic_blocks, args.ctx_topn, args.ctx_item_max_chars, args.ctx_max_chars)

            # BM25 chỉ làm context bổ sung (hints)
            bm25_context = ""
            if bm_hits:
                hints = _build_bm25_hints_block(records, bm_hits[:1], max_chars_per_item=400)
                if hints.strip():
                    bm25_context = f"\n\n[THAM KHẢO BM25]\n{hints}"

            # Display context combines semantic + BM25 hints, but LLM should only see semantic
            display_context = main_context + bm25_context
            final_context_for_llm = main_context

            if not args.no_print_prompt:
                print(_box("CONTEXT (SEMANTIC + BM25 HINTS)", display_context or "(rỗng)"))

            try:
                if args.enable_simple_prompting:
                    answer = simple_generate_answer(
                        llm, q_resolved, final_context_for_llm,
                        max_new_tokens=args.llm_max_new,
                        temperature=args.prompt_temperature,
                        max_prompt_chars=min(max(4000, args.ctx_max_chars), 16000)
                    )
                else:
                    answer = generate_answer(llm, q_resolved, final_context_for_llm, max_new_tokens=args.llm_max_new, temperature=args.llm_temp)
            except Exception as e:
                print(_box("TRẢ LỜI", f"[LLM ERROR] {e}"));
                import traceback; traceback.print_exc()
                return
            # Enrich short answers and optionally add a polite preface
            answer_final = _enrich_short_answer((answer or "").strip(), display_context)
            answer_final = _maybe_with_preface(answer_final, orig_q, args.enable_answer_preface)
            print(_box("TRẢ LỜI (CHẮC CHẮN)", (_clean_answer(answer_final) or "Thông tin chưa đủ."))); print(); return

        # --- semantic < 0.5 ---
        if not bm_hits:
            print(_box("KẾT LUẬN", "Không tìm thấy căn cứ đủ mạnh và không có gợi ý từ BM25.")); print(); return

        # Prefer BM25 suggestions from the `dialogue` channel (user request).
        # If no dialogue hits are available, fall back to the full BM25 results.
        try:
            bm_dialogue_hits = [h for h in bm_hits if records[h["rid"]].get("channel") == "dialogue"]
        except Exception:
            bm_dialogue_hits = []

        chosen_bm_hits = bm_dialogue_hits if bm_dialogue_hits else bm_hits

        bm25_block = _build_bm25_only_block(records, chosen_bm_hits, topn=3, max_chars=700)
        print(_box(f"CÂU HỎI GỢI Ý", bm25_block or "(rỗng)"))

    # Run once / REPL
    if args.once:
        run_query(args.once); return

    print(_box("HƯỚNG DẪN", "Nhập câu hỏi về lý thuyết lái xe (Enter để thoát). Ví dụ: 'Biển P.130 là gì?'"))
    while True:
        try:
            q = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not q: break
        run_query(q)

if __name__ == "__main__":
    main()

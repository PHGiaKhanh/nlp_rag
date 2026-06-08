from __future__ import annotations
from typing import Dict, Any, Callable, List, Optional, Tuple
import unicodedata, re

from services import bm25_service, embed_and_semantic_service, compare_semantic_service
from services import load_data_BM25_mock, load_data_semantic_mock, preprocess_data_service
from services.preprocess_data_service import normalize_text as _norm_text


# LLM caller injection (user can set their own)
_LLM_CALLER: Optional[Callable[[str], str]] = None

def set_llm_caller(func: Callable[[str], str]) -> None:
    """Set a synchronous LLM caller: func(prompt)->str"""
    global _LLM_CALLER
    _LLM_CALLER = func


# ================= Helper functions from main.py =================
_word = re.compile(r"\w+", flags=re.UNICODE)

def _tok(s: str) -> List[str]:
    if not s: return []
    return [w.lower() for w in _word.findall(s)]

def clean_text(s: str) -> str:
    """Clean text for comparison (align with main.py)"""
    if not s: return ""
    s = unicodedata.normalize("NFC", s).lower()
    s = re.sub(r"[^a-z0-9\u00c0-\u024f\u1e00-\u1eff\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _extract_q_and_a(r: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Extract question and answer from record"""
    qa = r.get("qa") or {}
    q = qa.get("question") or r.get("question")
    a = qa.get("answer")   or r.get("answer")
    q = q.strip() if isinstance(q, str) and q.strip() else None
    a = a.strip() if isinstance(a, str) and a.strip() else None
    return q, a

def _extract_text(r: Dict[str, Any]) -> str:
    """Extract text content from record"""
    for key in ["text","content","search_text","body","chunk_text","paragraph"]:
        v = r.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    q, a = _extract_q_and_a(r)
    if q or a: return ("Q: " + (q or "") + (f"\nA: {a}" if a else ""))
    return ""

def dedup_by_source(records: List[Dict[str, Any]], hits: List[Dict[str, Any]], max_per_source: int = 1) -> List[Dict[str, Any]]:
    """Deduplicate hits by source (channel + file)"""
    seen, kept = {}, []
    for h in hits:
        r = records[h["rid"]]; key = f"{r.get('channel','?')}::{r.get('file','?')}"
        if seen.get(key, 0) < max_per_source:
            kept.append(h); seen[key] = seen.get(key, 0) + 1
    return kept

_DEF_PAT = re.compile(r"\b(là gì|định nghĩa|ý nghĩa|ký hiệu|đáp án)\b", re.IGNORECASE)

def sort_by_definition_first(records: List[Dict[str, Any]], hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prioritize hits containing definition keywords"""
    def score(h): return 1 if _DEF_PAT.search((records[h["rid"]].get("text") or "").lower()) else 0
    return sorted(hits, key=score, reverse=True)

def join_adjacent_chunks(records: List[Dict[str, Any]], hits: List[Dict[str, Any]], window: int = 1, max_chars: int = 1200) -> List[str]:
    """Join adjacent chunks from same file to provide more context"""
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


def _default_build_prompt(context_block: List[Dict[str, Any]], bm25_hints: List[Dict[str, Any]], query: str) -> str:
    parts: List[str] = []
    parts.append("Context:")
    for i, it in enumerate(context_block, start=1):
        parts.append(f"{i}. {it.get('text','')}")
    parts.append("BM25 Hints:")
    for i, it in enumerate(bm25_hints, start=1):
        parts.append(f"{i}. Q: {it.get('ques','')} A: {it.get('ans','')}")
    parts.append("User Query:")
    parts.append(query)
    parts.append("Answer:")
    return "\n".join(parts)


def run(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main pipeline entry.
    input_data: {"text": str, "top_k"?: int}
    returns dict with status and relevant payload.
    """
    text = (input_data or {}).get("text", "") or ""
    if not text.strip():
        return {"status": "empty_query"}

    top_k = int((input_data or {}).get("top_k", 5))
    if top_k <= 0:
        top_k = 5

    # preprocess the incoming text (use project's preprocess utility)
    try:
        prepped = preprocess_data_service.preprocess_text(text)
    except Exception:
        prepped = _norm_text(text)
    q = _norm_text(prepped)

    # If no semantic provider set, try to load local embedded corpus via services.load_data_semantic_mock
    try:
        if getattr(embed_and_semantic_service, "_CORPUS_PROVIDER", None) is None:
            mat, records_sem = load_data_semantic_mock.load_corpus()
            texts = [r.get("text") for r in records_sem]

            def _sem_provider_from_records():
                for t in texts:
                    yield t

            embed_and_semantic_service.set_corpus_provider(_sem_provider_from_records)
            # inject precomputed vectors into embed service so it doesn't re-embed
            try:
                vecs = [row.tolist() for row in mat]
                embed_and_semantic_service._CORPUS_TEXTS = texts
                embed_and_semantic_service._VECS = vecs
                embed_and_semantic_service._BUILT_FROM_TFIDF = True
            except Exception:
                pass
    except Exception:
        pass

    # If no BM25 provider set, try to load local data via services.load_data_BM25_mock
    try:
        if getattr(bm25_service, "_CORPUS_PROVIDER", None) is None:
            records, docs = load_data_BM25_mock.load_data_bm25()

            def _bm25_provider_from_records():
                for r in records:
                    qa = r.get("qa") or {}
                    q_ = qa.get("question") or None
                    a_ = qa.get("answer") or None
                    if q_ is not None:
                        yield q_, a_ or ""

            bm25_service.set_corpus_provider(_bm25_provider_from_records)
    except Exception:
        pass

    # 1) Semantic step first (per requested flow)
    sem_out = embed_and_semantic_service.run({"message": prepped, "top_k": top_k})
    sem_meta = sem_out.get("metadata", [])
    
    # Load records for advanced processing
    try:
        _, records_sem = load_data_semantic_mock.load_corpus()
    except Exception:
        records_sem = []
    
    # === Lesson channel priority (from main.py) ===
    # Prioritize semantic hits from "lesson" channel
    try:
        if records_sem and sem_meta:
            lesson_hits = [h for h in sem_meta if 
                          (records_sem[h.get("rid", -1)].get("channel") == "lesson" if 0 <= h.get("rid", -1) < len(records_sem) else False)]
            other_hits = [h for h in sem_meta if 
                         (records_sem[h.get("rid", -1)].get("channel") != "lesson" if 0 <= h.get("rid", -1) < len(records_sem) else True)]
            # Preserve original ordering within each group (already by score desc)
            sem_meta = lesson_hits + other_hits
    except Exception:
        pass  # Keep original sem_meta if anything goes wrong
    
    # === Lexical boost logic (from main.py) ===
    # Small lexical boost: if exact phrase (cleaned) appears inside record text (cleaned), bump slightly
    try:
        q_clean_for_boost = clean_text(text)
        if q_clean_for_boost and len(q_clean_for_boost) >= 8 and records_sem:
            boosted = []
            for h in sem_meta:
                rid = h.get("rid", -1)
                if 0 <= rid < len(records_sem):
                    rtxt = clean_text(records_sem[rid].get("text") or _extract_text(records_sem[rid]) or "")
                    score = h.get("thres", h.get("score", 0.0))
                    if rtxt and q_clean_for_boost in rtxt:
                        score = min(1.0, float(score) + 0.06)  # Boost +0.06
                    boosted.append({**h, "thres": float(score), "score": float(score)})
            sem_meta = boosted
    except Exception:
        pass  # Keep original sem_meta if boost fails
    
    sem_dec = compare_semantic_service.compare_semantic_service({"metadata": sem_meta, "thres": 0.0})
    sem_cmd = sem_dec.get("command_sematic")

    # 2) BM25 retrieval (we need BM25 hints for LLM path and also to check emptiness)
    # 2) BM25 retrieval (global hints)
    bm25_out = bm25_service.run({"text": q, "top_k": top_k})
    bm25_meta = bm25_out.get("metadata", [])

    # If no BM25 results at all -> no result
    if not bm25_meta:
        return {"status": "no_bm25_result", "bm25_meta": bm25_meta}

    # derive top BM25 normalized thres (0.0 if no candidates)
    try:
        top_bm25_thres = float(bm25_meta[0].get("thres", 0.0)) if bm25_meta else 0.0
    except Exception:
        top_bm25_thres = 0.0

    # 2a) Exam-specific BM25 using qa600 subset (align with retriever scoring)
    exam_top_thres = 0.0
    exam_top_answer: Optional[str] = None
    try:
        records_all, _ = load_data_BM25_mock.load_data_bm25()
        qa600_records = [r for r in records_all if r.get("channel") == "qa600"]
        if qa600_records:
            from services.bm25_service import BM25, tokenize, normalize_text

            docs_tokens = [tokenize(normalize_text(((r.get("qa") or {}).get("question") or ""))) for r in qa600_records]
            idx = BM25(docs_tokens) if docs_tokens else None
            if idx and idx.N > 0:
                # get raw scores for all docs in qa600 index
                ranked_all = idx.search(q, k=idx.N)
                if ranked_all:
                    # compute UB_equal for this query on the qa600 index (same formula as in bm25_service)
                    q_tokens = tokenize(normalize_text(q))
                    if q_tokens:
                        tf_q = {}
                        for t in q_tokens:
                            tf_q[t] = tf_q.get(t, 0) + 1
                        q_terms = list(dict.fromkeys(q_tokens))
                        dl_equal = sum(tf_q.values())
                        if idx.avgdl > 0:
                            K_equal = idx.k1 * (1.0 - idx.b + idx.b * (dl_equal / idx.avgdl))
                        else:
                            K_equal = idx.k1
                        UB_equal = 0.0
                        for t in q_terms:
                            idf_t = idx.idf.get(t, 0.0)
                            ft_q = tf_q.get(t, 0)
                            if ft_q + K_equal != 0:
                                UB_equal += idf_t * (ft_q * (idx.k1 + 1.0)) / (ft_q + K_equal)
                        if UB_equal <= 0.0:
                            UB_equal = 1e-9

                        # prepare coverage denominators
                        denom_idf = sum(idx.idf.get(t, 0.0) for t in q_terms) or 1e-9
                        qlen = max(len(q_terms), 1)

                        # compute normalized score aligned with retriever: ratio * coverage_final
                        best_i, best_z = None, 0.0
                        for (di, raw) in ranked_all:
                            # ratio part
                            ratio = min(float(raw) / UB_equal, 1.0)
                            # coverage part
                            try:
                                tf_doc = idx.tf[di]
                                matched_terms = sum(1 for t in q_terms if t in tf_doc)
                                num_idf = sum(idx.idf.get(t, 0.0) for t in q_terms if t in tf_doc)
                                coverage_idf = float(num_idf) / float(denom_idf) if denom_idf != 0 else 0.0
                                coverage_terms = float(matched_terms) / float(qlen)
                                coverage_final = 0.5 * coverage_idf + 0.5 * coverage_terms
                            except Exception:
                                coverage_final = 1.0  # fallback to ratio only
                            z = float(ratio * coverage_final)
                            if z > best_z:
                                best_z = z
                                best_i = di

                        if best_i is not None:
                            exam_top_thres = float(best_z)
                            exam_top_answer = (qa600_records[best_i].get("qa") or {}).get("answer")
    except Exception:
        exam_top_thres = 0.0

    # Decision mapping requested by user (final order):
    # 1) definite_exam_answer: if exam_top_thres >= 0.9 (qa600 check)
    # 2) definite_retrieve_answer: if exam_top_thres < 0.9 AND top_bm25_thres <= 0.9 AND sem_max >= 0.5
    # 3) suggest_top3_bm25: if sem_max < 0.5 AND top_bm25_thres > 0.0
    # 4) no_bm25_result: if sem_max < 0.5 AND top_bm25_thres == 0.0
    sem_max = float(sem_dec.get("thres", 0.0))

    # 1) exam fast path
    if exam_top_thres >= 0.9:
        # sanitize sem_meta and return only top-1 sem (strip internal keys)
        def _sanitize_item(it: dict) -> dict:
            it2 = dict(it)
            for k in ("rid", "raw", "matched_terms", "score_raw", "score", "coverage"):
                if k in it2:
                    del it2[k]
            return it2

        return {
            "status": "definite_exam_answer",
            "bm_result": exam_top_answer,
            "exam_thres": exam_top_thres,
        }

    # 2) definite_retrieve_answer (semantic confident enough)
    if exam_top_thres < 0.9 and top_bm25_thres <= 0.9 and sem_max >= 0.5:
        # Apply advanced processing: dedup, sort by definition, join chunks
        try:
            if records_sem and sem_meta:
                # Convert sem_meta to hit format for processing
                se_hits = [{"rid": h.get("rid", -1), "score": h.get("thres", h.get("score", 0.0))} for h in sem_meta]
                
                # Sort, dedup, and prioritize definition hits
                se_sorted = sorted(se_hits, key=lambda x: -x["score"])
                se_dedup = dedup_by_source(records_sem, se_sorted, max_per_source=2)
                se_def_first = sort_by_definition_first(records_sem, se_dedup)
                
                # Take top hits
                ctx_ids = se_def_first[:min(4, len(se_def_first))]
                
                # Join adjacent chunks for richer context
                semantic_blocks = join_adjacent_chunks(records_sem, ctx_ids, window=2, max_chars=2000)
                
                # Build context from joined blocks
                context_texts = []
                for idx, text in enumerate(semantic_blocks[:3], 1):
                    context_texts.append({"text": text, "idx": idx})
                context_block = context_texts
            else:
                context_block = sem_meta[:3]
        except Exception:
            # Fallback to original simple approach
            context_block = sem_meta[:3]
        
        bm25_hints = bm25_meta[:2]
        prompt = _default_build_prompt(context_block, bm25_hints, prepped)
        if _LLM_CALLER is None:
            return {
                "status": "definite_retrieve_answer",
                "reason": "no_llm_configured",
                "prompt": prompt,
                "sem_thres": sem_max,
            }
        answer = _LLM_CALLER(prompt)
        return {
            "status": "definite_retrieve_answer",
            "bm_result": answer,
            "sem_thres": sem_max,
        }

    # 3) suggest_top3_bm25 (semantic low confidence but BM25 has candidates)
    if sem_max < 0.5 and top_bm25_thres > 0.0:
        # supply top-3 BM25 suggestions and top1 as bm_result
        top3 = bm25_meta[:3]
        sanitized_top3 = []
        for it in top3:
            # strip internal keys including 'rid' so callers don't receive internal ids
            sanitized_top3.append({k: v for k, v in it.items() if k not in ("rid", "raw", "matched_terms", "score_raw", "score", "coverage")})
        return {
            "status": "suggest_top3_bm25",
            "suggestions": sanitized_top3
        
        }

    # 4) no_bm25_result (semantic low confidence and BM25 empty/zero)
    if sem_max < 0.5 and top_bm25_thres == 0.0:
        return {
            "status": "no_bm25_result",
            "bm_result": None,
                }

    # Fallback: return recommendations (safe default)
    return {
        "status": "suggest_top3_bm25",
        "suggestions": [{k: v for k, v in it.items() if k not in ("rid", "raw", "matched_terms", "score_raw", "score", "coverage")} for it in (bm25_meta[:3] or [])],
        "sem_thres": sem_max,
        "bm25_thres": top_bm25_thres,
        "exam_thres": exam_top_thres,
    }



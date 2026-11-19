"""
eval_rag.py - Evaluation harness for the PDF RAG Chatbot (Pdf_Chatbot)

Measures:
  - Retrieval quality : Hit@1, Hit@3, Hit@5, MRR@5
  - Retrieval latency : avg and p95 (ms)
  - Answer quality    : keyword match rate (rough proxy)
  - LLM latency       : avg and p95 (s)
  - Cost proxy        : avg tokens per query (from Groq usage)

Pipeline mirrors app.py exactly: PyMuPDF -> fixed-size char chunks (400 / 80
overlap, chunks <= 50 chars dropped) -> all-MiniLM-L6-v2 -> FAISS IndexFlatL2
-> top-3 chunks -> Groq llama-3.1-8b-instant via the same REST call and prompt.

Developer: Suraj Kumar (surajkumar805122@gmail.com)

Usage:
  python eval_rag.py --pdf sample.pdf --questions eval_questions.json
  python eval_rag.py --pdf sample.pdf --questions eval_questions.json --no-llm
  python eval_rag.py --pdf sample.pdf --questions eval_questions.json --chunk-size 800 --overlap 150
"""

import argparse
import csv
import json
import os
import re
import statistics
import time
from pathlib import Path

import faiss
import fitz  # PyMuPDF
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# ---- Defaults copied from app.py ----
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
TOP_K = 3
MAX_CONTEXT_CHARS = 2500
TEMPERATURE = 0.3
MAX_TOKENS = 512

# Same system prompt as app.py (context goes inside the system message)
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer questions based on the document context below. "
    "If the answer is not in the context, say so.\n\n"
    "CONTEXT:\n{context}"
)
CONTEXT_SEPARATOR = "\n\n---\n\n"


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace so PDF line breaks don't break matching."""
    return re.sub(r"\s+", " ", text).strip().lower()


def load_pdf_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    chunks, start = [], 0
    step = max(1, size - overlap)
    while start < len(text):
        chunks.append(text[start:start + size].strip())
        start += step
    return [c for c in chunks if len(c) > 50]  # same filter as app.py


def build_index(chunks: list[str], model: SentenceTransformer):
    emb = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    emb = emb.astype("float32")
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)
    return index


def retrieve(question: str, model, index, k: int):
    q = model.encode([question], convert_to_numpy=True).astype("float32")
    _, idx = index.search(q, k)
    return [int(i) for i in idx[0] if i != -1]


def first_hit_rank(retrieved_ids, chunks, evidence: str):
    """1-based rank of the first chunk containing the evidence phrase, else None."""
    ev = normalize(evidence)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if ev in normalize(chunks[cid]):
            return rank
    return None


def ask_llm(api_key: str, question: str, context: str):
    """Same request as ask_groq() in app.py (no chat history, single question)."""
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": question},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    headers = {"Authorization": f"Bearer {api_key.strip()}", "Content-Type": "application/json"}
    t0 = time.perf_counter()
    resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
    latency = time.perf_counter() - t0
    if resp.status_code != 200:
        raise RuntimeError(f"Groq Error {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    answer = data["choices"][0]["message"]["content"] or ""
    tokens = data.get("usage", {}).get("total_tokens")
    return answer, latency, tokens


def keyword_match(answer: str, keywords: list[str]) -> bool:
    """True if every expected keyword appears in the answer (case-insensitive)."""
    if not keywords:
        return False
    a = normalize(answer)
    return all(normalize(k) in a for k in keywords)


def p95(values):
    if not values:
        return None
    return float(np.percentile(values, 95))


def main():
    ap = argparse.ArgumentParser(description="Evaluate the PDF RAG chatbot.")
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    ap.add_argument("--top-k", type=int, default=TOP_K, help="chunks sent to the LLM")
    ap.add_argument("--no-llm", action="store_true", help="only evaluate retrieval")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    print(f"Loading PDF: {args.pdf}")
    text = load_pdf_text(args.pdf)
    chunks = chunk_text(text, args.chunk_size, args.overlap)
    print(f"{len(chunks)} chunks (size={args.chunk_size}, overlap={args.overlap})")

    model = SentenceTransformer(EMBED_MODEL)
    t0 = time.perf_counter()
    index = build_index(chunks, model)
    index_time = time.perf_counter() - t0

    # Warm-up so the first query's latency isn't inflated
    retrieve("warm up", model, index, 1)

    client = None  # holds the API key when the LLM part is enabled
    if not args.no_llm:
        client = os.environ.get("GROQ_API_KEY")
        if not client:
            raise SystemExit("Set GROQ_API_KEY first, or run with --no-llm.")

    rows, ret_lat, llm_lat, tokens_used = [], [], [], []
    hits = {1: 0, 3: 0, 5: 0}
    rr_sum, kw_hits, kw_total = 0.0, 0, 0
    eval_k = max(5, args.top_k)

    for i, q in enumerate(questions, start=1):
        t = time.perf_counter()
        ids = retrieve(q["question"], model, index, eval_k)
        ret_ms = (time.perf_counter() - t) * 1000
        ret_lat.append(ret_ms)

        rank = first_hit_rank(ids, chunks, q.get("evidence", ""))
        for k in hits:
            if rank is not None and rank <= k:
                hits[k] += 1
        if rank is not None and rank <= 5:
            rr_sum += 1.0 / rank

        answer, llm_s, tok, kw_ok = "", None, None, None
        if client:
            context = CONTEXT_SEPARATOR.join(chunks[c] for c in ids[:args.top_k])[:MAX_CONTEXT_CHARS]
            try:
                answer, llm_s, tok = ask_llm(client, q["question"], context)
                llm_lat.append(llm_s)
                if tok:
                    tokens_used.append(tok)
            except Exception as e:  # keep going if one call fails
                answer = f"ERROR: {e}"
            time.sleep(2)  # stay under Groq free-tier rate limits
            if q.get("answer_keywords"):
                kw_total += 1
                kw_ok = keyword_match(answer, q["answer_keywords"])
                kw_hits += int(kw_ok)

        rows.append({
            "id": i,
            "question": q["question"],
            "hit_rank": rank if rank else "",
            "retrieval_ms": round(ret_ms, 1),
            "llm_s": round(llm_s, 2) if llm_s else "",
            "tokens": tok or "",
            "keywords_ok": "" if kw_ok is None else kw_ok,
            "answer": answer.replace("\n", " ")[:300],
        })
        status = f"hit@{rank}" if rank else "MISS"
        print(f"[{i:02d}/{len(questions)}] {status:7s} {q['question'][:60]}")

    n = len(questions)
    summary = {
        "questions": n,
        "chunks": len(chunks),
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "index_build_s": round(index_time, 2),
        "hit@1": round(hits[1] / n, 3),
        "hit@3": round(hits[3] / n, 3),
        "hit@5": round(hits[5] / n, 3),
        "mrr@5": round(rr_sum / n, 3),
        "retrieval_ms_avg": round(statistics.mean(ret_lat), 1),
        "retrieval_ms_p95": round(p95(ret_lat), 1),
    }
    if client:
        summary.update({
            "llm_s_avg": round(statistics.mean(llm_lat), 2) if llm_lat else None,
            "llm_s_p95": round(p95(llm_lat), 2) if llm_lat else None,
            "tokens_avg": round(statistics.mean(tokens_used)) if tokens_used else None,
            "keyword_match": round(kw_hits / kw_total, 3) if kw_total else None,
        })

    tag = f"c{args.chunk_size}_o{args.overlap}"
    with open(out_dir / f"eval_results_{tag}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))

    md = ["| Metric | Value |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in summary.items()]
    (out_dir / f"summary_{tag}.md").write_text("\n".join(md) + "\n")

    print("\n===== SUMMARY =====")
    for k, v in summary.items():
        print(f"{k:18s} {v}")
    print(f"\nSaved to {out_dir}/ (csv, json, md)")


if __name__ == "__main__":
    main()

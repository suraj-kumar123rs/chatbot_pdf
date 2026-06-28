import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import faiss
import os
import requests
import json
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# GROQ_MODEL   = "llama3-8b-8192"
GROQ_MODEL   = "llama-3.1-8b-instant"
CHUNK_SIZE   = 400
CHUNK_OVERLAP = 80
TOP_K        = 3

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄", layout="wide")

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_pdf(pdf_file) -> str:
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def chunk_text(text: str) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end].strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c) > 50]


def build_faiss_index(chunks, model):
    embeddings = model.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def retrieve_chunks(query, chunks, index, model, top_k=TOP_K):
    q_emb = model.encode([query], show_progress_bar=False)
    q_emb = np.array(q_emb, dtype="float32")
    distances, indices = index.search(q_emb, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


def ask_groq(api_key, context, question, history):
    context = context[:2500]

    system_prompt = (
        "You are a helpful assistant. Answer questions based on the document context below. "
        "If the answer is not in the context, say so.\n\n"
        f"CONTEXT:\n{context}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    for h in history[-2:]:
        messages.append(h)
    messages.append({"role": "user", "content": question})

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 512
    }

    resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)

    if resp.status_code != 200:
        try:
            err = resp.json()
            raise Exception(f"Groq Error {resp.status_code}: {err}")
        except Exception as e:
            raise Exception(f"Groq Error {resp.status_code}: {resp.text}") from e

    return resp.json()["choices"][0]["message"]["content"]


# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
for key, default in {
    "chunks": [],
    "faiss_index": None,
    "chat_history": [],
    "docs_ready": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.title("Setup")
    st.markdown("**Step 1 — Groq API Key**")
    st.markdown("Get free key → [console.groq.com](https://console.groq.com)")
    api_key = st.text_input("Paste Groq API Key", type="password", placeholder="gsk_...")

    st.markdown("---")
    st.markdown("**Step 2 — Upload PDFs**")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    process_btn = st.button("Process PDFs", use_container_width=True, type="primary")

    if process_btn:
        if not api_key:
            st.error("Enter your Groq API key first!")
        elif not uploaded_files:
            st.error("Upload at least one PDF!")
        else:
            with st.spinner("Processing..."):
                model = load_embedding_model()
                all_chunks = []
                for f in uploaded_files:
                    text = extract_text_from_pdf(f)
                    all_chunks.extend(chunk_text(text))

                index = build_faiss_index(all_chunks, model)
                st.session_state.chunks = all_chunks
                st.session_state.faiss_index = index
                st.session_state.docs_ready = True
                st.session_state.chat_history = []
            st.success(f"Ready! {len(all_chunks)} chunks from {len(uploaded_files)} PDF(s).")

    if st.session_state.docs_ready:
        st.markdown("---")
        if st.button("Clear & Reset", use_container_width=True):
            st.session_state.chunks = []
            st.session_state.faiss_index = None
            st.session_state.chat_history = []
            st.session_state.docs_ready = False
            st.rerun()

    st.markdown("---")
    st.caption("Model: llama3-8b (Groq) | Embeddings: MiniLM-L6 | VectorDB: FAISS")


# ─────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────
st.title("📄 PDF RAG Chatbot")
st.caption("Upload PDFs → Ask anything → Get answers from your documents")

if not st.session_state.docs_ready:
    st.info("Upload your PDFs and click **Process PDFs** from the sidebar to start.")
else:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask a question about your documents...")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    model = load_embedding_model()
                    relevant = retrieve_chunks(
                        user_input,
                        st.session_state.chunks,
                        st.session_state.faiss_index,
                        model
                    )
                    context = "\n\n---\n\n".join(relevant)
                    answer = ask_groq(api_key, context, user_input, st.session_state.chat_history)
                    st.write(answer)
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(str(e))
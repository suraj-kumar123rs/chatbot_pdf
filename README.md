# PDF RAG Chatbot

Multi-PDF Q&A chatbot using RAG (Retrieval-Augmented Generation).
Free to run — uses Groq API (free tier) + local embeddings.

## Tech Stack
- **PyMuPDF** — PDF text extraction
- **sentence-transformers** (all-MiniLM-L6-v2) — local embeddings, no API needed
- **FAISS** — vector similarity search (local)
- **Groq API** (llama3-8b) — free LLM for answer generation
- **Streamlit** — web UI

## Setup (Windows 11 + VS Code)

### Step 1 — Get free Groq API key
1. Go to https://console.groq.com
2. Sign up (free, no credit card)
3. Go to API Keys → Create API Key
4. Copy the key (starts with `gsk_`)

### Step 2 — Create project folder & open in VS Code
```
mkdir pdf-rag-chatbot
cd pdf-rag-chatbot
```
Copy all files here (app.py, requirements.txt)

### Step 3 — Create virtual environment
Open VS Code terminal (Ctrl + `) and run:
```bash
python -m venv venv
venv\Scripts\activate
```
You should see `(venv)` in your terminal.

### Step 4 — Install dependencies
```bash
pip install -r requirements.txt
```
First time takes 2-3 min (downloads embedding model ~90MB).

### Step 5 — Run the app
```bash
streamlit run app.py
```
Browser opens automatically at http://localhost:8501

## How to use
1. Paste your Groq API key in the sidebar
2. Upload one or more PDFs
3. Click "Process PDFs"
4. Start chatting!

## Project Structure
```
pdf-rag-chatbot/
├── app.py              # Main application
├── requirements.txt    # Dependencies
└── README.md
```

## How RAG works (for interviews)
1. PDF text extracted → split into overlapping chunks
2. Each chunk embedded into a vector (semantic meaning captured)
3. All vectors stored in FAISS index
4. User asks question → question embedded → top-4 similar chunks retrieved
5. Retrieved chunks sent as context to LLM → grounded answer generated

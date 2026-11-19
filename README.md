# PDF RAG Chatbot

Multi-PDF question answering with Retrieval-Augmented Generation. Upload PDFs, ask questions, get answers grounded in the documents. Runs on the Groq free tier with local embeddings.

Developed by **Suraj Kumar** · surajkumar805122@gmail.com

## Tech stack
- **PyMuPDF**: PDF text extraction
- **sentence-transformers** (all-MiniLM-L6-v2): local embeddings
- **FAISS**: dense vector search
- **rank_bm25**: keyword search
- **Groq API** (openai/gpt-oss-20b): answer generation
- **Streamlit**: web UI

## How it works
1. Text is extracted from each PDF and split into 400-character chunks with 80 characters of overlap.
2. Each chunk is embedded with MiniLM and stored in a FAISS index. A BM25 index is built over the same chunks.
3. For a question, the top 20 candidates from FAISS and the top 20 from BM25 are merged with reciprocal rank fusion, and the best 3 chunks are kept.
4. Those chunks (capped at 2,500 characters) are sent to the LLM as context, along with the last two chat messages.

## Evaluation
`eval_rag.py` measures retrieval quality, latency and token cost on a fixed question set. The set in `eval_questions.json` has 20 curated factual questions about the original RAG paper ([Lewis et al., 2020](https://arxiv.org/abs/2005.11401)). Each question is tied to an evidence phrase from the paper, and a hit means a retrieved chunk contains that phrase.

**Chunk size** (dense retrieval)

| Config | Hit@1 | Hit@3 | MRR@5 | Retrieval p95 | LLM avg | Tokens/query | Keyword match |
|---|---|---|---|---|---|---|---|
| 400 / 80 | 0.45 | 0.45 | 0.46 | 26.9 ms | 0.37 s | 519 | 0.55 |
| 800 / 150 | 0.40 | 0.45 | 0.454 | 31.0 ms | 0.44 s | 879 | 0.35 |

Larger chunks did not improve retrieval, used 69% more tokens per query, and gave fewer correct answers, so the app keeps 400 / 80.

**Retrieval method** (400 / 80 chunks)

| Retrieval | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
|---|---|---|---|---|
| Dense only | 0.45 | 0.45 | 0.50 | 0.46 |
| Hybrid BM25 + dense (RRF) | 0.55 | 0.65 | 0.70 | 0.604 |

Dense embeddings missed questions that hinge on exact technical terms (FAISS, DPR, BLEU). Adding BM25 recovered several of them, so the app now uses hybrid retrieval.

Notes: 20 questions is a small set, so treat these as directional numbers. Keyword match is a rough proxy for answer correctness.

Run it yourself:
```bash
python eval_rag.py --pdf paper.pdf --questions eval_questions.json            # dense
python eval_rag.py --pdf paper.pdf --questions eval_questions.json --hybrid   # hybrid
python eval_rag.py --pdf paper.pdf --questions eval_questions.json --no-llm   # retrieval only
```
Results are written to `results/` as CSV, JSON and a Markdown table.

## Setup (Windows + VS Code)
1. Get a free Groq API key at https://console.groq.com (API Keys → Create API Key).
2. Create a `.env` file in the project folder:
   ```
   GROQ_API_KEY=your_key_here
   ```
   On Streamlit Cloud, add the same key under the app's **Secrets** instead.
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## How to use
1. Upload one or more PDFs in the sidebar.
2. Click **Process PDFs**.
3. Ask questions in the chat box.

## Project structure
```
chatbot_pdf/
├── app.py                # Streamlit app
├── eval_rag.py           # Evaluation harness
├── eval_questions.json   # 20 curated eval questions
├── requirements.txt
└── README.md
```

# ResearchGPT 🔬

A Retrieval-Augmented Generation (RAG) system for querying scientific research papers using natural language. Powered by **Gemini** for both embeddings and generation, with **FAISS** for fast vector search.

Made by CHIKI & AKSHYAT

---

## Architecture

```
User Question
     │
     ▼
[React Frontend]
     │  POST /load  (first time)
     │  POST /upload
     │  POST /query
     ▼
[FastAPI Backend]
     │
     ├─ /load    → fetch PDF → extract text → chunk → Gemini Embed → FAISS index
     ├─ /upload  → accept PDF upload → same pipeline
     │
     └─ /query   → embed question → FAISS similarity search → top-K chunks → Gemini LLM → answer
```

### Key design choices

| Component | Choice | Why |
|-----------|--------|-----|
| Embeddings | `gemini-embedding-001` (Gemini) | No local model needed; high quality semantic search |
| Vector DB | FAISS `IndexFlatIP` (in-memory) | Zero infra, fast cosine search via inner product |
| LLM | `gemini-2.5-flash-lite` | Fast, cheap, long context for grounded answers |
| Chunking | Section-split → sliding window (1200 chars, 100 overlap) | Preserves semantic sections + handles long sections |
| Paper fetch | Direct PDF download from arxiv | No scraping needed; arxiv allows programmatic access |

---

## Setup

### Prerequisites
- Python 3.10+
- A free [Gemini API key](https://aistudio.google.com/app/apikey)

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env and set your GEMINI_API_KEY
```

### 2. Backend

```bash
pip install -r requirements.txt
./start.sh
# Server runs at http://localhost:8000
```

### 3. Frontend (React)

`ResearchGPT.jsx` is a self-contained React component. Use it in any React app:

```bash
# Option A: copy into your existing React project
cp ResearchGPT.jsx src/App.jsx

# Option B: use with Vite
npm create vite@latest my-app -- --template react
cd my-app && cp ../ResearchGPT.jsx src/App.jsx
npm install && npm run dev
```

Set `VITE_API_URL` in a `.env` file if your backend is not on `http://localhost:8000`.

---

## API Reference

The API key is configured server-side via `.env` — clients do not send it.

### `POST /load`
Download, parse, chunk, and index a PDF from a URL.

```json
Request:  { "url": "https://arxiv.org/abs/1706.03762" }
Response: { "paper_id": "abc123", "title": "...", "author": "...", "pages": 15, "chunks": 42, "message": "Paper loaded and indexed ✓" }
```

### `POST /upload`
Upload and index a local PDF (`multipart/form-data`, field name: `file`).

### `POST /query`
Ask a question against a loaded paper.

```json
Request:  { "paper_id": "abc123", "question": "What is the main contribution?" }
Response: { "answer": "...", "sources_used": 5, "paper_id": "abc123" }
```

### `GET /papers`
List all currently loaded papers.

### `GET /health`
Health check (includes whether `GEMINI_API_KEY` is configured).

---

## Chunking Strategy

Two-level approach (inspired by [talk2arxiv](https://github.com/evanhu1/talk2arxiv)):

1. **Section split** — regex on common academic headers (Abstract, Introduction, Method, Results, etc.) to preserve logical structure.
2. **Sliding window** — for sections longer than 1200 chars, split with 100-char overlap, breaking at sentence boundaries.

---

## Extending the Project

### Switch to Qdrant (persistent vector DB)
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(":memory:")  # or path="./qdrant_data" for persistence
client.create_collection("papers", vectors_config=VectorParams(size=768, distance=Distance.COSINE))
```

### Add reranking (like talk2arxiv)
```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = reranker.predict([(query, chunk) for chunk in candidates])
```

### Multi-paper support
The backend supports loading multiple papers simultaneously. Each gets a unique `paper_id` based on its URL or content hash. Use `GET /papers` to list them.

---

## Known Limitations

- **In-memory only** — paper indices are lost on server restart. Add Qdrant or SQLite for persistence.
- **Text PDFs only** — scanned/image PDFs won't extract. For those, add OCR (pytesseract, AWS Textract).
- **No multi-turn memory** — each query is independent. Add conversation history to the Gemini prompt for follow-up awareness.
- **Blocking I/O** — embedding and LLM calls are synchronous. Add a task queue for heavy concurrent load.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, FastAPI, Uvicorn |
| Vector search | FAISS (faiss-cpu) |
| Embeddings | Gemini `gemini-embedding-001` |
| LLM | Gemini `gemini-2.5-flash-lite` |
| PDF parsing | PyPDF2 |
| Frontend | React (CSS variables, no extra deps) |

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
     │  POST /query
     ▼
[FastAPI Backend]
     │
     ├─ /load  → fetch PDF → extract text → chunk → Gemini Embed → FAISS index
     │
     └─ /query → embed question → FAISS similarity search → top-K chunks → Gemini LLM → answer
```

### Key design choices

| Component | Choice | Why |
|-----------|--------|-----|
| Embeddings | `text-embedding-004` (Gemini) | No local model needed; 768-dim, high quality |
| Vector DB | FAISS `IndexFlatIP` (in-memory) | Zero infra, fast cosine search via inner product |
| LLM | `gemini-1.5-flash` | Fast, cheap, long context for grounded answers |
| Chunking | Section-split → sliding window | Preserves semantic sections + handles long sections |
| Paper fetch | Direct PDF download from arxiv | No scraping needed; arxiv allows programmatic access |

---

## Setup

### Prerequisites
- Python 3.10+
- A free [Gemini API key](https://aistudio.google.com/app/apikey)

### Backend

```bash
cd researchgpt
pip install -r requirements.txt
./start.sh
# Server runs at http://localhost:8000
```

### Frontend (React)

The `ResearchGPT.jsx` file is a self-contained React component. Use it in any React app:

```bash
# Option A: copy into your existing React project
cp ResearchGPT.jsx src/App.jsx

# Option B: use with Vite
npm create vite@latest my-app -- --template react
cd my-app && cp path/to/ResearchGPT.jsx src/App.jsx
npm install && npm run dev
```

Or open `ResearchGPT.jsx` directly in Claude as an artifact for a live preview.

---

## API Reference

### `POST /load`
Download, parse, chunk, and index a PDF.

```json
Request:  { "url": "https://arxiv.org/abs/1706.03762", "api_key": "AIza..." }
Response: { "paper_id": "abc123", "title": "...", "author": "...", "pages": 15, "chunks": 142, "message": "Paper loaded ✓" }
```

### `POST /query`
Ask a question against a loaded paper.

```json
Request:  { "paper_id": "abc123", "question": "What is the main contribution?", "api_key": "AIza..." }
Response: { "answer": "...", "sources_used": 5, "paper_id": "abc123" }
```

### `GET /papers`
List all currently loaded papers.

### `GET /health`
Health check.

---

## Chunking Strategy

Two-level approach (inspired by talk2arxiv):

1. **Section split** — regex on common academic headers (Abstract, Introduction, Method, Results, etc.) to preserve logical structure.
2. **Sliding window** — for sections longer than 600 chars, recursively split with 80-char overlap, breaking at sentence boundaries.

This keeps semantically related text together while ensuring no chunk is too large for the embedding model.

---

## Extending the Project

### Switch to Qdrant (persistent vector DB)
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(":memory:")  # or path="./qdrant_data" for persistence
client.create_collection("papers", vectors_config=VectorParams(size=768, distance=Distance.COSINE))
```

### Add reranking (like talk2arxiv)
```python
# After FAISS retrieval, rerank with a cross-encoder
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = reranker.predict([(query, chunk) for chunk in candidates])
```

### Multi-paper support
The backend already supports loading multiple papers simultaneously. Each gets a unique `paper_id` based on its URL hash. Use `GET /papers` to list them.

### PDF upload (instead of URL)
Add a `POST /upload` endpoint that accepts `multipart/form-data` instead of a URL.

---

## Known Limitations

- **In-memory only** — paper indices are lost on server restart. Add Qdrant or SQLite for persistence.
- **Text PDFs only** — scanned/image PDFs won't extract. For those, add OCR (pytesseract, AWS Textract).
- **No multi-turn memory** — each query is independent. Add conversation history to the Gemini prompt for follow-up awareness.
- **Single-threaded** — heavy concurrent load will slow embedding. Add async batching or a task queue.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, FastAPI, Uvicorn |
| Vector search | FAISS (faiss-cpu) |
| Embeddings | Gemini text-embedding-004 |
| LLM | Gemini 1.5 Flash (via LangChain or direct SDK) |
| PDF parsing | PyPDF2 |
| Frontend | React, Tailwind (via CSS variables) |

Inspired by [talk2arxiv](https://github.com/evanhu1/talk2arxiv).

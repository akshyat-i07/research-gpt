# ResearchGPT 🔬

A Retrieval-Augmented Generation (RAG) system for querying scientific research papers using natural language. Load papers from arXiv or upload PDFs, then ask questions and get grounded answers with streaming responses.

Powered by **Gemini** for embeddings and generation, **FAISS** for vector search, and **PyMuPDF** for reliable PDF text extraction.

**🌐 Live demo:** [researchgpt-ai.vercel.app](https://researchgpt-ai.vercel.app/)
 
> Backend runs on a free Render instance, which sleeps after periods of inactivity. The first request after idle time may take 30–60s to wake up.

---

## Features

- Load papers from arXiv URLs or local PDF uploads
- Multi-paper support with sidebar switching
- Hybrid retrieval (keyword + semantic + spread sampling for summaries)
- Streaming answers with markdown and LaTeX math rendering
- Automatic model fallback on Gemini overload/rate limits
- Boilerplate filtering for scientific PDF headers/footers

---

## Architecture

```
User Question
     │
     ▼
[React Frontend]  (Vite, localhost:5173)
     │  POST /load
     │  POST /upload
     │  POST /query/stream
     ▼
[FastAPI Backend]  (localhost:8000)
     │
     ├─ /load    → fetch PDF → PyMuPDF extract → clean → chunk → embed → FAISS index
     ├─ /upload  → same pipeline for local PDFs
     │
     └─ /query/stream → hybrid retrieval → Gemini LLM → SSE stream
```

### Key design choices

| Component | Choice | Why |
|-----------|--------|-----|
| PDF parsing | PyMuPDF (`fitz`) | Much better text extraction than PyPDF2 on multi-column scientific PDFs |
| Text cleaning | Repeated-line removal + chunk quality scoring | Filters affiliation/header noise common in journal PDFs |
| Embeddings | `gemini-embedding-001` with task types | `RETRIEVAL_DOCUMENT` for chunks, `QUESTION_ANSWERING` for queries |
| Vector DB | FAISS `IndexFlatIP` (in-memory) | Zero infra, fast cosine search via inner product |
| LLM | `gemini-2.5-flash` (default) | Good balance of speed and accuracy; `gemini-2.5-pro` for max quality |
| Retrieval | Hybrid keyword + semantic + spread | Keyword hits for definitions; spread sampling for paper summaries |
| Chunking | Section-split → sliding window (1200 chars, 100 overlap) | Preserves semantic sections; up to 80 substantive chunks per paper |
| Frontend | React + Vite + react-markdown + KaTeX | Streaming chat UI with math rendering |

---

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+ (for the frontend)
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
# API runs at http://localhost:8000
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
# UI runs at http://localhost:5173
```

Set `VITE_API_URL` in `frontend/.env` if the backend is not on `http://localhost:8000`.

---

## Configuration

All settings are optional in `.env` (defaults shown):

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Required. Your Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Primary LLM for answers |
| `GEMINI_FALLBACK_MODELS` | `gemini-2.0-flash,gemini-2.5-flash-lite` | Tried if primary is unavailable |
| `GEMINI_THINKING_BUDGET` | `1024` | Reasoning tokens for 2.5/pro models (`0` to disable) |
| `EMBED_MODEL` | `gemini-embedding-001` | Embedding model |
| `TOP_K` | `5` | Chunks retrieved per question |
| `SUMMARY_TOP_K` | `12` | Chunks used for summary/overview queries |
| `MAX_CHUNKS` | `80` | Max chunks indexed per paper |
| `MAX_CHUNK_CHARS` | `1200` | Max chars per chunk sent to the LLM |
| `MAX_OUTPUT_TOKENS` | `1024` | Max tokens in LLM response |

For maximum accuracy (slower, costlier):

```bash
GEMINI_MODEL=gemini-2.5-pro
```

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

Ask a question against a loaded paper (non-streaming).

```json
Request:  { "paper_id": "abc123", "question": "What is the main contribution?" }
Response: { "answer": "...", "sources_used": 5, "paper_id": "abc123" }
```

### `POST /query/stream`

Same as `/query`, but streams the answer as Server-Sent Events. Used by the frontend.

### `GET /papers`

List all currently loaded papers.

### `GET /health`

Health check (includes whether `GEMINI_API_KEY` is configured and which models are active).

---

## RAG Pipeline

### 1. PDF extraction & cleaning

- **PyMuPDF** extracts text with reading-order sorting
- Repeated headers/footers (journal names, affiliations, page numbers) are stripped
- Section headers (`Abstract`, `Introduction`, etc.) are normalized when missing from PDF text

### 2. Chunking

Two-level approach (inspired by [talk2arxiv](https://github.com/evanhu1/talk2arxiv)):

1. **Section split** — regex on common academic headers to preserve logical structure
2. **Sliding window** — sections longer than 1200 chars are split with 100-char overlap at sentence boundaries
3. **Quality filter** — drops short/boilerplate chunks (affiliation-only noise)
4. **Dedup** — content-hash deduplication, capped at 80 chunks per paper

### 3. Retrieval

Hybrid strategy based on query type:

| Query type | Strategy |
|------------|----------|
| Definition (`what is X?`) | Keyword match on `X` + abstract/intro chunks + semantic search |
| Summary (`summarize this paper`) | 12 chunks spread evenly across the full document |
| General | Keyword boost + semantic search, ranked by relevance and chunk quality |

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

The backend supports loading multiple papers simultaneously. Each gets a unique `paper_id` based on its URL or content hash. Use `GET /papers` to list them and switch between papers in the sidebar.

---

## Known Limitations

- **In-memory only** — paper indices are lost on server restart. Reload papers after restarting the backend.
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
| LLM | Gemini `gemini-2.5-flash` (configurable) |
| PDF parsing | PyMuPDF |
| Frontend | React 19, Vite, react-markdown, KaTeX |

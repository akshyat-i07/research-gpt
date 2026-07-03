"""
ResearchGPT Backend - RAG system for scientific papers
FastAPI server with FAISS vector search + Gemini LLM + Gemini Embeddings
"""

import json
import os
import re
import hashlib
import logging
import time
from collections import Counter

import faiss
import fitz
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pathlib import Path
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ResearchGPT API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Constants ─────────────────────────────────────────────────────────────────
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "gemini-embedding-001")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100
TOP_K = int(os.environ.get("TOP_K", "5"))
SUMMARY_TOP_K = int(os.environ.get("SUMMARY_TOP_K", "12"))
MAX_CHUNKS = int(os.environ.get("MAX_CHUNKS", "80"))
MAX_CHUNK_CHARS = int(os.environ.get("MAX_CHUNK_CHARS", "1200"))
QUERY_STOP_WORDS = {
    "what", "is", "the", "a", "an", "of", "in", "to", "for", "and", "or", "are",
    "was", "were", "how", "why", "when", "where", "does", "do", "mean", "means",
    "explain", "describe", "tell", "about", "that", "this", "with", "from",
}
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "1024"))
EMBED_BATCH_DELAY = float(os.environ.get("EMBED_BATCH_DELAY", "0.3"))
MAX_RETRIES = 4
RETRY_BASE_DELAY = 2
RETRYABLE_STATUS = {429, 500, 503}
FALLBACK_MODELS = [
    m.strip()
    for m in os.environ.get(
        "GEMINI_FALLBACK_MODELS", "gemini-2.0-flash,gemini-2.5-flash-lite"
    ).split(",")
    if m.strip()
]
THINKING_BUDGET = os.environ.get("GEMINI_THINKING_BUDGET", "1024")

STATIC_DIR = Path(__file__).resolve().parent / "frontend" / "dist"

# In-memory store: paper_id -> {index, chunks, metadata}
paper_store: dict = {}
_genai_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        if not GEMINI_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.",
            )
        _genai_client = genai.Client(api_key=GEMINI_API_KEY)
    return _genai_client


# ── Helpers ───────────────────────────────────────────────────────────────────

def _status_code(exc: Exception) -> int | None:
    if isinstance(exc, genai_errors.APIError):
        return exc.code
    return None


def _is_retryable(exc: Exception) -> bool:
    code = _status_code(exc)
    return code in RETRYABLE_STATUS


def call_with_retry(label: str, fn, *args, **kwargs):
    """Retry transient Gemini API failures (429/503)."""
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc) or attempt == MAX_RETRIES - 1:
                raise
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning(
                f"{label} failed ({_status_code(exc)}), retrying in {delay}s "
                f"(attempt {attempt + 1}/{MAX_RETRIES})"
            )
            time.sleep(delay)
    raise last_exc


def friendly_gemini_error(exc: Exception) -> HTTPException:
    code = _status_code(exc)
    msg = str(exc)
    if code == 503 or "UNAVAILABLE" in msg or "high demand" in msg.lower():
        return HTTPException(
            status_code=503,
            detail="Gemini is temporarily overloaded. Wait a few seconds and try again.",
        )
    if code == 429 or "RESOURCE_EXHAUSTED" in msg:
        return HTTPException(
            status_code=429,
            detail="Gemini rate limit reached. Wait a minute and try again.",
        )
    return HTTPException(status_code=500, detail=f"Gemini API error: {msg}")


def arxiv_url_to_pdf(url: str) -> str:
    url = url.strip()
    if not url.startswith("http"):
        url = "https://" + url
    url = re.sub(r"arxiv\.org/abs/", "arxiv.org/pdf/", url)
    url = url.replace("talk2arxiv.org", "arxiv.org")
    if not url.endswith(".pdf"):
        url += ".pdf"
    return url


def fetch_pdf_bytes(url: str) -> bytes:
    headers = {"User-Agent": "Mozilla/5.0 (ResearchGPT/1.0; research tool)"}
    r = requests.get(url, headers=headers, timeout=60, allow_redirects=True)
    r.raise_for_status()
    if b"%PDF" not in r.content[:10]:
        raise ValueError("URL did not return a valid PDF file.")
    return r.content


def extract_text_from_pdf(pdf_bytes: bytes) -> tuple[str, dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [page.get_text("text", sort=True) or "" for page in doc]
    full_text = clean_pdf_text("\n\n".join(pages))
    meta = doc.metadata or {}
    metadata = {
        "title": (meta.get("title") or "").strip() or "Research Paper",
        "author": (meta.get("author") or "").strip() or "Unknown Author",
        "pages": doc.page_count,
    }
    doc.close()
    return full_text, metadata


def clean_pdf_text(text: str) -> str:
    """Remove repeated headers/footers and collapse noisy PDF whitespace."""
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = [line.strip() for line in text.split("\n")]
    counts = Counter(line for line in lines if len(line) > 12)
    repeated = {line for line, count in counts.items() if count >= 3}

    cleaned = []
    for line in lines:
        if not line:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        if line in repeated:
            continue
        if re.fullmatch(r"\d{1,4}", line):
            continue
        cleaned.append(line)

    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()


def chunk_quality(chunk: str) -> float:
    """Score how substantive a chunk is (filters affiliation/header noise)."""
    words = chunk.split()
    if len(words) < 35:
        return 0.0

    lower = chunk.lower()
    boilerplate_markers = (
        "university",
        "institute",
        "department",
        "corresponding author",
        "e-mail",
        "email:",
        "doi:",
        "copyright",
        "all rights reserved",
        "kent state",
    )
    marker_hits = sum(1 for marker in boilerplate_markers if marker in lower)
    lines = [line.strip() for line in chunk.split("\n") if line.strip()]
    avg_line_len = sum(len(line) for line in lines) / max(len(lines), 1)

    score = min(len(words) / 80, 2.0)
    if avg_line_len < 45:
        score *= 0.6
    score -= marker_hits * 0.35
    return max(score, 0.0)


def normalize_pdf_text(text: str) -> str:
    """Insert breaks before section headers — PDF extract often omits them."""
    headers = (
        "Abstract", "Introduction", "Background", "Related Work",
        "Methodology", "Methods", "Method", "Experimental", "Experiment",
        "Results", "Discussion", "Conclusion", "Appendix", "References",
        "Acknowledgements",
    )
    for header in headers:
        text = re.sub(
            rf"(?<=[.!?]\s)({header})\b",
            rf"\n\n\1",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            rf"(?<!\n)({header})\s+",
            rf"\n\n\1 ",
            text,
            count=1,
            flags=re.IGNORECASE,
        )
    return text


def chunk_text(text: str) -> list[str]:
    """Two-level chunking: section split + sliding window."""
    text = normalize_pdf_text(text)
    section_re = re.compile(
        r"(?:^|\n)(?=(?:Abstract|Introduction|Background|Related Work|"
        r"Method(?:ology)?|Experiment|Results?|Discussion|"
        r"Conclusion|Appendix|References?|Acknowledgements?)"
        r"[^\n]{0,80}\n?)",
        re.IGNORECASE | re.MULTILINE,
    )
    sections = [s.strip() for s in section_re.split(text) if len(s.strip()) > 60]

    if not sections:
        sections = [text]

    chunks = []
    for sec in sections:
        if len(sec) <= CHUNK_SIZE:
            chunks.append(sec)
        else:
            start = 0
            while start < len(sec):
                piece = sec[start : start + CHUNK_SIZE]
                bp = piece.rfind(". ")
                if bp > CHUNK_SIZE // 2:
                    piece = piece[: bp + 1]
                chunks.append(piece.strip())
                start += max(len(piece) - CHUNK_OVERLAP, 1)

    seen, unique = set(), []
    for c in chunks:
        if chunk_quality(c) < 0.25:
            continue
        key = hashlib.md5(c.encode()).hexdigest()[:16]
        if key not in seen:
            seen.add(key)
            unique.append(c)
    if not unique:
        unique = sorted(chunks, key=len, reverse=True)[:MAX_CHUNKS]
    return unique[:MAX_CHUNKS]


def get_embeddings(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
    """Batch-embed texts with Gemini embeddings."""
    client = get_client()
    all_vecs = []
    batch_size = 20
    embed_config = genai_types.EmbedContentConfig(task_type=task_type)
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = call_with_retry(
            "embed_content",
            client.models.embed_content,
            model=EMBED_MODEL,
            contents=batch,
            config=embed_config,
        )
        vecs = [emb.values for emb in response.embeddings]
        all_vecs.extend(vecs)
        if EMBED_BATCH_DELAY > 0 and i + batch_size < len(texts):
            time.sleep(EMBED_BATCH_DELAY)
    arr = np.array(all_vecs, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr /= np.maximum(norms, 1e-9)
    return arr


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def extract_query_terms(query: str) -> list[str]:
    words = re.findall(r"\b[\w'-]{3,}\b", query.lower())
    return [w for w in words if w not in QUERY_STOP_WORDS]


def is_definition_query(query: str) -> bool:
    return bool(
        re.search(
            r"\b(what\s+is|what\s+are|define|definition|meaning\s+of|explain)\b",
            query,
            re.IGNORECASE,
        )
    )


def is_summary_query(query: str) -> bool:
    return bool(
        re.search(
            r"\b(summary|summarize|summarise|overview|main points|key findings|"
            r"entire paper|whole paper|what is this paper about|tl;dr)\b",
            query,
            re.IGNORECASE,
        )
    )


def spread_chunk_indices(total: int, count: int) -> list[int]:
    if total <= count:
        return list(range(total))
    return sorted({int(round(i * (total - 1) / (count - 1))) for i in range(count)})


def retrieve_chunks(query: str, paper_id: str, top_k: int = TOP_K) -> list[str]:
    """Hybrid retrieval: keyword matches, semantic search, and spread sampling."""
    store = paper_store[paper_id]
    chunks = store["chunks"]
    if not chunks:
        return []

    if is_summary_query(query):
        top_k = min(SUMMARY_TOP_K, len(chunks))
        ordered = spread_chunk_indices(len(chunks), top_k)
        return [chunks[i] for i in ordered]

    terms = extract_query_terms(query)
    scored: list[tuple[float, int]] = []
    seen: set[int] = set()

    def add_idx(idx: int, score: float) -> None:
        if 0 <= idx < len(chunks) and idx not in seen:
            seen.add(idx)
            scored.append((score, idx))

    if terms:
        for i, chunk in enumerate(chunks):
            lower = chunk.lower()
            keyword_score = sum(lower.count(term) for term in terms)
            if keyword_score > 0:
                add_idx(i, keyword_score * 10 + chunk_quality(chunk))

    client = get_client()
    resp = call_with_retry(
        "embed_content",
        client.models.embed_content,
        model=EMBED_MODEL,
        contents=[query],
        config=genai_types.EmbedContentConfig(task_type="QUESTION_ANSWERING"),
    )
    q_vec = np.array([resp.embeddings[0].values], dtype="float32")
    q_vec /= np.maximum(np.linalg.norm(q_vec), 1e-9)
    search_k = min(max(top_k * 4, top_k), len(chunks))
    scores, indices = store["index"].search(q_vec, search_k)
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx >= 0 and score > 0.03:
            quality = chunk_quality(chunks[int(idx)])
            add_idx(int(idx), float(score) * 5 + quality - rank * 0.05)

    if is_definition_query(query):
        for idx, chunk in enumerate(chunks[:4]):
            lower = chunk.lower()
            if not terms or any(term in lower for term in terms):
                add_idx(idx, 8 + chunk_quality(chunk))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return [chunks[i] for _, i in scored[:top_k]]


def generation_config(model: str) -> genai_types.GenerateContentConfig:
    config = genai_types.GenerateContentConfig(
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.2,
    )
    if "lite" in model:
        return config
    if "2.5" in model or "pro" in model:
        try:
            budget = int(THINKING_BUDGET)
        except ValueError:
            budget = 1024
        if budget > 0:
            config.thinking_config = genai_types.ThinkingConfig(thinking_budget=budget)
    return config


def build_prompt(query: str, context_chunks: list[str], metadata: dict) -> str:
    context = "\n\n---\n\n".join(
        (chunk[:MAX_CHUNK_CHARS] + "…") if len(chunk) > MAX_CHUNK_CHARS else chunk
        for chunk in context_chunks
    )
    return (
        f'You are answering questions about the research paper "{metadata.get("title")}".\n\n'
        f"Excerpts from the paper:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Use only information from the excerpts. Synthesize across multiple excerpts when needed. "
        "Be clear and accurate. Use bullet points for lists. "
        "If the excerpts truly lack the answer, say what is missing."
    )


def _llm_models() -> list[str]:
    return [GEMINI_MODEL] + [m for m in FALLBACK_MODELS if m != GEMINI_MODEL]


def ask_gemini(query: str, context_chunks: list[str], metadata: dict) -> str:
    client = get_client()
    prompt = build_prompt(query, context_chunks, metadata)
    models = _llm_models()
    last_exc = None
    for model in models:
        try:
            response = call_with_retry(
                f"generate_content ({model})",
                client.models.generate_content,
                model=model,
                contents=prompt,
                config=generation_config(model),
            )
            if model != GEMINI_MODEL:
                logger.info(f"Answered using fallback model: {model}")
            return response.text
        except Exception as exc:
            last_exc = exc
            if _is_retryable(exc):
                logger.warning(f"Model {model} unavailable, trying next fallback")
                continue
            raise
    raise last_exc


def stream_gemini(query: str, context_chunks: list[str], metadata: dict):
    client = get_client()
    prompt = build_prompt(query, context_chunks, metadata)
    models = _llm_models()
    last_exc = None
    for model in models:
        try:
            stream = client.models.generate_content_stream(
                model=model,
                contents=prompt,
                config=generation_config(model),
            )
            if model != GEMINI_MODEL:
                logger.info(f"Streaming with fallback model: {model}")
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
            return
        except Exception as exc:
            last_exc = exc
            if _is_retryable(exc):
                logger.warning(f"Model {model} unavailable, trying next fallback")
                continue
            raise
    if last_exc:
        raise last_exc


# ── Schemas ───────────────────────────────────────────────────────────────────

class LoadPaperRequest(BaseModel):
    url: str


class QueryRequest(BaseModel):
    paper_id: str
    question: str


class LoadPaperResponse(BaseModel):
    paper_id: str
    title: str
    author: str
    pages: int
    chunks: int
    message: str


class QueryResponse(BaseModel):
    answer: str
    sources_used: int
    paper_id: str


def index_paper(pdf_bytes: bytes, source_url: str, fallback_title: str | None = None) -> LoadPaperResponse:
    paper_id = hashlib.md5(pdf_bytes[:512]).hexdigest()[:12]

    if paper_id in paper_store:
        store = paper_store[paper_id]
        return LoadPaperResponse(
            paper_id=paper_id,
            title=store["metadata"]["title"],
            author=store["metadata"]["author"],
            pages=store["metadata"]["pages"],
            chunks=len(store["chunks"]),
            message="Loaded from cache ✓",
        )

    full_text, metadata = extract_text_from_pdf(pdf_bytes)

    if metadata["title"] == "Research Paper" and fallback_title:
        metadata["title"] = fallback_title

    if len(full_text.strip()) < 200:
        raise HTTPException(
            status_code=400,
            detail="PDF has no extractable text. It may be scanned/image-only.",
        )

    chunks = chunk_text(full_text)
    embeddings = get_embeddings(chunks)
    index = build_faiss_index(embeddings)

    paper_store[paper_id] = {
        "index": index,
        "chunks": chunks,
        "metadata": metadata,
        "url": source_url,
    }

    return LoadPaperResponse(
        paper_id=paper_id,
        title=metadata["title"],
        author=metadata["author"],
        pages=metadata["pages"],
        chunks=len(chunks),
        message="Paper loaded and indexed ✓",
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    if STATIC_DIR.is_dir():
        return FileResponse(STATIC_DIR / "index.html")
    return {
        "app": "ResearchGPT API",
        "docs": "/docs",
        "health": "/health",
        "frontend": "Run the React UI at http://localhost:5173",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "embed_model": EMBED_MODEL,
        "llm": GEMINI_MODEL,
        "api_key_configured": bool(GEMINI_API_KEY),
    }


@app.post("/load", response_model=LoadPaperResponse)
def load_paper(req: LoadPaperRequest):
    try:
        pdf_url = arxiv_url_to_pdf(req.url)
        paper_id = hashlib.md5(pdf_url.encode()).hexdigest()[:12]

        if paper_id in paper_store:
            store = paper_store[paper_id]
            return LoadPaperResponse(
                paper_id=paper_id,
                title=store["metadata"]["title"],
                author=store["metadata"]["author"],
                pages=store["metadata"]["pages"],
                chunks=len(store["chunks"]),
                message="Loaded from cache ✓",
            )

        logger.info(f"Fetching: {pdf_url}")
        pdf_bytes = fetch_pdf_bytes(pdf_url)

        logger.info("Extracting text…")
        full_text, metadata = extract_text_from_pdf(pdf_bytes)

        if len(full_text.strip()) < 200:
            raise HTTPException(status_code=400, detail="PDF appears to have no extractable text.")

        logger.info("Chunking…")
        chunks = chunk_text(full_text)
        logger.info(f"  → {len(chunks)} chunks")

        logger.info("Embedding with Gemini…")
        embeddings = get_embeddings(chunks)

        logger.info("Building FAISS index…")
        index = build_faiss_index(embeddings)

        paper_store[paper_id] = {
            "index": index,
            "chunks": chunks,
            "metadata": metadata,
            "url": pdf_url,
        }

        return LoadPaperResponse(
            paper_id=paper_id,
            title=metadata["title"],
            author=metadata["author"],
            pages=metadata["pages"],
            chunks=len(chunks),
            message="Paper loaded and indexed ✓",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /load")
        raise HTTPException(status_code=500, detail=f"Failed to load paper: {str(e)}")


def _prepare_query(req: QueryRequest) -> tuple[list[str], dict] | QueryResponse:
    if req.paper_id not in paper_store:
        raise HTTPException(
            status_code=404,
            detail="Paper session expired or not loaded. Reload the paper and try again.",
        )
    chunks = retrieve_chunks(req.question, req.paper_id)
    if not chunks:
        return QueryResponse(
            answer="I couldn't find relevant sections to answer that question. Try rephrasing.",
            sources_used=0,
            paper_id=req.paper_id,
        )
    return chunks, paper_store[req.paper_id]["metadata"]


@app.post("/query", response_model=QueryResponse)
def query_paper(req: QueryRequest):
    try:
        prepared = _prepare_query(req)
        if isinstance(prepared, QueryResponse):
            return prepared
        chunks, metadata = prepared
        started = time.time()
        answer = ask_gemini(req.question, chunks, metadata)
        logger.info(f"Query answered in {time.time() - started:.1f}s")
        return QueryResponse(answer=answer, sources_used=len(chunks), paper_id=req.paper_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /query")
        raise friendly_gemini_error(e)


@app.post("/query/stream")
def query_paper_stream(req: QueryRequest):
    try:
        prepared = _prepare_query(req)
        if isinstance(prepared, QueryResponse):
            answer = prepared.answer

            def empty_stream():
                yield f"data: {json.dumps({'text': answer})}\n\n"
                yield f"data: {json.dumps({'done': True, 'sources_used': 0})}\n\n"

            return StreamingResponse(empty_stream(), media_type="text/event-stream")

        chunks, metadata = prepared
        sources_used = len(chunks)

        def event_stream():
            started = time.time()
            try:
                for text in stream_gemini(req.question, chunks, metadata):
                    yield f"data: {json.dumps({'text': text})}\n\n"
                logger.info(f"Streamed query in {time.time() - started:.1f}s")
                yield f"data: {json.dumps({'done': True, 'sources_used': sources_used})}\n\n"
            except Exception as exc:
                logger.exception("Error in /query/stream")
                detail = friendly_gemini_error(exc).detail
                yield f"data: {json.dumps({'error': detail})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    except HTTPException:
        raise


@app.post("/upload", response_model=LoadPaperResponse)
async def upload_paper(file: UploadFile = File(...)):
    try:
        pdf_bytes = await file.read()

        if b"%PDF" not in pdf_bytes[:10]:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid PDF.")

        fallback_title = None
        if file.filename:
            fallback_title = (
                file.filename.replace(".pdf", "").replace("_", " ").replace("-", " ").title()
            )

        return index_paper(
            pdf_bytes,
            source_url=f"uploaded://{file.filename or 'unknown.pdf'}",
            fallback_title=fallback_title,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /upload")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/papers")
def list_papers():
    return [
        {
            "paper_id": pid,
            "title": v["metadata"]["title"],
            "chunks": len(v["chunks"]),
            "url": v["url"],
        }
        for pid, v in paper_store.items()
    ]


if STATIC_DIR.is_dir():

    @app.get("/{path:path}")
    def serve_spa(path: str):
        file_path = STATIC_DIR / path
        if path and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        reload_excludes=["frontend/*", "*/node_modules/*"],
    )

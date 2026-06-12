"""
ResearchGPT Backend - RAG system for scientific papers
FastAPI server with FAISS vector search + Gemini LLM + Gemini Embeddings
"""

import io
import os
import re
import hashlib
import logging
import time

import faiss
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import errors as genai_errors
from pydantic import BaseModel
from PyPDF2 import PdfReader

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
TOP_K = 5
MAX_RETRIES = 4
RETRY_BASE_DELAY = 2
RETRYABLE_STATUS = {429, 500, 503}
FALLBACK_MODELS = [
    m.strip()
    for m in os.environ.get(
        "GEMINI_FALLBACK_MODELS", "gemini-2.5-flash-lite,gemini-2.0-flash"
    ).split(",")
    if m.strip()
]

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
    reader = PdfReader(io.BytesIO(pdf_bytes))
    meta = reader.metadata or {}
    pages = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n".join(pages)
    metadata = {
        "title": (meta.get("/Title") or "").strip() or "Research Paper",
        "author": (meta.get("/Author") or "").strip() or "Unknown Author",
        "pages": len(reader.pages),
    }
    return full_text, metadata


def chunk_text(text: str) -> list[str]:
    """Two-level chunking: section split + sliding window."""
    section_re = re.compile(
        r"\n(?=(?:Abstract|Introduction|Background|Related Work|"
        r"Method(?:ology)?|Experiment|Results?|Discussion|"
        r"Conclusion|Appendix|References?|Acknowledgements?)"
        r"[^\n]{0,80}\n)",
        re.IGNORECASE,
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
        key = c[:80]
        if key not in seen and len(c) > 30:
            seen.add(key)
            unique.append(c)
    return unique[:50]


def get_embeddings(texts: list[str]) -> np.ndarray:
    """Batch-embed texts with Gemini embeddings."""
    client = get_client()
    all_vecs = []
    batch_size = 20
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = call_with_retry(
            "embed_content",
            client.models.embed_content,
            model=EMBED_MODEL,
            contents=batch,
        )
        vecs = [emb.values for emb in response.embeddings]
        all_vecs.extend(vecs)
        time.sleep(1)
    arr = np.array(all_vecs, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr /= np.maximum(norms, 1e-9)
    return arr


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def retrieve_chunks(query: str, paper_id: str, top_k: int = TOP_K) -> list[str]:
    store = paper_store[paper_id]
    client = get_client()
    resp = call_with_retry(
        "embed_content",
        client.models.embed_content,
        model=EMBED_MODEL,
        contents=[query],
    )
    q_vec = np.array([resp.embeddings[0].values], dtype="float32")
    q_vec /= np.maximum(np.linalg.norm(q_vec), 1e-9)
    scores, indices = store["index"].search(q_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and score > 0.05:
            results.append(store["chunks"][idx])
    return results


def ask_gemini(query: str, context_chunks: list[str], metadata: dict) -> str:
    client = get_client()
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""You are ResearchGPT, an expert academic assistant.

Paper: "{metadata.get('title')}" by {metadata.get('author')}

Below are the most relevant excerpts retrieved from the paper:

{context}

---

User question: {query}

Instructions:
- Answer based ONLY on the provided excerpts.
- Be concise but thorough. Use bullet points where it helps clarity.
- If the excerpts don't contain enough information, say so clearly.
- Cite specific parts of the text when helpful (e.g. "According to the paper...").
- Do NOT hallucinate or add information not present in the excerpts.
"""
    models = [GEMINI_MODEL] + [m for m in FALLBACK_MODELS if m != GEMINI_MODEL]
    last_exc = None
    for model in models:
        try:
            response = call_with_retry(
                f"generate_content ({model})",
                client.models.generate_content,
                model=model,
                contents=prompt,
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


@app.post("/query", response_model=QueryResponse)
def query_paper(req: QueryRequest):
    if req.paper_id not in paper_store:
        raise HTTPException(status_code=404, detail="Paper not loaded. Call /load first.")
    try:
        chunks = retrieve_chunks(req.question, req.paper_id)
        if not chunks:
            return QueryResponse(
                answer="I couldn't find relevant sections to answer that question. Try rephrasing.",
                sources_used=0,
                paper_id=req.paper_id,
            )
        metadata = paper_store[req.paper_id]["metadata"]
        answer = ask_gemini(req.question, chunks, metadata)
        return QueryResponse(answer=answer, sources_used=len(chunks), paper_id=req.paper_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /query")
        raise friendly_gemini_error(e)


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)

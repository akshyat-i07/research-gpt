"""
ResearchGPT Backend - RAG system for scientific papers
FastAPI server with FAISS vector search + Gemini LLM + Gemini Embeddings
"""

import io
import re
import hashlib
import logging
import requests
from google import genai
from google.genai import Client
import time
import numpy as np

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import faiss
from PyPDF2 import PdfReader
from google import genai

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
GEMINI_MODEL = "gemini-2.0-flash"
EMBED_MODEL = "gemini-embedding-001"
GEMINI_API_KEY = "AQ.Ab8RN6IaiA6A2Hv23YiJtrfCe9BeLw-CqHpIB3XSA0PuQ5EvuQ"
CHUNK_SIZE     = 1200
CHUNK_OVERLAP  = 100
TOP_K          = 5

# In-memory store: paper_id -> {index, chunks, metadata}
paper_store: dict = {}

# ── Helpers ───────────────────────────────────────────────────────────────────

def arxiv_url_to_pdf(url: str) -> str:
    url = url.strip()
    # ensure https
    if not url.startswith("http"):
        url = "https://" + url
    url = re.sub(r"arxiv\.org/abs/", "arxiv.org/pdf/", url)
    url = url.replace("talk2arxiv.org", "arxiv.org")
    if not url.endswith(".pdf"):
        url += ".pdf"
    return url


def fetch_pdf_bytes(url: str) -> bytes:
    headers = {
        "User-Agent": "Mozilla/5.0 (ResearchGPT/1.0; research tool)"
    }
    r = requests.get(url, headers=headers, timeout=60, allow_redirects=True)
    r.raise_for_status()
    # verify we actually got a PDF
    if b"%PDF" not in r.content[:10]:
        raise ValueError("URL did not return a valid PDF file.")
    return r.content


def extract_text_from_pdf(pdf_bytes: bytes) -> tuple[str, dict]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    meta = reader.metadata or {}
    pages = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n".join(pages)
    metadata = {
        "title":  (meta.get("/Title") or "").strip() or "Research Paper",
        "author": (meta.get("/Author") or "").strip() or "Unknown Author",
        "pages":  len(reader.pages),
    }
    return full_text, metadata


def chunk_text(text: str) -> list[str]:
    """Two-level chunking: section split + sliding window."""
    section_re = re.compile(
        r'\n(?=(?:Abstract|Introduction|Background|Related Work|'
        r'Method(?:ology)?|Experiment|Results?|Discussion|'
        r'Conclusion|Appendix|References?|Acknowledgements?)'
        r'[^\n]{0,80}\n)',
        re.IGNORECASE,
    )
    sections = [s.strip() for s in section_re.split(text) if len(s.strip()) > 60]

    # fallback: if no sections detected, chunk the whole text directly
    if not sections:
        sections = [text]

    chunks = []
    for sec in sections:
        if len(sec) <= CHUNK_SIZE:
            chunks.append(sec)
        else:
            start = 0
            while start < len(sec):
                piece = sec[start: start + CHUNK_SIZE]
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
    """Batch-embed texts with Gemini text-embedding-004."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    all_vecs = []
    batch_size = 20
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        response = client.models.embed_content(
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
    client = genai.Client(api_key=GEMINI_API_KEY)
    resp = client.models.embed_content(model=EMBED_MODEL, contents=[query])
    q_vec = np.array([resp.embeddings[0].values], dtype="float32")
    q_vec /= np.maximum(np.linalg.norm(q_vec), 1e-9)
    scores, indices = store["index"].search(q_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and score > 0.05:
            results.append(store["chunks"][idx])
    return results


def ask_gemini(query: str, context_chunks: list[str], metadata: dict) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)
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
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text


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


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "embed_model": EMBED_MODEL, "llm": GEMINI_MODEL}


@app.post("/load", response_model=LoadPaperResponse)
async def load_paper(req: LoadPaperRequest):
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
            raise HTTPException(400, "PDF appears to have no extractable text.")

        logger.info("Chunking…")
        chunks = chunk_text(full_text)
        logger.info(f"  → {len(chunks)} chunks")

        logger.info("Embedding with Gemini…")
        embeddings = get_embeddings(chunks)   # ← fixed: no api_key arg

        logger.info("Building FAISS index…")
        index = build_faiss_index(embeddings)

        paper_store[paper_id] = {
            "index":    index,
            "chunks":   chunks,
            "metadata": metadata,
            "url":      pdf_url,
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
        raise HTTPException(500, f"Failed to load paper: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_paper(req: QueryRequest):
    if req.paper_id not in paper_store:
        raise HTTPException(404, "Paper not loaded. Call /load first.")
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
    except Exception as e:
        logger.exception("Error in /query")
        raise HTTPException(500, f"Query failed: {str(e)}")


@app.post("/upload", response_model=LoadPaperResponse)
async def upload_paper(file: UploadFile = File(...)):
    # ← fixed: no api_key form field needed
    try:
        pdf_bytes = await file.read()

        if b"%PDF" not in pdf_bytes[:10]:
            raise HTTPException(400, "Uploaded file is not a valid PDF.")

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

        # use filename as title if PDF has no metadata title
        if metadata["title"] == "Research Paper":
            metadata["title"] = file.filename.replace(".pdf", "").replace("_", " ").replace("-", " ").title()

        if len(full_text.strip()) < 200:
            raise HTTPException(400, "PDF has no extractable text. It may be scanned/image-only.")

        chunks = chunk_text(full_text)
        embeddings = get_embeddings(chunks)
        index = build_faiss_index(embeddings)

        paper_store[paper_id] = {
            "index":    index,
            "chunks":   chunks,
            "metadata": metadata,
            "url":      f"uploaded://{file.filename}",
        }

        return LoadPaperResponse(
            paper_id=paper_id,
            title=metadata["title"],
            author=metadata["author"],
            pages=metadata["pages"],
            chunks=len(chunks),
            message="Paper uploaded and indexed ✓",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /upload")
        raise HTTPException(500, f"Upload failed: {str(e)}")


@app.get("/papers")
def list_papers():
    return [
        {"paper_id": pid, "title": v["metadata"]["title"],
         "chunks": len(v["chunks"]), "url": v["url"]}
        for pid, v in paper_store.items()
    ]


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
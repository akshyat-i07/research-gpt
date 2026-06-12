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
from typing import Optional

import faiss
import numpy as np
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from pydantic import BaseModel, Field
from PyPDF2 import PdfReader

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "gemini-embedding-001")
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100
TOP_K = 5
MAX_CHUNKS = 150

# In-memory store: paper_id -> {index, chunks, metadata, url}
paper_store: dict = {}


AUTH_ERROR_HINT = (
    "Gemini authentication failed. Create a new API key at "
    "https://aistudio.google.com/app/apikey — it must start with 'AIza'. "
    "Keys starting with 'AQ.' are Vertex Express keys and often fail with this app; "
    "link a Google Cloud project in AI Studio and regenerate, or use a standard AIza key."
)


def resolve_api_key(request_key: Optional[str] = None) -> str:
    key = (request_key or "").strip() or os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise HTTPException(
            400,
            "Gemini API key required. Set GEMINI_API_KEY in .env or pass api_key in the request.",
        )
    return key


def is_express_key(key: str) -> bool:
    return key.startswith("AQ.")


def uses_vertex_mode(key: str) -> bool:
    if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true", "yes"):
        return True
    return is_express_key(key)


def describe_key(key: str) -> str:
    if key.startswith("AIza"):
        return "ai_studio"
    if is_express_key(key):
        return "vertex_express"
    return "unknown"


def get_client(api_key: Optional[str] = None) -> genai.Client:
    key = resolve_api_key(api_key)
    if uses_vertex_mode(key):
        return genai.Client(api_key=key, vertexai=True)
    return genai.Client(api_key=key)


def raise_auth_error(exc: Exception) -> None:
    msg = str(exc)
    if any(token in msg for token in ("401", "UNAUTHENTICATED", "API key not valid", "invalid authentication")):
        raise HTTPException(401, AUTH_ERROR_HINT)
    raise exc


def embed_via_rest(texts: list[str], api_key: str) -> np.ndarray:
    """REST fallback when the SDK auth path fails (common with AQ. express keys)."""
    if uses_vertex_mode(api_key):
        base = f"https://aiplatform.googleapis.com/v1/publishers/google/models/{EMBED_MODEL}:batchEmbedContents"
    else:
        base = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL}:batchEmbedContents"

    requests_payload = [
        {
            "model": f"models/{EMBED_MODEL}",
            "content": {"parts": [{"text": text}]},
        }
        for text in texts
    ]
    r = requests.post(
        base,
        params={"key": api_key},
        json={"requests": requests_payload},
        timeout=120,
    )
    if r.status_code == 401:
        raise HTTPException(401, AUTH_ERROR_HINT)
    if not r.ok:
        raise ValueError(f"Embedding API error ({r.status_code}): {r.text[:300]}")

    data = r.json()
    embeddings = data.get("embeddings") or []
    if not embeddings:
        raise ValueError("Embedding API returned no vectors.")

    vecs = [item.get("values") or item.get("embedding", {}).get("values") for item in embeddings]
    arr = np.array(vecs, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr /= np.maximum(norms, 1e-9)
    return arr


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    return unique[:MAX_CHUNKS]


def get_embeddings(texts: list[str], api_key: Optional[str] = None) -> np.ndarray:
    key = resolve_api_key(api_key)
    all_vecs = []
    batch_size = 20

    try:
        client = get_client(api_key)
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            response = client.models.embed_content(model=EMBED_MODEL, contents=batch)
            vecs = [emb.values for emb in response.embeddings]
            all_vecs.extend(vecs)
            if i + batch_size < len(texts):
                time.sleep(0.5)
    except HTTPException:
        raise
    except Exception as sdk_err:
        logger.warning("SDK embedding failed, trying REST fallback: %s", sdk_err)
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i: i + batch_size]
                arr = embed_via_rest(batch, key)
                all_vecs.extend(arr.tolist())
                if i + batch_size < len(texts):
                    time.sleep(0.5)
        except HTTPException:
            raise
        except Exception as rest_err:
            raise_auth_error(rest_err)

    arr = np.array(all_vecs, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr /= np.maximum(norms, 1e-9)
    return arr


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def retrieve_chunks(
    query: str,
    paper_id: str,
    top_k: int = TOP_K,
    api_key: Optional[str] = None,
) -> list[dict]:
    store = paper_store[paper_id]
    key = resolve_api_key(api_key)
    try:
        client = get_client(api_key)
        resp = client.models.embed_content(model=EMBED_MODEL, contents=[query])
        q_vec = np.array([resp.embeddings[0].values], dtype="float32")
    except Exception as sdk_err:
        logger.warning("SDK query embed failed, using REST fallback: %s", sdk_err)
        try:
            q_vec = embed_via_rest([query], key)
        except HTTPException:
            raise
        except Exception as rest_err:
            raise_auth_error(rest_err)
    q_vec /= np.maximum(np.linalg.norm(q_vec), 1e-9)
    scores, indices = store["index"].search(q_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and score > 0.05:
            text = store["chunks"][idx]
            preview = text[:300] + ("…" if len(text) > 300 else "")
            results.append({"text": text, "preview": preview, "score": round(float(score), 4)})
    return results


def ask_gemini(
    query: str,
    context_chunks: list[str],
    metadata: dict,
    api_key: Optional[str] = None,
) -> str:
    client = get_client(api_key)
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
- Cite specific parts of the text when helpful (e.g. "According to the paper…").
- Do NOT hallucinate or add information not present in the excerpts.
"""
    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    except Exception as exc:
        raise_auth_error(exc)
    if not response.text:
        raise ValueError("Gemini returned an empty response. The content may have been blocked.")
    return response.text


# ── Schemas ───────────────────────────────────────────────────────────────────

class LoadPaperRequest(BaseModel):
    url: str
    api_key: Optional[str] = None


class QueryRequest(BaseModel):
    paper_id: str
    question: str
    api_key: Optional[str] = None


class SourceCitation(BaseModel):
    preview: str
    score: float


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
    sources: list[SourceCitation] = Field(default_factory=list)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    return {
        "status": "ok",
        "embed_model": EMBED_MODEL,
        "llm": GEMINI_MODEL,
        "api_key_configured": bool(key),
        "api_key_type": describe_key(key) if key else None,
        "vertex_mode": uses_vertex_mode(key) if key else False,
    }


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

        logger.info("Fetching: %s", pdf_url)
        pdf_bytes = fetch_pdf_bytes(pdf_url)

        logger.info("Extracting text…")
        full_text, metadata = extract_text_from_pdf(pdf_bytes)

        if len(full_text.strip()) < 200:
            raise HTTPException(400, "PDF appears to have no extractable text.")

        logger.info("Chunking…")
        chunks = chunk_text(full_text)
        logger.info("  → %d chunks", len(chunks))

        logger.info("Embedding with Gemini…")
        embeddings = get_embeddings(chunks, req.api_key)

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
        raise HTTPException(500, f"Failed to load paper: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_paper(req: QueryRequest):
    if req.paper_id not in paper_store:
        raise HTTPException(404, "Paper not loaded. Call /load first.")
    try:
        retrieved = retrieve_chunks(req.question, req.paper_id, api_key=req.api_key)
        if not retrieved:
            return QueryResponse(
                answer="I couldn't find relevant sections to answer that question. Try rephrasing.",
                sources_used=0,
                paper_id=req.paper_id,
                sources=[],
            )
        chunks = [r["text"] for r in retrieved]
        metadata = paper_store[req.paper_id]["metadata"]
        answer = ask_gemini(req.question, chunks, metadata, req.api_key)
        citations = [SourceCitation(preview=r["preview"], score=r["score"]) for r in retrieved]
        return QueryResponse(
            answer=answer,
            sources_used=len(chunks),
            paper_id=req.paper_id,
            sources=citations,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /query")
        raise HTTPException(500, f"Query failed: {str(e)}")


@app.post("/upload", response_model=LoadPaperResponse)
async def upload_paper(
    file: UploadFile = File(...),
    api_key: Optional[str] = Form(None),
):
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

        if metadata["title"] == "Research Paper" and file.filename:
            metadata["title"] = (
                file.filename.replace(".pdf", "").replace("_", " ").replace("-", " ").title()
            )

        if len(full_text.strip()) < 200:
            raise HTTPException(400, "PDF has no extractable text. It may be scanned/image-only.")

        chunks = chunk_text(full_text)
        embeddings = get_embeddings(chunks, api_key)
        index = build_faiss_index(embeddings)

        paper_store[paper_id] = {
            "index": index,
            "chunks": chunks,
            "metadata": metadata,
            "url": f"uploaded://{file.filename}",
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

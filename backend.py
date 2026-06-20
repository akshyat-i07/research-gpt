"""
ResearchGPT Backend - RAG system for scientific papers.

FastAPI server with FAISS vector search, Gemini generation/embeddings, and
defensive fallbacks for quota, malformed PDFs, and transient API failures.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import Counter, OrderedDict
from contextlib import asynccontextmanager
from typing import Any, Callable
from urllib.parse import urlparse

import faiss
import fitz
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import BaseModel, Field

load_dotenv()

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("researchgpt.backend")

def validate_settings() -> list[str]:
    warnings: list[str] = []
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        warnings.append("GEMINI_API_KEY is missing or still set to the placeholder.")
    if EMBED_BATCH_SIZE < 1:
        warnings.append("EMBED_BATCH_SIZE must be >= 1.")
    if MAX_CHUNKS < 1:
        warnings.append("MAX_CHUNKS must be >= 1.")
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        warnings.append("CHUNK_OVERLAP should be smaller than CHUNK_SIZE.")
    for warning in warnings:
        logger.warning("Configuration warning: %s", warning)
    return warnings


@asynccontextmanager
async def lifespan(_: FastAPI):
    validate_settings()
    yield


app = FastAPI(title="ResearchGPT API", version="1.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
EMBED_MODEL = os.environ.get("EMBED_MODEL", "gemini-embedding-001").strip()
GEMINI_API_KEY = (os.environ.get("GEMINI_API_KEY") or "").strip().strip("\"'")

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1600"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "120"))
MAX_CHUNKS = int(os.environ.get("MAX_CHUNKS", "48"))
MAX_CHUNK_CHARS = int(os.environ.get("MAX_CHUNK_CHARS", "1200"))
MIN_PDF_TEXT_CHARS = int(os.environ.get("MIN_PDF_TEXT_CHARS", "200"))
MAX_PDF_BYTES = int(os.environ.get("MAX_PDF_BYTES", str(25 * 1024 * 1024)))
MAX_PDF_PAGES = int(os.environ.get("MAX_PDF_PAGES", "120"))
MAX_PAPERS_IN_MEMORY = int(os.environ.get("MAX_PAPERS_IN_MEMORY", "20"))

TOP_K = int(os.environ.get("TOP_K", "5"))
SUMMARY_TOP_K = int(os.environ.get("SUMMARY_TOP_K", "10"))
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "1024"))
THINKING_BUDGET = os.environ.get("GEMINI_THINKING_BUDGET", "1024")

EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "8"))
EMBED_BATCH_DELAY = float(os.environ.get("EMBED_BATCH_DELAY", "0.5"))
EMBED_REQUESTS_PER_MINUTE = float(os.environ.get("EMBED_REQUESTS_PER_MINUTE", "12"))
GEMINI_REQUESTS_PER_MINUTE = float(os.environ.get("GEMINI_REQUESTS_PER_MINUTE", "30"))
MAX_RETRIES = int(os.environ.get("GEMINI_MAX_RETRIES", "5"))
RETRY_BASE_DELAY = float(os.environ.get("GEMINI_RETRY_BASE_DELAY", "2"))
RETRY_MAX_DELAY = float(os.environ.get("GEMINI_RETRY_MAX_DELAY", "60"))
HTTP_TIMEOUT_SECONDS = float(os.environ.get("HTTP_TIMEOUT_SECONDS", "45"))

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
FALLBACK_MODELS = [
    m.strip()
    for m in os.environ.get(
        "GEMINI_FALLBACK_MODELS", "gemini-2.0-flash,gemini-2.5-flash-lite"
    ).split(",")
    if m.strip()
]
QUERY_STOP_WORDS = {
    "what", "is", "the", "a", "an", "of", "in", "to", "for", "and", "or", "are",
    "was", "were", "how", "why", "when", "where", "does", "do", "mean", "means",
    "explain", "describe", "tell", "about", "that", "this", "with", "from",
}

# In-memory stores. FAISS indices are process-local by design.
paper_store: OrderedDict[str, dict[str, Any]] = OrderedDict()
embedding_cache: OrderedDict[str, list[float]] = OrderedDict()
paper_store_lock = threading.RLock()
embedding_cache_lock = threading.RLock()
_genai_client: genai.Client | None = None
_genai_client_lock = threading.Lock()


class RateLimiter:
    """Small process-local limiter to keep free-tier API usage predictable."""

    def __init__(self, requests_per_minute: float) -> None:
        self.min_interval = 60.0 / max(requests_per_minute, 0.1)
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next_allowed - now
            if delay > 0:
                time.sleep(delay)
                now = time.monotonic()
            self._next_allowed = now + self.min_interval


embed_limiter = RateLimiter(EMBED_REQUESTS_PER_MINUTE)
generate_limiter = RateLimiter(GEMINI_REQUESTS_PER_MINUTE)


class AppError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        retry_after: float | None = None,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        self.retryable = retryable
        self.retry_after = retry_after
        super().__init__(message)


class LoadPaperRequest(BaseModel):
    url: str = Field(..., min_length=3, max_length=2048)


class QueryRequest(BaseModel):
    paper_id: str = Field(..., min_length=1, max_length=64)
    question: str = Field(..., min_length=1, max_length=4000)


class LoadPaperResponse(BaseModel):
    paper_id: str
    title: str
    author: str
    pages: int
    chunks: int
    message: str
    indexing_status: str = "semantic"
    warning: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources_used: int
    paper_id: str


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    body = {
        "detail": exc.message,
        "error": {
            "code": exc.code,
            "message": exc.message,
            "retryable": exc.retryable,
        },
    }
    headers = {}
    if exc.retry_after is not None:
        headers["Retry-After"] = str(max(int(exc.retry_after), 1))
        body["error"]["retry_after_seconds"] = exc.retry_after
    return JSONResponse(status_code=exc.status_code, content=body, headers=headers)


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, str):
        exc.detail = {
            "detail": exc.detail,
            "error": {
                "code": "http_error",
                "message": exc.detail,
                "retryable": exc.status_code in {408, 429, 500, 502, 503, 504},
            },
        }
        return JSONResponse(status_code=exc.status_code, content=exc.detail, headers=exc.headers)
    return await http_exception_handler(request, exc)


def get_client() -> genai.Client:
    global _genai_client
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        raise AppError(
            503,
            "missing_api_key",
            "Gemini API key is not configured. Set GEMINI_API_KEY in .env and restart the backend.",
        )
    with _genai_client_lock:
        if _genai_client is None:
            _genai_client = genai.Client(api_key=GEMINI_API_KEY)
    return _genai_client


def _status_code(exc: Exception) -> int | None:
    if isinstance(exc, genai_errors.APIError):
        return getattr(exc, "code", None)
    status = getattr(exc, "status_code", None)
    return int(status) if status else None


def _extract_retry_delay(exc: Exception) -> float | None:
    for attr in ("details", "response_json", "message"):
        value = getattr(exc, attr, None)
        if value:
            match = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s", str(value))
            if match:
                return float(match.group(1))
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s", str(exc))
    if match:
        return float(match.group(1))
    retry_after = getattr(exc, "retry_after", None)
    return float(retry_after) if retry_after else None


def _is_retryable(exc: Exception) -> bool:
    code = _status_code(exc)
    msg = str(exc).upper()
    return (
        code in RETRYABLE_STATUS
        or "RESOURCE_EXHAUSTED" in msg
        or "UNAVAILABLE" in msg
        or "DEADLINE_EXCEEDED" in msg
        or isinstance(exc, (requests.Timeout, requests.ConnectionError))
    )


def _is_quota_error(exc: Exception) -> bool:
    return _status_code(exc) == 429 or "RESOURCE_EXHAUSTED" in str(exc).upper()


def call_with_retry(
    label: str,
    fn: Callable[..., Any],
    *args: Any,
    limiter: RateLimiter | None = None,
    **kwargs: Any,
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            if limiter:
                limiter.wait()
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            retry_after = _extract_retry_delay(exc)
            if not _is_retryable(exc) or attempt >= MAX_RETRIES - 1:
                break
            delay = min(retry_after or RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
            logger.warning(
                "%s failed with status=%s; retrying in %.1fs (attempt %s/%s)",
                label,
                _status_code(exc),
                delay,
                attempt + 1,
                MAX_RETRIES,
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def app_error_from_exception(exc: Exception, *, default_prefix: str = "Request failed") -> AppError:
    if isinstance(exc, AppError):
        return exc
    if isinstance(exc, requests.Timeout):
        return AppError(504, "network_timeout", f"{default_prefix}: network timeout.", retryable=True)
    if isinstance(exc, requests.ConnectionError):
        return AppError(502, "network_error", f"{default_prefix}: network connection failed.", retryable=True)

    code = _status_code(exc)
    retry_after = _extract_retry_delay(exc)
    if _is_quota_error(exc):
        return AppError(
            429,
            "gemini_quota_exceeded",
            "Gemini quota or rate limit was reached. The paper was kept usable where possible; wait and try again.",
            retryable=True,
            retry_after=retry_after or 60,
        )
    if code in {500, 502, 503, 504} or "UNAVAILABLE" in str(exc).upper():
        return AppError(
            503,
            "gemini_unavailable",
            "Gemini is temporarily unavailable or overloaded. Please try again shortly.",
            retryable=True,
            retry_after=retry_after,
        )
    return AppError(500, "backend_error", f"{default_prefix}. Please try again.")


def arxiv_url_to_pdf(url: str) -> str:
    url = url.strip()
    if not url:
        raise AppError(400, "invalid_url", "Please provide a paper URL.")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise AppError(400, "invalid_url", "Please provide a valid HTTP or HTTPS URL.")
    url = re.sub(r"arxiv\.org/abs/", "arxiv.org/pdf/", url)
    url = url.replace("talk2arxiv.org", "arxiv.org")
    if "arxiv.org/pdf/" in url and not url.endswith(".pdf"):
        url += ".pdf"
    return url


def fetch_pdf_bytes(url: str) -> bytes:
    headers = {"User-Agent": "Mozilla/5.0 (ResearchGPT/1.1; research tool)"}
    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=HTTP_TIMEOUT_SECONDS,
            allow_redirects=True,
            stream=True,
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        data = response.content
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 502
        raise AppError(status, "pdf_download_failed", "Could not download the PDF from that URL.") from exc
    except (requests.Timeout, requests.ConnectionError) as exc:
        raise app_error_from_exception(exc, default_prefix="Could not download the PDF") from exc

    if len(data) > MAX_PDF_BYTES:
        raise AppError(413, "pdf_too_large", f"PDF is too large. Limit is {MAX_PDF_BYTES // (1024 * 1024)} MB.")
    if b"%PDF" not in data[:1024] and "pdf" not in content_type:
        raise AppError(400, "invalid_pdf", "URL did not return a valid PDF file.")
    return data


def validate_pdf_bytes(pdf_bytes: bytes) -> None:
    if not pdf_bytes:
        raise AppError(400, "empty_file", "Uploaded file is empty.")
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise AppError(413, "pdf_too_large", f"PDF is too large. Limit is {MAX_PDF_BYTES // (1024 * 1024)} MB.")
    if b"%PDF" not in pdf_bytes[:1024]:
        raise AppError(400, "invalid_pdf", "Uploaded file is not a valid PDF.")


def clean_pdf_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = [line.strip() for line in text.split("\n")]
    counts = Counter(line for line in lines if len(line) > 12)
    repeated = {line for line, count in counts.items() if count >= 3}

    cleaned: list[str] = []
    for line in lines:
        if not line:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        if line in repeated or re.fullmatch(r"\d{1,4}", line):
            continue
        cleaned.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()


def _first_page_lines(first_page_text: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", line).strip()
        for line in first_page_text.replace("\r", "\n").split("\n")
        if line.strip()
    ]


def _normalized_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _normalized_text(text: str) -> str:
    return " ".join(_normalized_words(text))


def _is_title_match(candidate: str, title: str) -> bool:
    candidate_words = set(_normalized_words(candidate))
    title_words = set(_normalized_words(title))
    if not candidate_words or not title_words or _normalized_text(title) == "research paper":
        return False

    candidate_norm = " ".join(_normalized_words(candidate))
    title_norm = " ".join(_normalized_words(title))
    if candidate_norm == title_norm or candidate_norm in title_norm or title_norm in candidate_norm:
        return True

    overlap = len(candidate_words & title_words) / max(len(title_words), 1)
    return len(title_words) >= 3 and overlap >= 0.75


def _find_title_end_index(lines: list[str], title: str) -> int:
    if not title or _normalized_text(title) == "research paper":
        return 0

    search_limit = min(len(lines), 30)
    best_end = 0
    best_overlap = 0.0
    title_words = set(_normalized_words(title))
    for start in range(search_limit):
        combined: list[str] = []
        for end in range(start, min(search_limit, start + 4)):
            line = lines[end]
            if re.match(r"^(abstract|introduction|keywords?)\b", line, re.IGNORECASE):
                break
            combined.append(line)
            candidate = " ".join(combined)
            if _is_title_match(candidate, title):
                return end + 1
            candidate_words = set(_normalized_words(candidate))
            overlap = len(candidate_words & title_words) / max(len(title_words), 1)
            if overlap > best_overlap:
                best_overlap = overlap
                best_end = end + 1
    return best_end if best_overlap >= 0.6 else 0


def _is_metadata_noise_line(line: str) -> bool:
    lower = line.lower()
    blocked_markers = (
        "abstract", "introduction", "keywords", "university", "department",
        "institute", "school", "college", "laboratory", "email", "e-mail",
        "arxiv", "doi", "conference", "journal", "proceedings", "received",
        "accepted", "published", "copyright", "corresponding", "affiliation",
        "address", "preprint", "submitted", " lab", "research group", "faculty",
        "press", "elsevier", "springer", "ieee", "acm","cite this","read online","metrics",
        "article recommendations","supporting information","downloaded via",
    )
    return any(marker in f" {lower} " for marker in blocked_markers) or bool(re.search(r"\S+@\S+", line))


def _looks_like_title_line(line: str) -> bool:
    line = re.sub(r"\s+", " ", line).strip()
    if not line or len(line) < 8 or len(line) > 220:
        return False
    if _is_metadata_noise_line(line):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", line)
    if len(words) < 3:
        return False
    lower = line.lower()
    if "," in line and not any(token in lower for token in (":", " via ", " with ", " for ", " of ")):
        return False
    return True


def _has_strong_author_evidence(line: str) -> bool:
    """True only when a line shows clear byline formatting: footnote markers/digits/
    asterisks (affiliation superscripts) or two-or-more comma/semicolon-separated
    name-like segments (an actual author list). A single bare capitalized phrase
    (e.g. a second line of a multi-line title such as "Bridged Stilbene Mesogens")
    can satisfy `_looks_like_person_name` by accident, so that alone must never be
    treated as proof of an author line -- doing so previously caused multi-line
    titles to be truncated mid-title and their tail to leak into the author field.
    """
    if re.search(r"[*\u2020\u2021\u00a7\u00b6]", line) or re.search(r"\d", line):
        return True
    segments = _author_segments(line)
    name_like = [segment for segment in segments if _looks_like_person_name(segment)]
    return len(name_like) >= 2


def infer_title_from_first_page(first_page_text: str) -> str | None:
    lines = _first_page_lines(first_page_text)
    if not lines:
        return None

    title_lines: list[str] = []
    for line in lines[:30]:
        if re.match(r"^(abstract|introduction|keywords?)\b", line, re.IGNORECASE):
            break
        looks_like_title = _looks_like_title_line(line)
        looks_like_author = _looks_like_author_line(line, " ".join(title_lines), len(title_lines))
        if title_lines and looks_like_author and (_has_strong_author_evidence(line) or not looks_like_title):
            break
        if looks_like_title:
            title_lines.append(line)
            if len(title_lines) >= 3:
                break
            continue
        if title_lines:
            break

    title = " ".join(title_lines).strip(" ,;")
    return title or None


def _clean_author_line(line: str) -> str:
    line = line.replace("·", ",").replace("|", ",")
    line = re.sub(r"\S+@\S+", "", line)
    line = re.sub(r"https?://\S+", "", line)
    line = re.sub(r"\bORCID\b[:\s\dXx-]*", "", line, flags=re.IGNORECASE)
    line = re.sub(r"\[[^\]]*\]", "", line)
    line = re.sub(r"\([^)]*@[^)]*\)", "", line)
    line = re.sub(r"[*\u2020\u2021\u00a7\u00b6\u00b9\u00b2\u00b3\u2070-\u2079]+", "", line)
    line = re.sub(r"(?<=\w)[0-9]+(?=,|\s|$)", "", line)
    line = re.sub(r"\b[0-9]+\b", "", line)
    line = re.sub(r"\s*(,|;)\s*", r"\1 ", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip(" ,;")


def _is_affiliation_or_contact_line(line: str) -> bool:
    cleaned = _clean_author_line(line)

    if not cleaned:
        return True

    lower = cleaned.lower()

    affiliation_markers = (
        "university",
        "department",
        "institute",
        "school",
        "college",
        "faculty",
        "laboratory",
        "lab",
        "centre",
        "center",
        "research lab",
        "corporation",
        "inc.",
        " llc",
        "ltd",
        "gmbh",
        "street",
        "avenue",
        "road",
        "email",
        "e-mail",
        "@",
    )

    if any(marker in lower for marker in affiliation_markers):
        return True

    letters = sum(ch.isalpha() for ch in cleaned)
    digits = sum(ch.isdigit() for ch in cleaned)

    if digits > letters * 0.25:
        return True

    return False

def _author_segments(line: str) -> list[str]:
    cleaned = _clean_author_line(line)
    cleaned = re.sub(r"\s+\band\b\s+", ", ", cleaned, flags=re.IGNORECASE)
    return [segment.strip(" ,;") for segment in re.split(r"[,;]", cleaned) if segment.strip(" ,;")]


def _looks_like_person_name(segment: str) -> bool:
    segment = segment.strip()
    words = re.findall(
        r"(?:[A-Za-z]\.)|(?:[A-Za-z][A-Za-z'.-]*)",
        segment
    )

    if not 2 <= len(words) <= 8:
        return False

    lower = segment.lower()

    blocked_terms = {
    "abstract",
    "keywords",
    "introduction",
    "department",
    "university",
    "institute",
    "school",
    "college",
    "laboratory",
    "research",
    "group",
    "email",
    "corresponding",
    "author",
    "cite",
    "read",
    "online",
    "metrics",
    "article",
    "recommendations",
    "supporting",
    "information",
    "downloaded",
    "via",
    "langmuir",
    "acs",
    "springer",
    "elsevier",
    "wiley",
    "journal",
    "volume",
    "issue",
}

    if any(term in lower for term in blocked_terms):
        return False

    capitalized = 0

    for word in words:
        if len(word) == 2 and word.endswith("."):
            # Initials like A.
            capitalized += 1
        elif word[0].isupper():
            capitalized += 1

    return capitalized / len(words) >= 0.6

def _looks_like_author_line(line: str, title: str, line_index: int) -> bool:
    cleaned = _clean_author_line(line)
    if not cleaned or len(cleaned) > 240:
        return False

    if _is_title_match(cleaned, title):
        return False

    if _is_affiliation_or_contact_line(line):
        return False
    if re.search(r"[.!?]$", cleaned):
        return False

    segments = _author_segments(cleaned)
    if not segments:
        return False
    return any(_looks_like_person_name(segment) for segment in segments)


def _format_author_list(authors: list[str], title: str) -> str | None:
    if not authors:
        return None
    author = ", ".join(authors)
    author = re.sub(r"\s*([,;])\s*", r"\1 ", author)
    author = re.sub(r"\s+\band\b\s+", " and ", author)
    author = re.sub(r"\s+", " ", author).strip(" ,;")
    return None if _is_title_match(author, title) else author or None


def _collect_author_names(lines: list[str], title: str) -> str | None:
    """Pull person-like name segments out of a set of candidate lines, skipping
    anything that overlaps the title, affiliations, or contact info. Shared by both
    the plain-text heuristic path and the font/layout-based path so author-name
    parsing (initials, "and", commas, footnote symbols, etc.) stays consistent."""
    authors: list[str] = []
    seen: set[str] = set()
    for line in lines:
        for segment in _author_segments(line):
            if not _looks_like_person_name(segment) or _is_title_match(segment, title) or _is_metadata_noise_line(segment):
                continue
            key = _normalized_text(segment)
            if key and key not in seen:
                seen.add(key)
                authors.append(segment)
    return _format_author_list(authors, title)


def infer_author_from_first_page(first_page_text: str, title: str) -> str | None:
    lines = _first_page_lines(first_page_text)
    if not lines:
        return None

    stop_index = len(lines)
    for index, line in enumerate(lines[:80]):
        if re.match(r"^(abstract|introduction|keywords?)\b", line.strip(), re.IGNORECASE):
            stop_index = index
            break

    start_index = _find_title_end_index(lines[:stop_index], title)
    candidates = lines[start_index : min(stop_index, 40)]
    author_lines: list[str] = []
    for index, line in enumerate(candidates):
        if _looks_like_author_line(line, title, start_index + index):
            author_lines.append(line)
        elif author_lines:
            break

    return _collect_author_names(author_lines, title)


def _page_top_styled_lines(page: "fitz.Page", max_lines: int = 60) -> list[dict[str, Any]]:
    """Return the first lines of a page with font-size/weight info, in reading order.

    Each item is {"text": str, "y0": float, "size": float, "bold": bool}. This gives
    a layout signal (how a human reader visually distinguishes a large/bold title
    from a smaller author byline) that plain extracted text doesn't carry.
    """
    try:
        raw = page.get_text("dict")
    except Exception:
        return []

    items: list[dict[str, Any]] = []
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = [s for s in (line.get("spans") or []) if (s.get("text") or "").strip()]
            if not spans:
                continue
            span_texts = [
                s.get("text", "").strip()
                for s in spans
                if s.get("text", "").strip()
            ]
            text = " ".join(span_texts)
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            size = max(s.get("size", 0.0) for s in spans)
            bold = any("bold" in (s.get("font") or "").lower() for s in spans)
            y0 = line.get("bbox", [0, 0, 0, 0])[1]
            items.append({"text": text, "y0": y0, "size": round(size, 1), "bold": bold})

    items.sort(key=lambda item: item["y0"])
    return items[:max_lines]

def clean_title(title: str) -> str:
    if not title:
        return title

    # Remove line-break hyphenation
    title = re.sub(r"-\s*\n\s*", "-", title)

    # Replace remaining newlines with spaces
    title = title.replace("\n", " ")

    # Collapse spaces
    title = re.sub(r"\s+", " ", title)

    # Remove spaces around hyphens
    title = re.sub(r"\s*-\s*", "-", title)

    return title.strip()

def infer_title_author_from_layout(page: "fitz.Page") -> tuple[str | None, str | None]:
    """Primary title/author extraction using font size, not just text shape.

    Across IEEE/ACM/Springer/arXiv/MDPI templates the title is reliably the largest
    font on the first page, and the author byline immediately follows it in a
    distinct (smaller) size/weight, before affiliations. Using that visual signal
    avoids the ambiguity plain-text heuristics run into (e.g. a title's second line
    and an author's name can look identical as bare text -- "Bridged Stilbene
    Mesogens" vs "Gen-ichi Konishi" -- but never share a font size on the page).

    Returns (None, None) when font metadata isn't informative enough to trust (e.g.
    flattened/scanned PDFs where every span reports the same size), so the caller
    should fall back to the plain-text heuristics.
    """
    lines = _page_top_styled_lines(page)
    if not lines:
        return None, None

    stop_at = len(lines)
    for index, item in enumerate(lines):
        if re.match(r"^(abstract|introduction|keywords?)\b", item["text"], re.IGNORECASE):
            stop_at = index
            break
    lines = lines[:stop_at]

    candidates = [item for item in lines if len(item["text"]) >= 3 and not _is_metadata_noise_line(item["text"])]
    if not candidates:
        return None, None

    title_size = max(item["size"] for item in candidates)
    smallest_size = min(item["size"] for item in candidates)
    if title_size - smallest_size < 1.5:
        # Not enough font-size variation to trust as a layout signal.
        return None, None

    title_lines: list[str] = []
    cursor = 0
    for index, item in enumerate(candidates):
        if abs(item["size"] - title_size) <= 0.4:
            title_lines.append(item["text"])
            cursor = index + 1
            continue
        if title_lines:
            break
        # Skip section labels like "Article" that sit above the title in the same pass.
        cursor = index + 1

    if not title_lines:
        return None, None
    title = clean_title(" ".join(title_lines).strip(" ,;"))

    author_lines: list[str] = []
    for item in candidates[cursor:]:
        if _is_affiliation_or_contact_line(item["text"]):
            break
        if item["size"] >= title_size - 0.4:
            break
        author_lines.append(item["text"])

    author = _collect_author_names(author_lines, title) if author_lines else None
    return title or None, author


def _valid_metadata_author(author: str, title: str) -> str | None:
    cleaned = _clean_author_line(author)
    if not cleaned or cleaned.lower() in {"unknown", "unknown author", "anonymous"}:
        return None
    if _is_title_match(cleaned, title) or _is_affiliation_or_contact_line(cleaned):
        return None
    return cleaned


def extract_text_from_pdf(pdf_bytes: bytes) -> tuple[str, dict[str, Any]]:
    validate_pdf_bytes(pdf_bytes)
    doc = None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if doc.page_count == 0:
            raise AppError(400, "empty_pdf", "PDF has no pages.")
        if doc.page_count > MAX_PDF_PAGES:
            raise AppError(413, "pdf_too_many_pages", f"PDF has too many pages. Limit is {MAX_PDF_PAGES}.")
        pages = [page.get_text("text", sort=True) or "" for page in doc]
        full_text = clean_pdf_text("\n\n".join(pages))
        meta = doc.metadata or {}
        first_page_text = pages[0] if pages else ""

        layout_title: str | None = None
        layout_author: str | None = None
        try:
            layout_title, layout_author = infer_title_author_from_layout(doc[0])
        except Exception:
            logger.debug("Layout-based title/author extraction failed; using text heuristics", exc_info=True)

        inferred_title = layout_title or infer_title_from_first_page(first_page_text)
        metadata_title = (meta.get("title") or "").strip()
        title = clean_title(inferred_title or metadata_title or "Research Paper")
        metadata_author = _valid_metadata_author((meta.get("author") or "").strip(), title)
        inferred_author = layout_author or infer_author_from_first_page(first_page_text, title)
        metadata = {
            "title": title,
            "author": inferred_author or metadata_author or "Unknown Author",
            "pages": doc.page_count,
        }
        return full_text, metadata
    except AppError:
        raise
    except (fitz.FileDataError, RuntimeError, ValueError) as exc:
        raise AppError(400, "pdf_parse_failed", "PDF could not be parsed. It may be corrupted or unsupported.") from exc
    finally:
        if doc is not None:
            doc.close()


def normalize_pdf_text(text: str) -> str:
    headers = (
        "Abstract", "Introduction", "Background", "Related Work",
        "Methodology", "Methods", "Method", "Experimental", "Experiment",
        "Results", "Discussion", "Conclusion", "Appendix", "References",
        "Acknowledgements", "Acknowledgments",
    )
    for header in headers:
        text = re.sub(rf"(?<=[.!?]\s)({header})\b", rf"\n\n\1", text, flags=re.IGNORECASE)
        text = re.sub(rf"(?<!\n)({header})\s+", rf"\n\n\1 ", text, count=1, flags=re.IGNORECASE)
    return text


def chunk_quality(chunk: str) -> float:
    words = chunk.split()
    if len(words) < 45:
        return 0.0

    lower = chunk.lower()
    boilerplate_markers = (
        "university", "institute", "department", "corresponding author", "e-mail",
        "email:", "doi:", "copyright", "all rights reserved", "arxiv:",
    )
    marker_hits = sum(1 for marker in boilerplate_markers if marker in lower)
    lines = [line.strip() for line in chunk.split("\n") if line.strip()]
    avg_line_len = sum(len(line) for line in lines) / max(len(lines), 1)
    alpha_ratio = sum(ch.isalpha() for ch in chunk) / max(len(chunk), 1)

    score = min(len(words) / 100, 2.0)
    if avg_line_len < 45:
        score *= 0.65
    if alpha_ratio < 0.55:
        score *= 0.55
    score -= marker_hits * 0.3
    return max(score, 0.0)


def chunk_text(text: str) -> list[str]:
    text = normalize_pdf_text(text)
    section_re = re.compile(
        r"(?:^|\n)(?=(?:Abstract|Introduction|Background|Related Work|"
        r"Method(?:ology)?|Experiments?|Results?|Discussion|"
        r"Conclusion|Appendix|References?|Acknowledgements?|Acknowledgments)"
        r"[^\n]{0,80}\n?)",
        re.IGNORECASE | re.MULTILINE,
    )
    sections = [s.strip() for s in section_re.split(text) if len(s.strip()) > 120] or [text]

    raw_chunks: list[str] = []
    for section in sections:
        start = 0
        while start < len(section):
            piece = section[start : start + CHUNK_SIZE]
            breakpoint_idx = max(piece.rfind(". "), piece.rfind("\n\n"))
            if breakpoint_idx > CHUNK_SIZE // 2:
                piece = piece[: breakpoint_idx + 1]
            piece = piece.strip()
            if piece:
                raw_chunks.append(piece)
            start += max(len(piece) - CHUNK_OVERLAP, CHUNK_SIZE - CHUNK_OVERLAP)

    seen: set[str] = set()
    unique: list[str] = []
    for chunk in raw_chunks:
        key = hashlib.sha256(chunk.encode("utf-8")).hexdigest()[:20]
        if key in seen or chunk_quality(chunk) < 0.3:
            continue
        seen.add(key)
        unique.append(chunk)

    if not unique:
        unique = sorted(raw_chunks, key=len, reverse=True)
    unique = sorted(unique, key=lambda c: (-chunk_quality(c), text.find(c)))
    return unique[:MAX_CHUNKS]


def _embedding_key(text: str, task_type: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{EMBED_MODEL}:{task_type}:{digest}"


def _cache_get(key: str) -> list[float] | None:
    with embedding_cache_lock:
        value = embedding_cache.get(key)
        if value is not None:
            embedding_cache.move_to_end(key)
        return value


def _cache_set(key: str, value: list[float]) -> None:
    max_items = int(os.environ.get("EMBED_CACHE_ITEMS", "4096"))
    with embedding_cache_lock:
        embedding_cache[key] = value
        embedding_cache.move_to_end(key)
        while len(embedding_cache) > max_items:
            embedding_cache.popitem(last=False)


def normalize_embeddings(vectors: list[list[float]]) -> np.ndarray:
    if not vectors:
        raise AppError(503, "no_embeddings", "No embeddings were generated.")
    arr = np.array(vectors, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr /= np.maximum(norms, 1e-9)
    return arr


def get_embeddings(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
    if not texts:
        raise AppError(400, "no_chunks", "No text chunks were available for embedding.")

    client = get_client()
    results: list[list[float] | None] = [None] * len(texts)
    pending: OrderedDict[str, tuple[str, list[int]]] = OrderedDict()

    for index, text in enumerate(texts):
        key = _embedding_key(text, task_type)
        cached = _cache_get(key)
        if cached is not None:
            results[index] = cached
            continue
        if key not in pending:
            pending[key] = (text, [])
        pending[key][1].append(index)

    embed_config = genai_types.EmbedContentConfig(task_type=task_type)
    pending_items = list(pending.items())
    for start in range(0, len(pending_items), EMBED_BATCH_SIZE):
        batch_items = pending_items[start : start + EMBED_BATCH_SIZE]
        batch_texts = [item[1][0] for item in batch_items]
        response = call_with_retry(
            "embed_content",
            client.models.embed_content,
            model=EMBED_MODEL,
            contents=batch_texts,
            config=embed_config,
            limiter=embed_limiter,
        )
        embeddings = getattr(response, "embeddings", None) or []
        if len(embeddings) != len(batch_items):
            raise AppError(502, "embedding_response_mismatch", "Gemini returned an incomplete embedding response.", retryable=True)
        for (key, (_, indices)), embedding in zip(batch_items, embeddings):
            vector = list(embedding.values)
            _cache_set(key, vector)
            for index in indices:
                results[index] = vector
        if EMBED_BATCH_DELAY > 0 and start + EMBED_BATCH_SIZE < len(pending_items):
            time.sleep(EMBED_BATCH_DELAY)

    return normalize_embeddings([vector for vector in results if vector is not None])


def get_query_embedding(query: str) -> np.ndarray | None:
    try:
        return get_embeddings([query], task_type="QUESTION_ANSWERING")
    except Exception as exc:
        logger.warning("Query embedding unavailable; falling back to keyword retrieval: %s", exc)
        return None


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise AppError(500, "invalid_embeddings", "Embedding matrix is empty.")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def extract_query_terms(query: str) -> list[str]:
    words = re.findall(r"\b[\w'-]{3,}\b", query.lower())
    return [word for word in words if word not in QUERY_STOP_WORDS]


def is_definition_query(query: str) -> bool:
    return bool(re.search(r"\b(what\s+is|what\s+are|define|definition|meaning\s+of|explain)\b", query, re.IGNORECASE))


def is_summary_query(query: str) -> bool:
    return bool(re.search(r"\b(summary|summarize|summarise|overview|main points|key findings|entire paper|whole paper|what is this paper about|tl;dr)\b", query, re.IGNORECASE))


def spread_chunk_indices(total: int, count: int) -> list[int]:
    if total <= count:
        return list(range(total))
    return sorted({int(round(i * (total - 1) / (count - 1))) for i in range(count)})


def retrieve_chunks(query: str, paper_id: str, top_k: int = TOP_K) -> list[str]:
    with paper_store_lock:
        store = paper_store.get(paper_id)
        if store:
            paper_store.move_to_end(paper_id)
    if not store:
        raise AppError(404, "paper_not_found", "Paper session expired or not loaded. Reload the paper and try again.")

    chunks: list[str] = store["chunks"]
    if not chunks:
        return []

    if is_summary_query(query):
        top_k = min(SUMMARY_TOP_K, len(chunks))
        return [chunks[i] for i in spread_chunk_indices(len(chunks), top_k)]

    terms = extract_query_terms(query)
    scored: dict[int, float] = {}

    if terms:
        for index, chunk in enumerate(chunks):
            lower = chunk.lower()
            hits = sum(lower.count(term) for term in terms)
            if hits:
                scored[index] = scored.get(index, 0.0) + hits * 10 + chunk_quality(chunk)

    q_vec = get_query_embedding(query) if store.get("index") is not None else None
    if q_vec is not None and store.get("index") is not None:
        search_k = min(max(top_k * 4, top_k), len(chunks))
        scores, indices = store["index"].search(q_vec, search_k)
        for rank, (score, index) in enumerate(zip(scores[0], indices[0])):
            if index >= 0 and score > 0.03:
                idx = int(index)
                scored[idx] = max(
                    scored.get(idx, 0.0),
                    float(score) * 5 + chunk_quality(chunks[idx]) - rank * 0.05,
                )

    if is_definition_query(query):
        for index, chunk in enumerate(chunks[:4]):
            lower = chunk.lower()
            if not terms or any(term in lower for term in terms):
                scored[index] = max(scored.get(index, 0.0), 8 + chunk_quality(chunk))

    if not scored:
        fallback_count = min(top_k, len(chunks))
        return [chunks[i] for i in spread_chunk_indices(len(chunks), fallback_count)]

    ranked = sorted(scored.items(), key=lambda item: (-item[1], item[0]))
    return [chunks[index] for index, _ in ranked[:top_k]]


def generation_config(model: str) -> genai_types.GenerateContentConfig:
    config = genai_types.GenerateContentConfig(max_output_tokens=MAX_OUTPUT_TOKENS, temperature=0.2)
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


def build_prompt(query: str, context_chunks: list[str], metadata: dict[str, Any]) -> str:
    context = "\n\n---\n\n".join(
        (chunk[:MAX_CHUNK_CHARS] + "...") if len(chunk) > MAX_CHUNK_CHARS else chunk
        for chunk in context_chunks
    )
    status_note = ""
    if metadata.get("indexing_status") == "keyword_only":
        status_note = "\nSemantic embeddings were unavailable, so excerpts were selected with keyword and document-position retrieval.\n"
    return (
        f'You are answering questions about the research paper "{metadata.get("title")}".\n'
        f"{status_note}\nExcerpts from the paper:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Use only information from the excerpts. Synthesize across multiple excerpts when needed. "
        "Be clear and accurate. Use bullet points for lists. "
        "If the excerpts truly lack the answer, say what is missing."
    )


def _llm_models() -> list[str]:
    return [GEMINI_MODEL] + [model for model in FALLBACK_MODELS if model != GEMINI_MODEL]


def ask_gemini(query: str, context_chunks: list[str], metadata: dict[str, Any]) -> str:
    client = get_client()
    prompt = build_prompt(query, context_chunks, metadata)
    last_exc: Exception | None = None
    for model in _llm_models():
        try:
            response = call_with_retry(
                f"generate_content ({model})",
                client.models.generate_content,
                model=model,
                contents=prompt,
                config=generation_config(model),
                limiter=generate_limiter,
            )
            text = (getattr(response, "text", None) or "").strip()
            if not text:
                raise AppError(502, "empty_model_response", "Gemini returned an empty answer.", retryable=True)
            if model != GEMINI_MODEL:
                logger.info("Answered using fallback model: %s", model)
            return text
        except Exception as exc:
            last_exc = exc
            if _is_retryable(exc):
                logger.warning("Model %s failed; trying next fallback if available.", model)
                continue
            raise
    assert last_exc is not None
    raise last_exc


def stream_gemini(query: str, context_chunks: list[str], metadata: dict[str, Any]):
    client = get_client()
    prompt = build_prompt(query, context_chunks, metadata)
    last_exc: Exception | None = None
    for model in _llm_models():
        try:
            generate_limiter.wait()
            stream = client.models.generate_content_stream(
                model=model,
                contents=prompt,
                config=generation_config(model),
            )
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
            return
        except Exception as exc:
            last_exc = exc
            if _is_retryable(exc):
                logger.warning("Streaming model %s failed; trying next fallback if available.", model)
                continue
            raise
    if last_exc:
        raise last_exc


def _paper_id_for_bytes(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()[:16]


def _evict_old_papers_if_needed() -> None:
    with paper_store_lock:
        while len(paper_store) > MAX_PAPERS_IN_MEMORY:
            evicted_id, _ = paper_store.popitem(last=False)
            logger.info("Evicted paper from memory: %s", evicted_id)


def index_paper(pdf_bytes: bytes, source_url: str, fallback_title: str | None = None) -> LoadPaperResponse:
    paper_id = _paper_id_for_bytes(pdf_bytes)
    with paper_store_lock:
        cached = paper_store.get(paper_id)
        if cached:
            paper_store.move_to_end(paper_id)
            return LoadPaperResponse(
                paper_id=paper_id,
                title=cached["metadata"]["title"],
                author=cached["metadata"]["author"],
                pages=cached["metadata"]["pages"],
                chunks=len(cached["chunks"]),
                message="Loaded from cache",
                indexing_status=cached.get("indexing_status", "semantic"),
                warning=cached.get("warning"),
            )

    full_text, metadata = extract_text_from_pdf(pdf_bytes)
    if metadata["title"] == "Research Paper" and fallback_title:
        metadata["title"] = fallback_title
    if len(full_text.strip()) < MIN_PDF_TEXT_CHARS:
        raise AppError(400, "no_extractable_text", "PDF has no extractable text. It may be scanned or image-only.")

    chunks = chunk_text(full_text)
    if not chunks:
        raise AppError(400, "no_usable_chunks", "PDF text was extracted, but no usable research sections were found.")

    index = None
    indexing_status = "semantic"
    warning = None
    try:
        embeddings = get_embeddings(chunks)
        index = build_faiss_index(embeddings)
    except Exception as exc:
        if _is_quota_error(exc) or _is_retryable(exc):
            app_err = app_error_from_exception(exc, default_prefix="Could not create embeddings")
            indexing_status = "keyword_only"
            warning = app_err.message
            logger.warning("Embedding unavailable for %s; storing keyword-only index: %s", paper_id, exc)
        else:
            raise

    metadata["indexing_status"] = indexing_status
    with paper_store_lock:
        paper_store[paper_id] = {
            "index": index,
            "chunks": chunks,
            "metadata": metadata,
            "url": source_url,
            "indexing_status": indexing_status,
            "warning": warning,
            "loaded_at": time.time(),
        }
        paper_store.move_to_end(paper_id)
    _evict_old_papers_if_needed()

    message = "Paper loaded and indexed" if indexing_status == "semantic" else "Paper loaded with keyword-only search"
    return LoadPaperResponse(
        paper_id=paper_id,
        title=metadata["title"],
        author=metadata["author"],
        pages=metadata["pages"],
        chunks=len(chunks),
        message=message,
        indexing_status=indexing_status,
        warning=warning,
    )


def _prepare_query(req: QueryRequest) -> tuple[list[str], dict[str, Any]] | QueryResponse:
    chunks = retrieve_chunks(req.question, req.paper_id)
    if not chunks:
        return QueryResponse(
            answer="I could not find relevant sections to answer that question. Try rephrasing.",
            sources_used=0,
            paper_id=req.paper_id,
        )
    with paper_store_lock:
        metadata = dict(paper_store[req.paper_id]["metadata"])
    return chunks, metadata


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
    warnings = validate_settings()
    return {
        "status": "ok" if not warnings else "degraded",
        "warnings": warnings,
        "embed_model": EMBED_MODEL,
        "llm": GEMINI_MODEL,
        "api_key_configured": bool(GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here"),
        "papers_loaded": len(paper_store),
        "embedding_cache_items": len(embedding_cache),
    }


@app.post("/load", response_model=LoadPaperResponse)
def load_paper(req: LoadPaperRequest):
    try:
        pdf_url = arxiv_url_to_pdf(req.url)
        logger.info("Fetching PDF: %s", pdf_url)
        pdf_bytes = fetch_pdf_bytes(pdf_url)
        return index_paper(pdf_bytes, pdf_url)
    except AppError:
        raise
    except Exception as exc:
        logger.exception("Error in /load")
        raise app_error_from_exception(exc, default_prefix="Failed to load paper") from exc


@app.post("/upload", response_model=LoadPaperResponse)
async def upload_paper(file: UploadFile = File(...)):
    try:
        filename = file.filename or "unknown.pdf"
        if not filename.lower().endswith(".pdf"):
            raise AppError(400, "unsupported_file_type", "Only PDF files are supported.")
        pdf_bytes = await file.read()
        validate_pdf_bytes(pdf_bytes)
        fallback_title = filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
        return index_paper(pdf_bytes, source_url=f"uploaded://{filename}", fallback_title=fallback_title)
    except AppError:
        raise
    except Exception as exc:
        logger.exception("Error in /upload")
        raise app_error_from_exception(exc, default_prefix="Upload failed") from exc
    finally:
        await file.close()


@app.post("/query", response_model=QueryResponse)
def query_paper(req: QueryRequest):
    try:
        prepared = _prepare_query(req)
        if isinstance(prepared, QueryResponse):
            return prepared
        chunks, metadata = prepared
        started = time.time()
        answer = ask_gemini(req.question, chunks, metadata)
        logger.info("Query answered in %.1fs", time.time() - started)
        return QueryResponse(answer=answer, sources_used=len(chunks), paper_id=req.paper_id)
    except AppError:
        raise
    except Exception as exc:
        logger.exception("Error in /query")
        raise app_error_from_exception(exc, default_prefix="Query failed") from exc


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
                logger.info("Streamed query in %.1fs", time.time() - started)
                yield f"data: {json.dumps({'done': True, 'sources_used': sources_used})}\n\n"
            except Exception as exc:
                logger.exception("Error in /query/stream")
                app_err = app_error_from_exception(exc, default_prefix="Query failed")
                yield f"data: {json.dumps({'error': app_err.message, 'code': app_err.code, 'retryable': app_err.retryable})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    except AppError:
        raise


@app.get("/papers")
def list_papers():
    with paper_store_lock:
        return [
            {
                "paper_id": paper_id,
                "title": value["metadata"]["title"],
                "author": value["metadata"]["author"],
                "pages": value["metadata"]["pages"],
                "chunks": len(value["chunks"]),
                "url": value["url"],
                "indexing_status": value.get("indexing_status", "semantic"),
                "warning": value.get("warning"),
            }
            for paper_id, value in paper_store.items()
        ]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=os.environ.get("RELOAD", "true").lower() == "true",
        reload_excludes=["frontend/*", "*/node_modules/*", ".venv/*"],
    )
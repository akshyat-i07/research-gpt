import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import fitz
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import backend  # noqa: E402


class QuotaError(Exception):
    status_code = 429

    def __str__(self):
        return 'RESOURCE_EXHAUSTED retryDelay: "1s"'


@pytest.fixture(autouse=True)
def clear_state(monkeypatch):
    backend.paper_store.clear()
    backend.embedding_cache.clear()
    monkeypatch.setattr(backend.embed_limiter, "wait", lambda: None)
    monkeypatch.setattr(backend.generate_limiter, "wait", lambda: None)
    monkeypatch.setattr(backend, "MAX_RETRIES", 1)


def test_health_degrades_when_api_key_missing(monkeypatch):
    monkeypatch.setattr(backend, "GEMINI_API_KEY", "")

    payload = backend.health()

    assert payload["status"] == "degraded"
    assert payload["api_key_configured"] is False


def test_invalid_upload_returns_sanitized_error():
    client = TestClient(backend.app)

    response = client.post(
        "/upload",
        files={"file": ("not-a-pdf.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_file_type"
    assert "Traceback" not in response.text


def test_embedding_quota_falls_back_to_keyword_only(monkeypatch):
    monkeypatch.setattr(
        backend,
        "extract_text_from_pdf",
        lambda _: (
            "Introduction " + ("transformer attention mechanism improves sequence modeling. " * 80),
            {"title": "Quota Paper", "author": "Tester", "pages": 1},
        ),
    )
    monkeypatch.setattr(backend, "get_embeddings", lambda chunks: (_ for _ in ()).throw(QuotaError()))

    response = backend.index_paper(b"%PDF-1.4 test", "uploaded://quota.pdf")

    assert response.indexing_status == "keyword_only"
    assert response.warning
    assert backend.paper_store[response.paper_id]["index"] is None


def test_embedding_cache_deduplicates_requests(monkeypatch):
    calls = []

    class FakeModels:
        def embed_content(self, *, model, contents, config):
            calls.append(list(contents))
            return SimpleNamespace(
                embeddings=[
                    SimpleNamespace(values=[float(len(text)), 1.0, 0.0])
                    for text in contents
                ]
            )

    monkeypatch.setattr(backend, "get_client", lambda: SimpleNamespace(models=FakeModels()))
    monkeypatch.setattr(backend, "EMBED_BATCH_SIZE", 10)

    arr = backend.get_embeddings(["same text", "same text", "different text"])

    assert arr.shape == (3, 3)
    assert calls == [["same text", "different text"]]


def test_keyword_retrieval_works_without_faiss_index():
    backend.paper_store["p1"] = {
        "index": None,
        "chunks": [
            "The abstract discusses optimization and training.",
            "The method uses transformer attention for sequence modeling.",
            "The conclusion mentions future work.",
        ],
        "metadata": {"title": "Paper", "author": "A", "pages": 1, "indexing_status": "keyword_only"},
        "url": "uploaded://paper.pdf",
    }

    chunks = backend.retrieve_chunks("How does transformer attention work?", "p1")

    assert chunks
    assert "transformer attention" in chunks[0]


def test_query_quota_error_is_structured(monkeypatch):
    client = TestClient(backend.app)
    backend.paper_store["p1"] = {
        "index": None,
        "chunks": ["The method uses transformer attention for sequence modeling."],
        "metadata": {"title": "Paper", "author": "A", "pages": 1, "indexing_status": "keyword_only"},
        "url": "uploaded://paper.pdf",
    }
    monkeypatch.setattr(backend, "ask_gemini", lambda *args, **kwargs: (_ for _ in ()).throw(QuotaError()))

    response = client.post("/query", json={"paper_id": "p1", "question": "summarize"})

    assert response.status_code == 429
    body = response.json()
    assert body["error"]["code"] == "gemini_quota_exceeded"
    assert body["error"]["retryable"] is True
    assert isinstance(body["detail"], str)


def test_chunking_caps_large_documents(monkeypatch):
    monkeypatch.setattr(backend, "MAX_CHUNKS", 5)
    text = "Introduction\n" + ("This section explains a robust research method with meaningful details. " * 1000)

    chunks = backend.chunk_text(text)

    assert 1 <= len(chunks) <= 5


def test_vector_database_operations():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")

    index = backend.build_faiss_index(vectors)
    scores, indices = index.search(np.array([[1.0, 0.0]], dtype="float32"), 1)

    assert indices[0][0] == 0
    assert scores[0][0] == pytest.approx(1.0)


def test_author_is_inferred_from_first_page_when_metadata_is_blank():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "A Reliable Parser for Research Papers\n"
        "Alice Smith, Bob Jones\n"
        "Department of Computer Science, Example University\n"
        "Abstract\n"
        + ("This paper presents a robust backend system for extracting metadata. " * 20),
    )
    pdf_bytes = doc.tobytes()
    doc.close()

    _, metadata = backend.extract_text_from_pdf(pdf_bytes)

    assert metadata["title"] == "A Reliable Parser for Research Papers"
    assert metadata["author"] == "Alice Smith, Bob Jones"


def test_title_with_comma_is_not_mistaken_for_author():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Deep Learning for Documents, Tables, and Figures\n"
        "Jane Doe and John Roe\n"
        "Example AI Lab\n"
        "Abstract\n"
        + ("This paper studies document understanding with careful metadata extraction. " * 20),
    )
    pdf_bytes = doc.tobytes()
    doc.close()

    _, metadata = backend.extract_text_from_pdf(pdf_bytes)

    assert metadata["title"] == "Deep Learning for Documents, Tables, and Figures"
    assert metadata["author"] == "Jane Doe, John Roe"


@pytest.mark.parametrize(
    ("publisher", "first_page", "expected_title", "expected_author"),
    [
        (
            "arxiv",
            "Attention Is All You Need\n"
            "Ashish Vaswani1, Noam Shazeer1, Niki Parmar1\n"
            "1 Google Brain\n"
            "Abstract\n"
            "The dominant sequence transduction models are based on complex recurrent networks.",
            "Attention Is All You Need",
            "Ashish Vaswani, Noam Shazeer, Niki Parmar",
        ),
        (
            "ieee",
            "Robust Semantic Retrieval for Scientific Documents\n"
            "John A. Smith, Member, IEEE, Maria Garcia, and Li Wei\n"
            "Department of Electrical Engineering, Example University\n"
            "john.smith@example.edu\n"
            "Abstract\n"
            "Semantic retrieval is an important component of modern document systems.",
            "Robust Semantic Retrieval for Scientific Documents",
            "John A. Smith, Maria Garcia, Li Wei",
        ),
        (
            "springer",
            "Neural Indexing for Research Assistants\n"
            "Priya Raman1 · Omar Khaled2 · Emily Chen3\n"
            "1 Department of Computer Science, Example University\n"
            "2 Springer Nature, London, UK\n"
            "Abstract\n"
            "Research assistants require robust indexing and retrieval.",
            "Neural Indexing for Research Assistants",
            "Priya Raman, Omar Khaled, Emily Chen",
        ),
        (
            "acm",
            "Metadata Extraction in Digital Libraries\n"
            "Sarah Connor\n"
            "ACM Research Lab\n"
            "Miguel Santos\n"
            "University of Example\n"
            "ABSTRACT\n"
            "Digital libraries often contain varied publication metadata.",
            "Metadata Extraction in Digital Libraries",
            "Sarah Connor, Miguel Santos",
        ),
        (
            "elsevier",
            "Reliable PDF Understanding for Scholarly Search\n"
            "Chen Li a,*, Robert King b, Ana Silva c\n"
            "a School of Information, Example University\n"
            "b Elsevier Research, Amsterdam, Netherlands\n"
            "c Department of Informatics, Example Institute\n"
            "Abstract\n"
            "Scholarly search depends on reliable PDF understanding.",
            "Reliable PDF Understanding for Scholarly Search",
            "Chen Li, Robert King, Ana Silva",
        ),
    ],
)
def test_author_extraction_publisher_layouts(publisher, first_page, expected_title, expected_author):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), first_page + "\n" + ("This paper adds enough extractable text. " * 20))
    pdf_bytes = doc.tobytes()
    doc.close()

    _, metadata = backend.extract_text_from_pdf(pdf_bytes)

    assert metadata["title"] == expected_title, publisher
    assert metadata["author"] == expected_author, publisher
    assert metadata["author"] != metadata["title"]


def test_metadata_author_fallback_rejects_title_as_author():
    doc = fitz.open()
    doc.set_metadata(
        {
            "title": "Metadata Title",
            "author": "Metadata Title",
        }
    )
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Metadata Title\n"
        "Abstract\n"
        + ("This paper intentionally has no visible author block. " * 20),
    )
    pdf_bytes = doc.tobytes()
    doc.close()

    _, metadata = backend.extract_text_from_pdf(pdf_bytes)

    assert metadata["title"] == "Metadata Title"
    assert metadata["author"] == "Unknown Author"

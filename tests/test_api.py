import csv
import io
from unittest.mock import AsyncMock, patch

import fitz
import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


# --- /health ---

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert "chromadb" in body["checks"]
    assert "sqlite" in body["checks"]


# --- /query ---

@pytest.fixture
def mock_query_pipeline():
    """Patch route_query and build_response to avoid real LLM/DB calls."""
    routed = {
        "intent": "macro",
        "vector_results": [{"text": "Rates rising.", "source": "report.pdf", "page": 1, "score": 0.9}],
        "sql_results": {"sql_generated": "", "rows": [], "error": None},
    }
    built = {"answer": "Rates are rising [Source: report.pdf, p.1].", "sources": ["[Source: report.pdf, p.1]"]}

    with (
        patch("main.route_query", new=AsyncMock(return_value=routed)),
        patch("main.build_response", return_value=built),
    ):
        yield


def test_query_returns_answer(mock_query_pipeline):
    r = client.post("/query", json={"question": "What is the interest rate outlook?"})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert "sources" in body
    assert "intent" in body
    assert body["intent"] == "macro"
    assert len(body["sources"]) > 0


def test_query_missing_field():
    r = client.post("/query", json={})
    assert r.status_code == 422


def test_query_propagates_500_on_error():
    with patch("main.route_query", new=AsyncMock(side_effect=RuntimeError("boom"))):
        r = client.post("/query", json={"question": "test"})
    assert r.status_code == 500
    assert "boom" in r.json()["detail"]


# --- /upload-pdf ---

@pytest.fixture
def valid_pdf_bytes():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Macro economic trends. " * 50)
    buf = doc.tobytes()
    doc.close()
    return buf


def test_upload_pdf_valid(valid_pdf_bytes):
    ingest_result = {"filename": "test.pdf", "chunks_added": 3, "status": "ok"}
    with patch("main.ingest_pdf", return_value=ingest_result):
        r = client.post(
            "/upload-pdf",
            files={"file": ("test.pdf", valid_pdf_bytes, "application/pdf")},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["detail"]["chunks_added"] == 3


def test_upload_pdf_wrong_extension():
    r = client.post(
        "/upload-pdf",
        files={"file": ("data.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 400
    assert "PDF" in r.json()["detail"]


def test_upload_pdf_corrupt_file():
    import fitz as _fitz
    with patch("main.ingest_pdf", side_effect=_fitz.FileDataError("bad pdf")):
        r = client.post(
            "/upload-pdf",
            files={"file": ("bad.pdf", b"not a pdf", "application/pdf")},
        )
    assert r.status_code == 422


def test_upload_pdf_server_error():
    with patch("main.ingest_pdf", side_effect=Exception("disk full")):
        r = client.post(
            "/upload-pdf",
            files={"file": ("test.pdf", b"%PDF-1.4", "application/pdf")},
        )
    assert r.status_code == 500


# --- /upload-csv ---

@pytest.fixture
def valid_csv_bytes():
    buf = io.StringIO()
    csv.writer(buf).writerows([
        ["Ticker", "Price", "Market Cap"],
        ["AAPL", "189.5", "2950"],
        ["TSLA", "245.1", "780"],
    ])
    return buf.getvalue().encode()


def test_upload_csv_valid(valid_csv_bytes):
    ingest_result = {"rows_inserted": 2, "columns": ["ticker", "price", "market_cap"], "status": "ok"}
    with patch("main.ingest_csv", return_value=ingest_result):
        r = client.post(
            "/upload-csv",
            files={"file": ("stocks.csv", valid_csv_bytes, "text/csv")},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["detail"]["rows_inserted"] == 2


def test_upload_csv_wrong_extension():
    r = client.post(
        "/upload-csv",
        files={"file": ("data.pdf", b"col1,col2", "application/pdf")},
    )
    assert r.status_code == 400
    assert "CSV" in r.json()["detail"]


def test_upload_csv_empty_file():
    with patch("main.ingest_csv", side_effect=ValueError("CSV 'empty.csv' is empty.")):
        r = client.post(
            "/upload-csv",
            files={"file": ("empty.csv", b"col1,col2\n", "text/csv")},
        )
    assert r.status_code == 422


def test_upload_csv_server_error(valid_csv_bytes):
    with patch("main.ingest_csv", side_effect=Exception("db locked")):
        r = client.post(
            "/upload-csv",
            files={"file": ("stocks.csv", valid_csv_bytes, "text/csv")},
        )
    assert r.status_code == 500


# --- DELETE /documents/{filename} ---

def _mock_con(doc_type):
    """Return a mock SQLite connection that returns (doc_type,) for fetchone."""
    from unittest.mock import MagicMock
    con = MagicMock()
    con.execute.return_value.fetchone.return_value = (doc_type,) if doc_type else None
    return con


def test_delete_pdf_document():
    with (
        patch("main.get_connection", return_value=_mock_con("pdf")),
        patch("main.delete_pdf", return_value={"filename": "report.pdf", "chunks_deleted": 5, "status": "deleted"}),
    ):
        r = client.delete("/documents/report.pdf")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "deleted"
    assert body["filename"] == "report.pdf"


def test_delete_csv_document():
    with (
        patch("main.get_connection", return_value=_mock_con("csv")),
        patch("main.delete_stocks", return_value={"status": "deleted"}),
    ):
        r = client.delete("/documents/stocks.csv")
    assert r.status_code == 200
    assert r.json()["status"] == "deleted"


def test_delete_document_not_found():
    with patch("main.get_connection", return_value=_mock_con(None)):
        r = client.delete("/documents/missing.pdf")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()

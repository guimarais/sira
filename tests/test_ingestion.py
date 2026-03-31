import csv
import sqlite3
from unittest.mock import MagicMock, patch

import fitz
import numpy as np
import pytest


@pytest.fixture
def pdf_2pages(tmp_path):
    """A 2-page PDF with enough text to produce multiple chunks."""
    doc = fitz.open()
    for i in range(2):
        page = doc.new_page()
        page.insert_text((50, 72), f"Page {i + 1}: Federal Reserve rate policy. " * 80)
    path = tmp_path / "macro.pdf"
    doc.save(str(path))
    doc.close()
    return str(path)


@pytest.fixture
def corrupt_pdf(tmp_path):
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"not a pdf")
    return str(path)


@pytest.fixture
def csv_5rows(tmp_path):
    rows = [
        ["Ticker", "Company Name", "Price (USD)", "P/E Ratio", "Market Cap (B)"],
        ["AAPL", "Apple Inc.", "189.5", "28.4", "2950"],
        ["TSLA", "Tesla Inc.", "245.1", "62.1", "780"],
        ["MSFT", "Microsoft Corp.", "415.2", "35.2", "3090"],
        ["GOOGL", "Alphabet Inc.", "175.8", "24.8", "2200"],
        ["AMZN", "Amazon.com Inc.", "188.4", "41.3", "1960"],
    ]
    path = tmp_path / "stocks.csv"
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return str(path)


@pytest.fixture
def empty_csv(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("col1,col2\n")  # header only, no data rows
    return str(path)


# --- PDF ingestion ---

def test_ingest_pdf_chunk_count(pdf_2pages, tmp_path):
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = np.zeros((20, 384))
    mock_collection = MagicMock()

    with (
        patch("utils.embedder._embedder", None),
        patch("utils.embedder.SentenceTransformer", return_value=mock_embedder),
        patch("ingestion.pdf_ingestor._collection", None),
        patch("ingestion.pdf_ingestor.settings.chroma_persist_dir", str(tmp_path)),
        patch("chromadb.PersistentClient") as mock_chroma,
    ):
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        from ingestion.pdf_ingestor import ingest_pdf
        result = ingest_pdf(pdf_2pages)

    assert result["status"] == "ok"
    assert result["filename"] == "macro.pdf"
    assert result["chunks_added"] > 0
    mock_collection.upsert.assert_called_once()
    ids_passed = mock_collection.upsert.call_args.kwargs["ids"]
    assert len(ids_passed) == result["chunks_added"]


def test_ingest_pdf_metadata_has_page_numbers(pdf_2pages, tmp_path):
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = np.zeros((20, 384))
    mock_collection = MagicMock()

    with (
        patch("utils.embedder._embedder", None),
        patch("utils.embedder.SentenceTransformer", return_value=mock_embedder),
        patch("ingestion.pdf_ingestor._collection", None),
        patch("ingestion.pdf_ingestor.settings.chroma_persist_dir", str(tmp_path)),
        patch("chromadb.PersistentClient") as mock_chroma,
    ):
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        from ingestion.pdf_ingestor import ingest_pdf
        ingest_pdf(pdf_2pages)

    metadatas = mock_collection.upsert.call_args.kwargs["metadatas"]
    pages_seen = {m["page"] for m in metadatas}
    assert pages_seen == {1, 2}
    assert all("source" in m for m in metadatas)


def test_ingest_pdf_corrupt_raises(corrupt_pdf):
    with pytest.raises(fitz.FileDataError):
        from ingestion.pdf_ingestor import ingest_pdf
        ingest_pdf(corrupt_pdf)


def test_ingest_pdf_no_text_returns_no_text_status(tmp_path):
    # PDF with no extractable text (image-only page)
    doc = fitz.open()
    doc.new_page()  # blank page, no text
    path = tmp_path / "blank.pdf"
    doc.save(str(path))
    doc.close()

    from ingestion.pdf_ingestor import ingest_pdf
    result = ingest_pdf(str(path))
    assert result["status"] == "no_text"
    assert result["chunks_added"] == 0


# --- CSV ingestion ---

def test_ingest_csv_row_and_column_count(csv_5rows, tmp_path):
    db_path = str(tmp_path / "test.db")
    with patch("ingestion.csv_ingestor.settings.sqlite_path", db_path):
        from ingestion.csv_ingestor import ingest_csv
        result = ingest_csv(csv_5rows)

    assert result["status"] == "ok"
    assert result["rows_inserted"] == 5
    assert len(result["columns"]) == 5


def test_ingest_csv_column_sanitization(csv_5rows, tmp_path):
    db_path = str(tmp_path / "test.db")
    with patch("ingestion.csv_ingestor.settings.sqlite_path", db_path):
        from ingestion.csv_ingestor import ingest_csv
        result = ingest_csv(csv_5rows)

    cols = result["columns"]
    # "Price (USD)" → "price_usd", "P/E Ratio" → "p_e_ratio", etc.
    assert "price_usd" in cols
    assert "p_e_ratio" in cols
    assert "market_cap_b" in cols
    assert all(c == c.lower() for c in cols)
    assert all(" " not in c for c in cols)


def test_ingest_csv_data_written_to_sqlite(csv_5rows, tmp_path):
    db_path = str(tmp_path / "test.db")
    with patch("ingestion.csv_ingestor.settings.sqlite_path", db_path):
        from ingestion.csv_ingestor import ingest_csv
        ingest_csv(csv_5rows)

    con = sqlite3.connect(db_path)
    count = con.execute("SELECT COUNT(*) FROM stocks").fetchone()[0]
    cols_meta = con.execute("SELECT value FROM metadata WHERE key='columns'").fetchone()[0]
    con.close()

    assert count == 5
    assert "ticker" in cols_meta


def test_ingest_csv_empty_raises(empty_csv, tmp_path):
    db_path = str(tmp_path / "test.db")
    with patch("ingestion.csv_ingestor.settings.sqlite_path", db_path):
        from ingestion.csv_ingestor import ingest_csv
        with pytest.raises(ValueError, match="empty"):
            ingest_csv(empty_csv)

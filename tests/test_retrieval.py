import sqlite3
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# --- Vector retrieval ---

def _make_chroma_result(docs, sources, pages, distances):
    return {
        "documents": [docs],
        "metadatas": [[{"source": s, "page": p} for s, p in zip(sources, pages)]],
        "distances": [distances],
    }


def test_retrieve_chunks_returns_correct_shape():
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = np.zeros(384)

    mock_collection = MagicMock()
    mock_collection.count.return_value = 3
    mock_collection.query.return_value = _make_chroma_result(
        ["chunk A", "chunk B", "chunk C"],
        ["report.pdf", "report.pdf", "note.pdf"],
        [1, 2, 3],
        [0.1, 0.3, 0.5],
    )

    with (
        patch("retrieval.vector_retriever._collection", None),
        patch("retrieval.vector_retriever.settings.chroma_persist_dir", "/tmp"),
        patch("chromadb.PersistentClient") as mock_chroma,
        patch("utils.embedder.get_embedder", return_value=mock_embedder),
    ):
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        from retrieval.vector_retriever import retrieve_chunks
        results = retrieve_chunks("inflation outlook", top_k=3)

    assert len(results) == 3
    for r in results:
        assert set(r.keys()) == {"text", "source", "page", "score"}

    # Highest score first (distance 0.1 → score 0.9)
    assert results[0]["score"] > results[1]["score"]
    assert results[0]["text"] == "chunk A"


def test_retrieve_chunks_score_is_one_minus_distance():
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = np.zeros(384)

    mock_collection = MagicMock()
    mock_collection.count.return_value = 1
    mock_collection.query.return_value = _make_chroma_result(
        ["chunk"], ["f.pdf"], [1], [0.25]
    )

    with (
        patch("retrieval.vector_retriever._collection", None),
        patch("retrieval.vector_retriever.settings.chroma_persist_dir", "/tmp"),
        patch("chromadb.PersistentClient") as mock_chroma,
        patch("utils.embedder.get_embedder", return_value=mock_embedder),
    ):
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        from retrieval.vector_retriever import retrieve_chunks
        results = retrieve_chunks("test")

    assert results[0]["score"] == round(1 - 0.25, 4)


def test_retrieve_chunks_empty_collection_returns_empty_list():
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0

    with (
        patch("retrieval.vector_retriever._collection", None),
        patch("retrieval.vector_retriever.settings.chroma_persist_dir", "/tmp"),
        patch("chromadb.PersistentClient") as mock_chroma,
    ):
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        from retrieval.vector_retriever import retrieve_chunks
        results = retrieve_chunks("anything")

    assert results == []


# --- SQL retrieval ---

@pytest.fixture
def stocks_db(tmp_path):
    """Minimal SQLite DB with stocks + metadata tables."""
    db = tmp_path / "stocks.db"
    con = sqlite3.connect(str(db))
    con.execute("CREATE TABLE stocks (ticker TEXT, price_usd REAL, target_price REAL)")
    con.execute("INSERT INTO stocks VALUES ('AAPL', 189.5, 210.0)")
    con.execute("INSERT INTO stocks VALUES ('TSLA', 245.1, 280.0)")
    con.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
    con.execute("INSERT INTO metadata VALUES ('columns', 'ticker,price_usd,target_price')")
    con.commit()
    con.close()
    return str(db)


def test_retrieve_structured_returns_correct_shape(stocks_db):
    with (
        patch("retrieval.sql_retriever.settings.sqlite_path", stocks_db),
        patch("retrieval.sql_retriever._generate_sql",
              return_value="SELECT ticker, price_usd FROM stocks"),
    ):
        from retrieval.sql_retriever import retrieve_structured
        result = retrieve_structured("What are the stock prices?")

    assert "sql_generated" in result
    assert "rows" in result
    assert result["error"] is None
    assert len(result["rows"]) == 2
    assert "ticker" in result["rows"][0]


def test_retrieve_structured_retries_on_bad_sql(stocks_db):
    call_count = 0

    def mock_generate(prompt):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "SELECT nonexistent_column FROM stocks"
        return "SELECT ticker FROM stocks"

    with (
        patch("retrieval.sql_retriever.settings.sqlite_path", stocks_db),
        patch("retrieval.sql_retriever._generate_sql", side_effect=mock_generate),
    ):
        from retrieval.sql_retriever import retrieve_structured
        result = retrieve_structured("give me tickers")

    assert call_count == 2
    assert result["error"] is None
    assert result["rows"][0]["ticker"] == "AAPL"


def test_retrieve_structured_returns_error_after_two_failures(stocks_db):
    with (
        patch("retrieval.sql_retriever.settings.sqlite_path", stocks_db),
        patch("retrieval.sql_retriever._generate_sql",
              return_value="SELECT bad_col FROM stocks"),
    ):
        from retrieval.sql_retriever import retrieve_structured
        result = retrieve_structured("broken query")

    assert result["rows"] == []
    assert result["error"] is not None


def test_retrieve_structured_blocks_dangerous_sql(stocks_db):
    for dangerous in ["DROP TABLE stocks", "DELETE FROM stocks", "UPDATE stocks SET price=1"]:
        with (
            patch("retrieval.sql_retriever.settings.sqlite_path", stocks_db),
            patch("retrieval.sql_retriever._generate_sql", return_value=dangerous),
        ):
            from retrieval.sql_retriever import retrieve_structured
            result = retrieve_structured("do something bad")

        assert result["rows"] == []
        assert result["error"] is not None


def test_retrieve_structured_no_stocks_table(tmp_path):
    empty_db = str(tmp_path / "empty.db")
    sqlite3.connect(empty_db).close()  # create empty db

    with patch("retrieval.sql_retriever.settings.sqlite_path", empty_db):
        from retrieval.sql_retriever import retrieve_structured
        result = retrieve_structured("anything")

    assert result["rows"] == []
    assert "Upload a CSV" in result["error"]

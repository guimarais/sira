# SIRA — Stock Investment Research Assistant

A FastAPI application that answers investment research queries by combining semantic search over PDF documents with Text-to-SQL over structured stock data, synthesised into cited natural-language responses via Claude (Anthropic API).

## Status

**In development.** Step 1 (config) is complete. The API server and ingestion pipeline are not yet implemented.

## Requirements

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/) package manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

```bash
# 1. Install dependencies
uv sync

# 2. Configure environment
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY
```

## Running

```bash
# Start the API server (once implemented)
uv run uvicorn main:app --reload

# Run tests
uv run pytest
```

## Architecture

Queries are classified by intent (`macro` / `stock` / `hybrid`), routed to the appropriate retriever(s) in parallel, and synthesised into a single cited response by Claude.

```
POST /query
    └─→ query_router   (intent classification)
            ├─→ vector_retriever   (ChromaDB — PDF chunks)
            ├─→ sql_retriever      (SQLite — stock CSV data)
            └─→ response_builder   (Claude synthesis + citations)
```

**Ingestion endpoints:**
- `POST /upload-pdf` — extract, chunk, embed, and store PDF into ChromaDB
- `POST /upload-csv` — load stock CSV into SQLite

## Configuration

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | **Required.** Your Anthropic API key |
| `CHROMA_PERSIST_DIR` | `data/chroma` | ChromaDB persistence path |
| `SQLITE_PATH` | `data/stocks.db` | SQLite database path |
| `PDF_FOLDER` | `data/pdfs` | Directory for uploaded PDFs |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model for embeddings |
| `TOP_K` | `5` | Number of PDF chunks returned per query |

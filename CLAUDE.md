# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SIRA** (Stock Investment Research Assistant) is a FastAPI application that answers investment research queries by combining:
- **Unstructured data**: PDF documents (macroeconomic reports, strategy notes) → ChromaDB vector store
- **Structured data**: CSV files with stock metrics → SQLite via pandas

All responses cite sources. The LLM is Claude (Anthropic API) exclusively.

## Commands

Always prefix commands with `uv run` — never activate the virtualenv manually.

```bash
# Install dependencies
uv sync

# Start dev server
uv run uvicorn main:app --reload

# Run all tests (asyncio_mode = "auto" is set in pyproject.toml)
uv run pytest

# Run a single test file
uv run pytest tests/test_api.py -v

# Run a single test
uv run pytest tests/test_api.py::test_health -v

# Ad-hoc module testing
uv run python -c "from ingestion.pdf_ingestor import ingest_pdf; print(ingest_pdf('data/pdfs/sample.pdf'))"
```

## Architecture

```
main.py (FastAPI: /health, /query, /upload-pdf, /upload-csv)
    └─→ orchestration/query_router.py  (intent classification → parallel retrieval)
            ├─→ retrieval/vector_retriever.py  (semantic search on ChromaDB)
            ├─→ retrieval/sql_retriever.py     (text-to-SQL on SQLite)
            └─→ synthesis/response_builder.py  (combine context → Claude → cited answer)
```

**Ingestion** (separate from query path):
- `ingestion/pdf_ingestor.py`: extract text with PyMuPDF → sliding-window chunks (500 tokens, 50 overlap) → embed with `all-MiniLM-L6-v2` → upsert to ChromaDB
- `ingestion/csv_ingestor.py`: pandas read → sanitize columns → SQLite table

**Query flow**:
1. Classify intent: `macro` / `stock` / `hybrid`
2. Run relevant retrievers in parallel via `asyncio.gather`
3. Combine PDF chunks + SQL result into a single Claude prompt
4. Return `{answer, sources, intent}`

## Technology Stack

| Component | Choice |
|-----------|--------|
| API | FastAPI + Uvicorn |
| LLM | `claude-sonnet-4-20250514` (Anthropic SDK) |
| Vector DB | ChromaDB (local, persisted to `data/chroma/`) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`, runs in-process) |
| Relational DB | SQLite (`data/stocks.db`) |
| PDF extraction | PyMuPDF (`fitz`) |
| Config | `python-dotenv` + `.env` |

## Configuration

Copy `.env.example` to `.env` and set:
- `ANTHROPIC_API_KEY` (required)
- `CHROMA_PERSIST_DIR` (default: `data/chroma`)
- `SQLITE_PATH` (default: `data/stocks.db`)
- `PDF_FOLDER` (default: `data/pdfs`)
- `EMBED_MODEL` (default: `all-MiniLM-L6-v2`)
- `TOP_K` (default: `5`)

## Key Constraints

- **No LangChain / LangGraph** — orchestration is explicit Python (`asyncio.gather`)
- **Anthropic API only** — no other LLM providers
- **Text-to-SQL safety** — generated SQL must pass a regex allowlist; retry once on error
- **Citation required** — PDF sources as `[Source: <file>, p.<n>]`; SQL as `[Source: stocks table, SQL: <query>]`
- **English only**


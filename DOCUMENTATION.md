# SIRA — Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [API Reference](#api-reference)
4. [Ingestion Pipeline](#ingestion-pipeline)
5. [Query Pipeline](#query-pipeline)
6. [Chat History](#chat-history)
7. [Data Storage](#data-storage)
8. [Prompt Configuration](#prompt-configuration)
9. [Streamlit Frontend](#streamlit-frontend)
10. [Docker Deployment](#docker-deployment)
11. [Configuration Reference](#configuration-reference)
12. [Development Guide](#development-guide)

---

## Overview

SIRA (Stock Investment Research Assistant) answers investment research questions by combining two data sources:

- **Unstructured**: PDF documents (macro reports, strategy notes) stored as vector embeddings in ChromaDB
- **Structured**: Stock metric CSVs and XLSX files stored in SQLite

Every response cites its sources. The LLM used for all generation tasks is Claude (Anthropic API), with no other LLM providers.

---

## Architecture

```
main.py  (FastAPI)
│
├── POST /query
│     └── orchestration/query_router.py
│           ├── intent: "macro"  → retrieval/vector_retriever.py  (ChromaDB)
│           ├── intent: "stock"  → retrieval/sql_retriever.py     (SQLite)
│           ├── intent: "hybrid" → both, in parallel (asyncio.gather)
│           └── synthesis/response_builder.py  (Claude → cited answer)
│
├── POST /upload-pdf   → ingestion/pdf_ingestor.py
├── POST /upload-csv   → ingestion/csv_ingestor.py
├── POST /upload-xlsx  → ingestion/xlsx_ingestor.py
└── DELETE /documents/{filename}
      ├── pdf  → ingestion/pdf_ingestor.delete_pdf()
      └── csv  → ingestion/csv_ingestor.delete_stocks()
```

### Module map

| Path | Responsibility |
|---|---|
| `config.py` | Settings loaded from `.env` via pydantic-settings |
| `main.py` | FastAPI app, all HTTP endpoints |
| `models/schemas.py` | Pydantic request/response models |
| `ingestion/pdf_ingestor.py` | PDF text extraction, chunking, embedding, ChromaDB upsert |
| `ingestion/csv_ingestor.py` | CSV → SQLite stocks table |
| `ingestion/xlsx_ingestor.py` | XLSX sheet → temp CSV → delegates to csv_ingestor |
| `retrieval/vector_retriever.py` | Semantic search over ChromaDB |
| `retrieval/sql_retriever.py` | Text-to-SQL via Claude, executes against SQLite |
| `orchestration/query_router.py` | Intent classification, parallel retriever dispatch |
| `synthesis/response_builder.py` | Formats context, calls Claude, returns cited answer |
| `utils/db.py` | SQLite connection factory and schema initialisation |
| `utils/chunker.py` | Word-based sliding-window text chunker |
| `utils/embedder.py` | Lazy singleton for the sentence-transformers model |
| `utils/prompt_loader.py` | YAML prompt file loader with in-process cache |
| `prompts/*.yaml` | Externalised LLM prompt configurations |
| `frontend/app.py` | Streamlit chat UI |

---

## API Reference

All endpoints are documented interactively at `http://localhost:8000/docs` when the server is running.

### `GET /health`

Returns the connectivity status of ChromaDB and SQLite.

**Response**
```json
{
  "status": "ok",
  "checks": {
    "chromadb": "ok",
    "sqlite": "ok"
  }
}
```
`status` is `"degraded"` if any check fails.

---

### `POST /query`

Answer an investment research question, optionally with prior conversation history.

**Request body**
```json
{
  "question": "What is the inflation outlook?",
  "history": [
    {"role": "user",      "content": "Previous question"},
    {"role": "assistant", "content": "Previous answer"}
  ]
}
```
`history` is optional and defaults to `[]`.

**Response**
```json
{
  "answer":  "Inflation is expected to remain elevated... [Source: macro_report.pdf, p.4]",
  "sources": ["[Source: macro_report.pdf, p.4]"],
  "intent":  "macro"
}
```

| Field | Type | Description |
|---|---|---|
| `answer` | string | Natural-language response with inline citations |
| `sources` | string[] | Deduplicated list of cited sources |
| `intent` | string | Classified query type: `macro`, `stock`, or `hybrid` |

---

### `POST /upload-pdf`

Ingest a PDF document into the vector store.

**Request**: `multipart/form-data`, field `file` (.pdf)

**Response**
```json
{
  "status": "ok",
  "detail": {"filename": "report.pdf", "chunks_added": 42, "status": "ok"}
}
```
Returns `status: "no_text"` if the PDF contains no extractable text.
Returns HTTP 422 if the file is not a valid PDF.

---

### `POST /upload-csv`

Ingest a CSV file into the `stocks` SQLite table. Replaces any previously loaded data.

**Request**: `multipart/form-data`, field `file` (.csv)

**Response**
```json
{
  "status": "ok",
  "detail": {"rows_inserted": 150, "columns": ["ticker", "price_usd", "pe_ratio"], "status": "ok"}
}
```
Column names are sanitised to lowercase with underscores.

---

### `POST /upload-xlsx`

Ingest one sheet of an XLSX file. Internally converts the sheet to CSV and delegates to the CSV ingestor.

**Request**: `multipart/form-data`, field `file` (.xlsx)

**Query parameters**

| Parameter | Default | Description |
|---|---|---|
| `sheet` | `"0"` | Sheet name (e.g. `"Sheet1"`) or zero-based index (e.g. `"0"`) |

**Response**: same structure as `/upload-csv`, with an added `sheet` field in `detail`.

---

### `DELETE /documents/{filename}`

Remove a document and all associated data.

- For PDF files: deletes all ChromaDB vectors where `metadata.source == filename`, plus the registry rows.
- For CSV/XLSX files: drops the `stocks` and `metadata` SQLite tables, plus the registry row.

**Path parameter**: `filename` — exact filename as stored in the `documents` registry.

**Response**
```json
{"filename": "report.pdf", "status": "deleted"}
```
Returns HTTP 404 if the filename is not in the registry.

---

## Ingestion Pipeline

### PDF ingestion (`ingestion/pdf_ingestor.py`)

1. Open the file with PyMuPDF (`fitz`). Raises `fitz.FileDataError` for invalid files.
2. Extract text page-by-page; skip blank pages.
3. Chunk each page's text with a sliding window: **500 words**, **50-word overlap** (`utils/chunker.py`).
4. Batch-encode all chunks with `sentence-transformers` (`all-MiniLM-L6-v2`).
5. Upsert to ChromaDB collection `pdf_chunks`. Chunk IDs follow the pattern `{stem}_p{page}_c{index}`.
6. Write to the SQLite registry: one row in `documents`, one row per chunk in `chunks`.

Chunk metadata stored in ChromaDB:
```json
{"source": "report.pdf", "page": 3}
```

### CSV ingestion (`ingestion/csv_ingestor.py`)

1. Read with pandas; raise `ValueError` if empty.
2. Sanitise column names: lowercase, non-alphanumeric → underscores.
3. Drop and recreate `stocks` and `metadata` tables in SQLite (fresh load on every upload).
4. Write the column list to `metadata` for schema introspection by the SQL retriever.
5. Write one row in the `documents` registry.

### XLSX ingestion (`ingestion/xlsx_ingestor.py`)

Reads the target sheet via pandas + openpyxl, writes to a temporary CSV, delegates to `ingest_csv()`, and returns the result enriched with the resolved sheet name.

---

## Query Pipeline

### 1. Intent classification (`orchestration/query_router.py`)

The router first attempts a **keyword match** — fast and free:

- Matches stock-specific terms (ticker, price, P/E, earnings, …) → `stock`
- Matches macro terms (inflation, GDP, interest rate, …) → `macro`
- Matches both → `hybrid`
- Matches neither → falls back to Claude (`query_classification.yaml`, max 10 tokens)

### 2. Retrieval

| Intent | Vector retriever | SQL retriever |
|---|---|---|
| `macro` | ✅ | — |
| `stock` | — | ✅ |
| `hybrid` | ✅ (parallel) | ✅ (parallel) |

For `hybrid`, both retrievers run concurrently via `asyncio.gather`. A failure in one does not block the other.

### 3. Vector retrieval (`retrieval/vector_retriever.py`)

- Encodes the query with the shared embedder singleton.
- Calls `collection.query()` with `n_results=TOP_K`.
- Converts cosine distances to similarity scores: `score = 1 - distance`.
- Returns `[{text, source, page, score}]` ordered by descending score.

### 4. SQL retrieval (`retrieval/sql_retriever.py`)

- Reads column names and a sample row from SQLite for context.
- Sends a prompt to Claude (`sql_generation.yaml`) requesting a single `SELECT` statement.
- Validates the response with a regex allowlist — only `SELECT` is permitted; `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `CREATE`, `REPLACE`, `TRUNCATE` are blocked.
- Executes the SQL and returns rows as a list of dicts.
- On failure, retries once with the error message appended to the prompt (`retry_suffix`).

### 5. Response synthesis (`synthesis/response_builder.py`)

- Formats the retrieved PDF chunks and SQL rows into a single context block with inline citation tags.
- Constructs a Claude message list: prior history turns (if any) followed by the synthesis prompt.
- Calls Claude (`response_synthesis.yaml`) to produce the final answer.
- Returns `{answer, sources}`.

Citation format:
- PDF: `[Source: report.pdf, p.3]`
- SQL: `[Source: stocks table, SQL: SELECT ...]`

---

## Chat History

The `/query` endpoint accepts an optional `history` field — a list of prior `{"role", "content"}` turns. These are prepended to the Claude messages array before the current synthesis prompt, giving the model conversation context for follow-up questions.

**Client responsibility**: the caller maintains and sends history. For long conversations, summarise earlier turns client-side to stay within Claude's context window.

```json
{
  "question": "How does that affect dividend stocks?",
  "history": [
    {"role": "user",      "content": "What is the Fed rate outlook?"},
    {"role": "assistant", "content": "Rates are expected to remain at 5.25%..."}
  ]
}
```

---

## Data Storage

### Directory layout

```
data/
├── chroma/      ChromaDB vector store (persisted)
├── pdfs/        Uploaded PDF files (optional storage)
└── stocks.db    SQLite database
```

### SQLite schema

```sql
-- Ingested document registry
CREATE TABLE documents (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    filename     TEXT    NOT NULL UNIQUE,
    type         TEXT    NOT NULL,          -- 'pdf' or 'csv'
    ingested_at  TEXT    NOT NULL,          -- ISO-8601 UTC
    chunks_count INTEGER NOT NULL DEFAULT 0
);

-- ChromaDB chunk cross-reference (PDF only)
CREATE TABLE chunks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chroma_id    TEXT    NOT NULL,          -- matches ChromaDB chunk ID
    page         INTEGER,
    chunk_index  INTEGER NOT NULL
);

-- Stock data (replaced on each CSV upload)
CREATE TABLE stocks (/* columns derived from uploaded CSV */);
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);
-- metadata stores: key='columns', value='col1,col2,...'
```

The `chunks` table cascades deletes from `documents`, so removing a document row also removes its chunk rows.

### Jupyter notebooks

Three notebooks in `notebooks/` provide direct SQL exploration:

| Notebook | Contents |
|---|---|
| `01_stocks_exploration.ipynb` | Schema, row counts, descriptive stats, top-N bar chart |
| `02_document_registry.ipynb` | Ingested files, type breakdown, chunks per document |
| `03_cross_reference.ipynb` | Full `documents JOIN chunks` view, live ChromaDB lookup by `chroma_id` |

Run with `uv run jupyter notebook`.

---

## Prompt Configuration

All LLM call parameters are externalised in `prompts/*.yaml`. No model names or prompt text are hardcoded in Python.

### `prompts/sql_generation.yaml`

Controls SQL generation in `sql_retriever.py`.

| Key | Value |
|---|---|
| `model` | `claude-sonnet-4-20250514` |
| `max_tokens` | 256 |
| `system` | Instructs Claude to output only a raw SQLite SELECT |
| `user_template` | Provides table name, columns, sample row, and question |
| `retry_suffix` | Appended on retry with the previous error message |

### `prompts/query_classification.yaml`

Controls intent classification fallback in `query_router.py`.

| Key | Value |
|---|---|
| `model` | `claude-sonnet-4-20250514` |
| `max_tokens` | 10 |
| `user_template` | Returns exactly one word: `stock`, `macro`, or `hybrid` |

### `prompts/response_synthesis.yaml`

Controls the final answer generation in `response_builder.py`.

| Key | Value |
|---|---|
| `model` | `claude-sonnet-4-20250514` |
| `max_tokens` | 1024 |
| `user_template` | Provides retrieval context and query; requires inline citations |

To change a model or tune a prompt, edit the relevant YAML file — no Python changes needed.

---

## Streamlit Frontend

The chat UI lives in `frontend/app.py` and communicates with the FastAPI backend over HTTP.

### Running

```bash
cd frontend
pip install -r requirements.txt
API_URL=http://localhost:8000 streamlit run app.py
```

`API_URL` defaults to `http://localhost:8000` if not set.

### Features

- **Chat window**: messages rendered with `st.chat_message`; sources and detected intent shown in a collapsible expander below each assistant reply.
- **Conversation history**: accumulated in `st.session_state` and sent with every request so follow-up questions work correctly.
- **Sidebar — PDF upload**: upload and ingest a PDF file.
- **Sidebar — CSV / XLSX upload**: upload a stock data file; for XLSX a sheet name or index can be specified.
- **Sidebar — Clear conversation**: resets the session history.
- **Sidebar — Backend health**: on-demand health check showing ChromaDB and SQLite status.

---

## Docker Deployment

### Services

| Service | Image | Port | Description |
|---|---|---|---|
| `backend` | Built from `Dockerfile.backend` | 8000 | FastAPI + uvicorn |
| `frontend` | Built from `frontend/Dockerfile` | 8501 | Streamlit UI |

Stock and vector data are persisted on a named Docker volume (`sira_data`) mounted at `/app/data` in the backend container.

### Setup

```bash
# Copy and configure environment
cp .env.example .env
# Set ANTHROPIC_API_KEY in .env

# Build and start both services
docker compose build
docker compose up
```

- Backend API: `http://localhost:8000`
- Frontend UI: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`

### Environment variables in Docker

The backend loads all variables from `.env` via `env_file`. The frontend receives `API_URL=http://backend:8000` from `docker-compose.yml` so it routes to the backend service over the internal Docker network.

### Persistent data

```bash
# List the volume
docker volume ls | grep sira

# Remove the volume (deletes all ingested data)
docker volume rm sira_sira_data
```

---

## Configuration Reference

All settings are read from environment variables (or `.env`). Defined in `config.py` via pydantic-settings.

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | **Required.** Anthropic API key |
| `CHROMA_PERSIST_DIR` | `data/chroma` | Directory where ChromaDB persists vector data |
| `SQLITE_PATH` | `data/stocks.db` | Path to the SQLite database file |
| `PDF_FOLDER` | `data/pdfs` | Directory for PDF storage (created automatically) |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model name for embeddings |
| `TOP_K` | `5` | Number of PDF chunks returned per semantic query |

---

## Development Guide

### Commands

```bash
# Install all dependencies (including dev)
uv sync

# Start the API server with live reload
uv run uvicorn main:app --reload

# Run the full test suite
uv run pytest

# Run a single test file
uv run pytest tests/test_api.py -v

# Run a single test
uv run pytest tests/test_api.py::test_health_ok -v

# Open Jupyter for notebook exploration
uv run jupyter notebook
```

Always prefix Python commands with `uv run` — do not activate the virtual environment manually.

### Test structure

| File | Coverage |
|---|---|
| `tests/test_api.py` | All HTTP endpoints via `TestClient`; mocks LLM and DB calls |
| `tests/test_ingestion.py` | PDF and CSV ingestors; uses real SQLite in `tmp_path` |
| `tests/test_retrieval.py` | Vector and SQL retrievers; mocks ChromaDB and SQLite |

Heavy dependencies (ChromaDB, sentence-transformers, Anthropic SDK) are always mocked in tests to keep the suite fast and offline.

### Adding a new ingest format

1. Create `ingestion/<format>_ingestor.py` with an `ingest_<format>(filepath) -> dict` function.
2. Register a new `POST /upload-<format>` endpoint in `main.py` following the existing pattern (temp file, cleanup, 400/422/500 error mapping).
3. Add a corresponding `delete_<format>()` function and handle the new type in the `DELETE /documents/{filename}` endpoint.

### Changing LLM prompts

Edit the relevant file in `prompts/`. The loader caches files in memory per process, so restart the server after editing. No Python changes are needed unless adding a new prompt variable (`{placeholder}`).

### Key constraints

- **No LangChain or LangGraph** — orchestration is explicit Python with `asyncio.gather`.
- **Anthropic API only** — no other LLM providers.
- **SQL safety** — generated SQL must begin with `SELECT` and must not contain any mutation or DDL keywords. Violations are caught before execution.
- **Citations required** — the synthesis prompt instructs Claude to cite every claim. Uncited answers indicate the context was empty.
- **English only** — prompts and responses are in English.

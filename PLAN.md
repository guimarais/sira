# SIRA — Stock Investment Research Assistant
## Improved Technical Plan for Claude Code

---

### Project goal

Build a FastAPI Python application that answers investment research queries by combining semantic retrieval over PDF documents (unstructured) with Text-to-SQL over a stock data table (structured), synthesised into a cited natural-language response via the Anthropic API.

---

### Constraints (non-negotiable)

- Python only
- No LangChain, LangGraph, or similar orchestration frameworks
- Use the Anthropic API (Claude) as the sole LLM
- All responses in English
- Every response must cite its data sources

---

### Technology choices — decide these before writing any code

| Concern | Choice | Rationale |
|---|---|---|
| Vector DB | ChromaDB (local, `chromadb` package) | Zero-config, runs in-process, persists to disk |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | No extra API key; runs locally |
| Relational DB | SQLite via `sqlite3` stdlib | No server needed; CSV ingestion is one `pandas` call |
| LLM | `claude-sonnet-4-20250514` via `anthropic` SDK | Fast, low cost, strong reasoning |
| PDF extraction | `pymupdf` (`fitz`) | Best text fidelity; handles multi-column layouts |
| API framework | FastAPI + Uvicorn | Async, auto-docs, fast |
| Config | `python-dotenv` + `.env` file | `ANTHROPIC_API_KEY` and paths only |
| Package manager | `uv` | Fast, reproducible, replaces pip + venv |

---

### Prerequisites — install `uv` first

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify
uv --version
```

`uv` manages the virtual environment, Python version, and dependencies in one tool. Do not use `pip`, `venv`, or `conda` anywhere in this project.

---

### Project initialisation — run these commands first

```bash
# Create project with uv (creates pyproject.toml, .python-version, .venv)
uv init sira
cd sira

# Pin Python version
uv python pin 3.11

# Add all runtime dependencies
uv add fastapi "uvicorn[standard]" anthropic chromadb pymupdf pandas \
       sentence-transformers python-dotenv

# Add dev/test dependencies to a separate group
uv add --dev httpx pytest pytest-asyncio

# Verify the lockfile was created
ls uv.lock
```

After this, `uv.lock` is the source of truth for reproducible installs. Commit both `pyproject.toml` and `uv.lock`.

---

### Repository layout — create this structure after init

```
sira/
├── pyproject.toml           # Project metadata + dependencies (managed by uv)
├── uv.lock                  # Lockfile — commit this
├── .python-version          # Pins Python 3.11 for uv
├── .env.example             # Template — copy to .env and fill in values
├── main.py                  # FastAPI app, route definitions
├── config.py                # Settings loaded from .env
├── ingestion/
│   ├── __init__.py
│   ├── pdf_ingestor.py      # Extract text → chunk → embed → upsert to ChromaDB
│   └── csv_ingestor.py      # Read CSV → create/replace SQLite table
├── retrieval/
│   ├── __init__.py
│   ├── vector_retriever.py  # Query ChromaDB, return (chunk, source, score) tuples
│   └── sql_retriever.py     # LLM generates SQL → execute → return rows + SQL string
├── orchestration/
│   ├── __init__.py
│   └── query_router.py      # Classify intent, call retrievers, pass context to synthesiser
├── synthesis/
│   ├── __init__.py
│   └── response_builder.py  # Build prompt with context → call Claude → return answer
├── models/
│   ├── __init__.py
│   └── schemas.py           # Pydantic request/response models
├── utils/
│   ├── __init__.py
│   └── chunker.py           # Sliding-window text chunker (standalone, testable)
├── data/
│   ├── pdfs/                # Drop PDFs here; ingestor reads from this folder
│   ├── stocks.db            # SQLite file (auto-created on first CSV upload)
│   └── chroma/              # ChromaDB persistence directory (auto-created)
├── tests/
│   ├── __init__.py
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_api.py
└── README.md
```

Do **not** create a `requirements.txt`. Dependencies live in `pyproject.toml` exclusively.

---

### Running the project — all commands use `uv run`

```bash
# Start the API server (uv run activates the venv automatically)
uv run uvicorn main:app --reload

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_api.py -v

# Run any ad-hoc script
uv run python -c "from ingestion.pdf_ingestor import ingest_pdf; print(ingest_pdf('data/pdfs/sample.pdf'))"
```

Never activate the virtual environment manually. Always prefix commands with `uv run`.

---

### Implementation steps — in strict order

#### Step 1 — Config (`config.py`)

Load from `.env` using `python-dotenv`:

- `ANTHROPIC_API_KEY`
- `CHROMA_PERSIST_DIR` (default: `data/chroma`)
- `SQLITE_PATH` (default: `data/stocks.db`)
- `PDF_FOLDER` (default: `data/pdfs`)
- `EMBED_MODEL` (default: `all-MiniLM-L6-v2`)
- `TOP_K` (default: `5`)

#### Step 2 — PDF ingestion (`ingestion/pdf_ingestor.py`)

Function: `ingest_pdf(filepath: str) -> dict`

Logic:
1. Open with `pymupdf`, extract text page by page
2. Chunk via sliding window: 500 tokens, 50-token overlap — call `utils/chunker.py`
3. Embed each chunk with `sentence-transformers`
4. Upsert to ChromaDB with metadata `{source: filename, page: n}`
5. Return `{filename, chunks_added, status}`

#### Step 3 — CSV ingestion (`ingestion/csv_ingestor.py`)

Function: `ingest_csv(filepath: str) -> dict`

Logic:
1. Read with `pandas`; sanitise column names (lowercase, underscores)
2. Drop and recreate the `stocks` table in SQLite
3. Insert all rows; store column list in a `metadata` table for later use by the SQL prompt
4. Return `{rows_inserted, columns, status}`

#### Step 4 — Vector retrieval (`retrieval/vector_retriever.py`)

Function: `retrieve_chunks(query: str, top_k: int) -> list[dict]`

Each dict: `{text, source, page, score}`. Return empty list (not an error) if the collection has no documents.

#### Step 5 — Text-to-SQL retrieval (`retrieval/sql_retriever.py`)

Function: `retrieve_structured(query: str) -> dict`

Logic:
1. Read column names and one sample row from SQLite
2. Prompt Claude: *"Output only valid SQLite SQL. Table: `stocks`. Columns: {cols}."*
3. Validate generated SQL — block `DROP`, `DELETE`, `UPDATE`, `INSERT` with a regex allowlist
4. Execute; on `sqlite3.OperationalError` retry once with the error appended to the prompt
5. Return `{sql_generated, rows: list[dict], error: str|None}`

#### Step 6 — Query router (`orchestration/query_router.py`)

Function: `route_query(query: str) -> dict`

- Keyword rules first (fast, no API call): detect stock/macro/hybrid intent
- LLM fallback if ambiguous
- `hybrid` intent → call both retrievers in parallel with `asyncio.gather`
- Return `{intent, vector_results, sql_results}`

#### Step 7 — Response synthesiser (`synthesis/response_builder.py`)

Function: `build_response(query: str, vector_results, sql_results) -> dict`

Single Claude prompt combining both context types. Cite PDF chunks as `[Source: <filename>, p.<n>]` and SQL results as `[Source: stocks table, SQL: <sql>]`. Return `{answer: str, sources: list[str]}`.

#### Step 8 — FastAPI routes (`main.py`)

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Check ChromaDB + SQLite availability |
| `POST` | `/query` | `{question: str}` → `{answer, sources, intent}` |
| `POST` | `/upload-pdf` | `multipart/form-data` PDF → ingest stats |
| `POST` | `/upload-csv` | `multipart/form-data` CSV → ingest stats |

#### Step 9 — Error handling

- `pdf_ingestor`: catch `fitz.FileDataError` for corrupted PDFs
- `sql_retriever`: retry once on bad SQL; return `{rows: [], error: "..."}` on second failure
- `main.py`: all routes in try/except; return `HTTP 500` with `{detail: str}` — never bubble HTML exceptions

#### Step 10 — Tests (`tests/`)

- `test_ingestion.py`: ingest a 2-page test PDF and 5-row CSV; assert chunk/row counts
- `test_retrieval.py`: mock ChromaDB and SQLite; assert return shapes
- `test_api.py`: use `httpx.AsyncClient` with FastAPI's test client; hit all four endpoints

---

### `pyproject.toml` — reference structure

```toml
[project]
name = "sira"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.111",
    "uvicorn[standard]>=0.29",
    "anthropic>=0.28",
    "chromadb>=0.5",
    "pymupdf>=1.24",
    "pandas>=2.2",
    "sentence-transformers>=3.0",
    "python-dotenv>=1.0",
]

[dependency-groups]
dev = [
    "httpx>=0.27",
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

---

### `.env.example`

```
ANTHROPIC_API_KEY=sk-ant-...
CHROMA_PERSIST_DIR=data/chroma
SQLITE_PATH=data/stocks.db
PDF_FOLDER=data/pdfs
EMBED_MODEL=all-MiniLM-L6-v2
TOP_K=5
```

---

### README quick-start section

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and enter the project
git clone <repo-url> && cd sira

# 3. Install all dependencies (reads uv.lock — no internet needed if lock is fresh)
uv sync

# 4. Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 5. Start the server
uv run uvicorn main:app --reload

# 6. Ingest sample data
curl -X POST http://localhost:8000/upload-pdf -F "file=@data/pdfs/macro_report.pdf"
curl -X POST http://localhost:8000/upload-csv -F "file=@data/stocks_sample.csv"

# 7. Query
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the target price of Tesla and what do macro trends say about growth stocks?"}'
```

---

### Evaluation criteria mapped to implementation

| Criterion | Where it lives |
|---|---|
| Code quality | One function per file, type hints everywhere, docstrings on public functions |
| Functionality | Steps 4–7 cover all three query types |
| Error handling | Step 9 — explicit, minimal, never silent |
| Creativity | Hybrid routing + single-prompt synthesis with inline SQL citation |
| Performance | `asyncio.gather` for hybrid queries; ChromaDB batched upsert for bulk PDF ingestion |
| Reproducibility | `uv.lock` committed; `uv sync` gives identical envs on any machine |
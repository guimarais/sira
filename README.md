# SIRA — Stock Investment Research Assistant

SIRA is a FastAPI application that answers investment research questions by combining semantic search over PDF documents with text-to-SQL over structured stock data, synthesised into cited natural-language responses by Claude (Anthropic API). A Streamlit chat UI is included for interactive use.

> Full documentation: [DOCUMENTATION.md](DOCUMENTATION.md)

## Requirements

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/) package manager
- An [Anthropic API key](https://console.anthropic.com/)

## Quick Start

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and install dependencies
git clone <repo-url> && cd sira
uv sync

# 3. Configure environment
cp .env.example .env
# Open .env and set ANTHROPIC_API_KEY=sk-ant-...

# 4. Start the API server
uv run uvicorn main:app --reload
# API available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

**Start the Streamlit frontend** (optional, separate terminal):

```bash
cd frontend && pip install -r requirements.txt
API_URL=http://localhost:8000 streamlit run app.py
# UI available at http://localhost:8501
```

## Ingest data and ask questions

```bash
# Upload a PDF report
curl -X POST http://localhost:8000/upload-pdf \
     -F "file=@/path/to/report.pdf"

# Upload stock data as CSV
curl -X POST http://localhost:8000/upload-csv \
     -F "file=@/path/to/stocks.csv"

# Ask a question (stateless)
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the inflation outlook?"}'

# Ask a follow-up (with conversation history)
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{
       "question": "How does that affect tech valuations?",
       "history": [
         {"role": "user",      "content": "What is the inflation outlook?"},
         {"role": "assistant", "content": "<answer from previous turn>"}
       ]
     }'
```

## Run tests

```bash
uv run pytest
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | **Required.** Anthropic API key |
| `CHROMA_PERSIST_DIR` | `data/chroma` | ChromaDB persistence directory |
| `SQLITE_PATH` | `data/stocks.db` | SQLite database file |
| `PDF_FOLDER` | `data/pdfs` | Uploaded PDF storage directory |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers embedding model |
| `TOP_K` | `5` | PDF chunks returned per query |

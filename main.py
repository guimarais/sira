import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import fitz
from fastapi import FastAPI, HTTPException, UploadFile

from config import settings
from ingestion.csv_ingestor import ingest_csv
from ingestion.pdf_ingestor import ingest_pdf
from models.schemas import IngestResponse, QueryRequest, QueryResponse
from orchestration.query_router import route_query
from synthesis.response_builder import build_response


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    yield


app = FastAPI(title="SIRA", description="Stock Investment Research Assistant", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    """Check that data directories and the SQLite file are reachable."""
    import sqlite3

    import chromadb

    checks: dict[str, str] = {}

    try:
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        client.list_collections()
        checks["chromadb"] = "ok"
    except Exception as exc:
        checks["chromadb"] = f"error: {exc}"

    try:
        con = sqlite3.connect(settings.sqlite_path)
        con.execute("SELECT 1")
        con.close()
        checks["sqlite"] = "ok"
    except Exception as exc:
        checks["sqlite"] = f"error: {exc}"

    status = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    return {"status": status, "checks": checks}


@app.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest) -> QueryResponse:
    """Answer an investment research question using PDF and stock data."""
    try:
        routed = await route_query(body.question)
        result = build_response(
            body.question,
            routed["vector_results"],
            routed["sql_results"],
        )
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            intent=routed["intent"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/upload-pdf", response_model=IngestResponse)
async def upload_pdf(file: UploadFile) -> IngestResponse:
    """Ingest a PDF document into the vector store."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        detail = ingest_pdf(tmp_path)
    except fitz.FileDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return IngestResponse(status="ok", detail=detail)


@app.post("/upload-csv", response_model=IngestResponse)
async def upload_csv(file: UploadFile) -> IngestResponse:
    """Ingest a CSV file into the stocks SQLite table."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV.")

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="wb") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        detail = ingest_csv(tmp_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return IngestResponse(status="ok", detail=detail)

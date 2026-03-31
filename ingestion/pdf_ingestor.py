from datetime import datetime, timezone
from pathlib import Path

import chromadb
import fitz  # PyMuPDF

from config import settings
from retrieval.vector_retriever import invalidate_bm25
from utils.chunker import chunk_text
from utils.db import get_connection
from utils.embedder import get_embedder

_collection: chromadb.Collection | None = None


def _get_collection() -> chromadb.Collection:
    """Lazily initialize and return the ChromaDB collection."""
    global _collection
    if _collection is None:
        settings.ensure_dirs()
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        _collection = client.get_or_create_collection("pdf_chunks")
    return _collection


def ingest_pdf(filepath: str) -> dict:
    """Extract text from a PDF, chunk it, embed it, and upsert to ChromaDB.

    Also records the document and each chunk in the SQLite registry so they
    can be cross-referenced and deleted by filename.

    Args:
        filepath: Absolute or relative path to the PDF file.

    Returns:
        dict with keys: filename, chunks_added, status.

    Raises:
        fitz.FileDataError: If the file is not a valid PDF.
    """
    path = Path(filepath)

    try:
        doc = fitz.open(filepath)
    except fitz.FileDataError as exc:
        raise fitz.FileDataError(f"Cannot open PDF '{path.name}': {exc}") from exc

    # Extract text per page, track page numbers for metadata
    page_texts: list[tuple[int, str]] = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            page_texts.append((page_num, text))
    doc.close()

    if not page_texts:
        return {"filename": path.name, "chunks_added": 0, "status": "no_text"}

    embedder = get_embedder()
    collection = _get_collection()

    ids: list[str] = []
    texts: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict] = []
    # (chroma_id, page, chunk_index) — used to populate the chunks registry table
    chunk_records: list[tuple[str, int, int]] = []

    for page_num, text in page_texts:
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"{path.stem}_p{page_num}_c{chunk_idx}"
            ids.append(chunk_id)
            texts.append(chunk)
            metadatas.append({"source": path.name, "page": page_num})
            chunk_records.append((chunk_id, page_num, chunk_idx))

    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

    # Upsert all chunks for this PDF in a single call to ChromaDB
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    # Write document and chunk records to the SQLite registry
    now = datetime.now(timezone.utc).isoformat()
    con = get_connection()
    try:
        con.execute(
            "INSERT OR REPLACE INTO documents (filename, type, ingested_at, chunks_count)"
            " VALUES (?, 'pdf', ?, ?)",
            (path.name, now, len(ids)),
        )
        doc_id = con.execute(
            "SELECT id FROM documents WHERE filename = ?", (path.name,)
        ).fetchone()[0]
        con.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        con.executemany(
            "INSERT INTO chunks (document_id, chroma_id, page, chunk_index) VALUES (?, ?, ?, ?)",
            [(doc_id, cid, page, cidx) for cid, page, cidx in chunk_records],
        )
        con.commit()
    finally:
        con.close()

    invalidate_bm25()
    return {"filename": path.name, "chunks_added": len(ids), "status": "ok"}


def delete_pdf(filename: str) -> dict:
    """Remove all ChromaDB chunks and registry records for a PDF document.

    Args:
        filename: The exact filename as stored in the documents registry.

    Returns:
        dict with keys: filename, chunks_deleted, status.
    """
    collection = _get_collection()

    # Count before deletion so we can report how many were removed
    existing = collection.get(where={"source": filename}, include=[])
    chunks_deleted = len(existing["ids"])

    if chunks_deleted:
        collection.delete(where={"source": filename})

    # Delete the documents row; chunks rows cascade via FK
    con = get_connection()
    try:
        con.execute("DELETE FROM documents WHERE filename = ?", (filename,))
        con.commit()
    finally:
        con.close()

    invalidate_bm25()
    return {"filename": filename, "chunks_deleted": chunks_deleted, "status": "deleted"}

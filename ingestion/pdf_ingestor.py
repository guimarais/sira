from pathlib import Path

import chromadb
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

from config import settings
from utils.chunker import chunk_text

_embedder: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None


def _get_embedder() -> SentenceTransformer:
    """Lazily initialize and return the SentenceTransformer embedder."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(settings.embed_model)
    return _embedder


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

    # Initialize embedder and collection once, reuse for all chunks
    embedder = _get_embedder()
    collection = _get_collection()

    # Prepare data for upsert: generate unique IDs, chunk text, and create metadata
    ids: list[str] = []
    documents: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict] = []

    # Process each page's text, chunk it, and prepare for embedding and upsert
    for page_num, text in page_texts:
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"{path.stem}_p{page_num}_c{chunk_idx}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({"source": path.name, "page": page_num})

    embeddings = embedder.encode(documents, show_progress_bar=False).tolist()

    # Upsert all chunks for this PDF in a single call to ChromaDB
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return {"filename": path.name, "chunks_added": len(ids), "status": "ok"}

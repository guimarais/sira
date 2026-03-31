"""
Module for retrieving relevant PDF chunks based on a natural-language query.
"""
import chromadb
from config import settings
from utils.embedder import get_embedder

_collection: chromadb.Collection | None = None


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        settings.ensure_dirs()
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        _collection = client.get_or_create_collection("pdf_chunks")
    return _collection


def retrieve_chunks(query: str, top_k: int | None = None) -> list[dict]:
    """Semantic search over ingested PDF chunks.

    Args:
        query: The natural-language query to embed and search.
        top_k: Number of results to return. Defaults to settings.top_k.

    Returns:
        List of dicts with keys: text, source, page, score.
        Ordered by relevance (highest score first).
        Returns an empty list if the collection has no documents.
    """
    if top_k is None:
        top_k = settings.top_k

    collection = _get_collection()
    if collection.count() == 0:
        return []

    query_embedding = get_embedder().encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append(
            {
                "text": doc,
                "source": meta["source"],
                "page": meta["page"],
                "score": round(1 - distance, 4),  # cosine distance → similarity
            }
        )

    return chunks

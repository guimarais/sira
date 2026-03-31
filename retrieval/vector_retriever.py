"""
Hybrid retrieval over ingested PDF chunks.

Combines dense (semantic) search via ChromaDB with sparse (BM25) keyword
search. Results are fused using Reciprocal Rank Fusion (RRF).
"""
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi

from config import settings
from utils.embedder import get_embedder

_collection: chromadb.Collection | None = None

# Cached BM25 state: (ids, texts, metadatas, BM25Okapi). Set to None to force
# a rebuild on the next query — call invalidate_bm25() after ingest/delete.
_bm25_state: tuple[list[str], list[str], list[dict], BM25Okapi] | None = None

# RRF constant — 60 is the standard value from the original RRF paper.
_RRF_K = 60


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        settings.ensure_dirs()
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        _collection = client.get_or_create_collection("pdf_chunks")
    return _collection


def invalidate_bm25() -> None:
    """Discard the cached BM25 index so it is rebuilt on the next query.

    Call this after any ingest or delete operation that changes the corpus.
    """
    global _bm25_state
    _bm25_state = None


def _get_bm25_state(
    collection: chromadb.Collection,
) -> tuple[list[str], list[str], list[dict], BM25Okapi]:
    """Return the cached BM25 index, building it from ChromaDB if necessary."""
    global _bm25_state
    if _bm25_state is None:
        result = collection.get(include=["documents", "metadatas"])
        ids: list[str] = result["ids"]
        texts: list[str] = result["documents"]
        metadatas: list[dict] = result["metadatas"]
        tokenized = [t.lower().split() for t in texts]
        _bm25_state = (ids, texts, metadatas, BM25Okapi(tokenized))
    return _bm25_state


def _rrf_merge(
    ranked_lists: list[list[Any]],
    id_key: str = "id",
    k: int = _RRF_K,
) -> dict[str, float]:
    """Apply Reciprocal Rank Fusion to multiple ranked lists.

    Args:
        ranked_lists: Each inner list is a sequence of dicts containing id_key,
            ordered from most to least relevant.
        id_key: The key in each dict that holds the document identifier.
        k: RRF constant (default 60).

    Returns:
        Dict mapping document ID to its combined RRF score.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            doc_id = item[id_key]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def retrieve_chunks(query: str, top_k: int | None = None) -> list[dict]:
    """Hybrid search over ingested PDF chunks using BM25 + semantic + RRF.

    Performs dense (semantic) and sparse (BM25) retrieval independently, then
    merges the ranked lists with Reciprocal Rank Fusion. The BM25 index is
    built once from all stored chunks and cached until invalidated.

    Args:
        query: The natural-language query.
        top_k: Number of results to return. Defaults to settings.top_k.

    Returns:
        List of dicts with keys: text, source, page, score (RRF score).
        Ordered by descending relevance. Empty if the collection is empty.
    """
    if top_k is None:
        top_k = settings.top_k

    collection = _get_collection()
    if collection.count() == 0:
        return []

    candidate_k = min(top_k * 2, collection.count())

    # ── Dense retrieval ───────────────────────────────────────────────────────
    query_embedding = get_embedder().encode(query).tolist()
    dense_raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_k,
        include=["documents", "metadatas", "distances"],
    )
    dense_ranked = [
        {"id": doc_id, "text": doc, "source": meta["source"], "page": meta["page"]}
        for doc_id, doc, meta in zip(
            dense_raw["ids"][0],
            dense_raw["documents"][0],
            dense_raw["metadatas"][0],
        )
    ]

    # ── Sparse retrieval (BM25) ───────────────────────────────────────────────
    all_ids, all_texts, all_metas, bm25 = _get_bm25_state(collection)
    bm25_scores = bm25.get_scores(query.lower().split())
    top_sparse_indices = sorted(
        range(len(all_ids)), key=lambda i: -bm25_scores[i]
    )[:candidate_k]
    sparse_ranked = [
        {
            "id": all_ids[i],
            "text": all_texts[i],
            "source": all_metas[i]["source"],
            "page": all_metas[i]["page"],
        }
        for i in top_sparse_indices
    ]

    # ── RRF fusion ────────────────────────────────────────────────────────────
    rrf_scores = _rrf_merge([dense_ranked, sparse_ranked])

    # Build a lookup so we can reconstruct chunk data from any fused ID
    doc_lookup: dict[str, dict] = {}
    for item in dense_ranked + sparse_ranked:
        doc_lookup.setdefault(item["id"], item)

    final = sorted(rrf_scores.items(), key=lambda x: -x[1])[:top_k]
    return [
        {
            "text": doc_lookup[doc_id]["text"],
            "source": doc_lookup[doc_id]["source"],
            "page": doc_lookup[doc_id]["page"],
            "score": round(score, 4),
        }
        for doc_id, score in final
    ]

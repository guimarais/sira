"""
Module for managing the shared SentenceTransformer instance used for embedding text.
"""
from sentence_transformers import SentenceTransformer

from config import settings

_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    """Return the shared SentenceTransformer instance (lazy-initialized)."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(settings.embed_model)
    return _embedder

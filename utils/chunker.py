def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping token-approximate chunks.

    Uses whitespace-split words as a token proxy (fast, no tokenizer dependency).
    Each chunk is at most `chunk_size` words; consecutive chunks overlap by `overlap` words.
    """
    words = text.split()
    if not words:
        return []

    # Handle edge case where text is shorter than chunk size
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap

    return chunks
